"""Reproduce one explicit fixed recipe using training data only.

Training metadata stays in --work-dir. --output contains only portable
inference manifests, their referenced assets, and config.json. Existing
directories are never reused or overwritten by this reproduction command.
"""
from __future__ import annotations
import os
import sys
sys.dont_write_bytecode=True
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import time
import zipfile
from optimization_finalists import BRANCHES,branch_fit_count,source_closure,safe_name

ROOT=Path(__file__).resolve().parent


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def inside(root,relative):
    path=Path(relative)
    if path.is_absolute() or ".." in path.parts:raise ValueError("Portable relative asset path required")
    target=(root/path).resolve()
    if root not in target.parents:raise ValueError("Asset path escapes branch directory")
    return target


def source_hashes():
    return {**source_closure(),Path(__file__).name:sha(__file__)}


def validate_recipe(path):
    recipe=read(path)
    if set(recipe)!={"name","weights"}:raise ValueError("Recipe must contain exactly name and weights")
    safe_name(recipe["name"])
    weights=recipe["weights"]
    if not isinstance(weights,dict) or not weights:raise ValueError("At least one fixed branch is required")
    for branch,weight in weights.items():
        if branch not in BRANCHES:raise ValueError("Unknown branch: "+str(branch))
        if isinstance(weight,bool) or not isinstance(weight,(int,float)) or not math.isfinite(weight) or weight<=0:
            raise ValueError("Branch weights must be finite and positive")
    if abs(sum(weights.values())-1)>1e-9:raise ValueError("Branch weights must sum to one")
    return recipe


def child_env():
    env=os.environ.copy()
    env.update(PYTHONUTF8="1",PYTHONIOENCODING="utf-8",PYTHONDONTWRITEBYTECODE="1")
    for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        env[key]="4"
    return env


def complete_branch(branch,directory):
    adapter,name,_=BRANCHES[branch]
    path=directory/"manifest.json";manifest=read(path)
    if manifest.get("branch")!=name or read(directory/"progress.json").get("stage")!="complete":
        raise RuntimeError("Incomplete or wrong branch: "+branch)
    if adapter=="core":
        if manifest.get("format")!="mp2-core-branch-v1":raise RuntimeError("Wrong core manifest")
        files=[(m["file"],m["sha256"]) for m in manifest["members"]]
        sources=manifest["source_sha256"]
    else:
        expected="mp2_sequence_branch_v1" if adapter=="sequence" else "mp2_structure_branch_v1"
        if manifest.get("format")!=expected:raise RuntimeError("Wrong branch manifest")
        files=[(manifest["asset_path"],manifest["asset_sha256"])]
        sources=manifest["inference_sources_sha256"]
    for name,value in sources.items():
        if sha(ROOT/name)!=value:raise RuntimeError("Source changed during reproduction: "+name)
    for name,value in files:
        p=inside(directory,name)
        if p.suffix.lower()!=".npz" or sha(p)!=value:raise RuntimeError("Invalid branch asset")
    count=int(manifest.get("model_fit_calls",1))
    if count!=branch_fit_count(branch):raise RuntimeError("Unexpected learner count")
    return path,manifest,files,count


def pack(recipe,branches,output):
    output.mkdir(parents=True);components=[]
    for branch,weight in recipe["weights"].items():
        source=branches[branch];manifest,_,files,_=complete_branch(branch,source)
        directory=inside(output,"branches/"+safe_name(branch));directory.mkdir(parents=True)
        destination=directory/"manifest.json";shutil.copyfile(manifest,destination)
        for name,_ in files:
            target=inside(directory,name);target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(inside(source,name),target)
        components.append(dict(adapter=BRANCHES[branch][0],manifest=destination.relative_to(output).as_posix(),
            manifest_sha256=sha(destination),weight=float(weight)))
    dump(output/"config.json",dict(format="mp2-integrated-verifier-v1",components=components))
    hashes={};sizes={}
    for path in sorted(output.rglob("*")):
        if not path.is_file():continue
        name=path.relative_to(output).as_posix();hashes[name]=sha(path)
        if path.suffix.lower()==".npz":
            with zipfile.ZipFile(path) as archive:sizes[name]=sum(x.file_size for x in archive.infolist())
        else:sizes[name]=path.stat().st_size
    if sum(sizes.values())>16*2**20:raise RuntimeError("Reproduced whole checkpoint exceeds16MiB")
    return dict(sha256=hashes,file_uncompressed_bytes=sizes,uncompressed_bytes=sum(sizes.values()))


def run(recipe_path,output,work,core_python,sequence_python):
    recipe_path=Path(recipe_path).resolve();recipe=validate_recipe(recipe_path)
    output=Path(output).resolve();work=Path(work).resolve()
    if output.exists() or work.exists():raise FileExistsError("Both output and work directory must be new")
    if output==work or output in work.parents or work in output.parents:
        raise ValueError("Keep checkpoint and training metadata in separate sibling directories")
    if ROOT.parent.parent not in work.parents:
        raise ValueError("Current training adapters require work-dir inside the task project directory")
    core_python=Path(core_python).resolve();sequence_python=Path(sequence_python).resolve()
    for runtime in (core_python,sequence_python):
        if not runtime.is_file():raise FileNotFoundError(runtime)
    work.mkdir(parents=True);(work/"branches").mkdir();(work/"logs").mkdir()
    sources=source_hashes();started=time.perf_counter()
    plan=dict(recipe=recipe,recipe_sha256=sha(recipe_path),source_sha256=sources,
        planned_model_fits=sum(branch_fit_count(b) for b in recipe["weights"]),
        core_python=str(core_python),sequence_python=str(sequence_python),threads=4,
        created_utc=datetime.now(timezone.utc).isoformat(),data_scope="supplied training split only",
        validation_evaluations=0,test_access=False,
        output_scope="Only inference manifests/assets/config; all logs, fitting scores and plans stay in work-dir",
        epoch_metadata="Sequence/structural adapters require the compact registered source-study protocol/summary/fit-record/hash files; no historical model assets are used")
    dump(work/"protocol.json",plan);dump(work/"recipe.json",recipe)
    records=[];branches={};completed=0
    def progress(stage,detail):
        dump(work/"progress.json",dict(stage=stage,completed_fits=completed,planned_fits=plan["planned_model_fits"],
            detail=detail,elapsed_seconds=time.perf_counter()-started,updated_utc=datetime.now(timezone.utc).isoformat()))
    progress("preflight","Checking prescribed branch metadata without fitting")
    try:
        # Fail before any learner fit if a required fixed epoch-provenance file
        # is missing. These adapter modes inspect metadata only.
        for branch in recipe["weights"]:
            adapter,name,environment=BRANCHES[branch]
            if adapter=="core":continue
            python=core_python if environment=="venv" else sequence_python
            mode="--describe" if adapter=="sequence" else "--plan"
            command=[str(python),"-B",str(ROOT/f"optimization_{adapter}_refit.py"),mode,"--branch",name]
            result=subprocess.run(command,cwd=ROOT,env=child_env(),capture_output=True,text=True,encoding="utf-8",
                creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
            dump(work/"logs"/(safe_name(branch)+"_preflight.json"),dict(command=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
            if result.returncode:raise RuntimeError("Missing/inconsistent branch provenance: "+branch)
        for branch in recipe["weights"]:
            adapter,name,environment=BRANCHES[branch]
            python=core_python if environment=="venv" else sequence_python
            directory=work/"branches"/safe_name(branch)
            command=[str(python),"-B","-u",str(ROOT/f"optimization_{adapter}_refit.py")]
            if adapter!="core":command.append("--fit")
            command.extend(("--branch",name,"--output",str(directory)))
            progress("fitting","Fresh full-training branch: "+branch)
            tick=time.perf_counter();log=work/"logs"/(safe_name(branch)+".log")
            with log.open("wb") as stream:
                result=subprocess.run(command,cwd=ROOT,env=child_env(),stdout=stream,stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
            if result.returncode:
                known=read(directory/"progress.json").get("completed_fits",0) if (directory/"progress.json").exists() else 0
                completed+=int(known)
                records.append(dict(branch=branch,status="failed",known_completed_fits=int(known),command=command,returncode=result.returncode,log=str(log)))
                raise RuntimeError("Refit failed; preserve training work directory: "+branch)
            path,manifest,_,count=complete_branch(branch,directory)
            completed+=count;branches[branch]=directory
            records.append(dict(branch=branch,adapter=adapter,model_fit_calls=count,seconds=time.perf_counter()-tick,
                manifest_sha256=sha(path),command=command,returncode=0,log=log.relative_to(work).as_posix()))
            dump(work/"fit_records.json",records)
        if source_hashes()!=sources:raise RuntimeError("Source changed during reproduction")
        if completed!=plan["planned_model_fits"]:raise RuntimeError("Incomplete reproduction fit count")
        artifact=pack(recipe,branches,output)
        summary=dict(status="complete",recipe=recipe,model_fit_calls=completed,branch_count=len(branches),
            checkpoint=str(output),training_work_directory=str(work),source_sha256=sources,
            checkpoint_inventory=artifact,seconds=time.perf_counter()-started,validation_evaluations=0,test_access=False,
            interpretation="Independent fixed-recipe full-training reproduction; no recipe selection or held-out evaluation")
        dump(work/"summary.json",summary);progress("complete","Fresh checkpoint packed; no validation/test evaluation")
        print(json.dumps(dict(status="complete",model_fit_calls=completed,checkpoint=str(output),uncompressed_bytes=artifact["uncompressed_bytes"]),ensure_ascii=False))
    except Exception as error:
        dump(work/"fit_records.json",records)
        dump(work/"failure.json",dict(status="failed",error=repr(error),known_completed_model_fits=completed,
            seconds=time.perf_counter()-started,validation_evaluations=0,test_access=False))
        progress("failed",repr(error));raise


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--work-dir",type=Path,required=True)
    parser.add_argument("--core-python",type=Path,default=Path(sys.executable))
    parser.add_argument("--sequence-python",type=Path,default=Path(sys.executable))
    args=parser.parse_args();run(args.recipe,args.output,args.work_dir,args.core_python,args.sequence_python)
