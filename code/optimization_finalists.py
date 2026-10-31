"""Prepare resource-qualified full-training finalists from a frozen fusion menu.

--register-only records the source/policy boundary without fits or data reads.
--run performs explicit full-training subprocess refits and training-fixture
resource checks. This driver never evaluates validation or test.
"""
from __future__ import annotations
import os
import sys
sys.dont_write_bytecode=True
for _name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_name]="4"
import argparse
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
import zipfile

ROOT=Path(__file__).resolve().parent
TOP=ROOT.parent.parent
OUT=ROOT/"runs"/"optimization_finalists"
FUSION=ROOT/"runs"/"optimization_fusion"
POLICY=ROOT.parent/"docs"/"optimization_selection_policy.md"
FIXTURE=TOP/".local"/"optimization_resource_fixture"
LIMIT_BYTES=16*1024*1024
LIMIT_SECONDS=25.
BRANCHES={
    "A":("core","A","venv"),"K":("core","K","venv"),"Q":("core","Q","venv"),
    "P":("core","P","venv"),"C":("core","C","venv"),
    "A14_5":("core","A5_epoch14","venv"),"Aema_5":("core","A5_ema40","venv"),
    "Anoise_5":("core","A5_noise_ema40","venv"),"Across_3":("core","A_D1D2","venv"),
    "Qcross":("core","Q_D1D2","venv"),"Ccross":("core","C_D1D2","venv"),
    "Asubset_3":("core","drop_one_half","venv"),
    "roles_cnn":("sequence","headtail_roles","sequenceenv"),
    "allq_cnn":("sequence","control_allq","sequenceenv"),"aux_cnn":("sequence","token_aux01","sequenceenv"),
    "token_attention":("sequence","token_attention","sequenceenv"),"set_attention":("sequence","token_set_attention","sequenceenv"),
    "digit_cnn":("sequence","digit_text_control","sequenceenv"),"corruption_cnn":("sequence","reference_corruption01","sequenceenv"),
    "node_mlp":("structure","node_mlp","sequenceenv"),"equation_gcn":("structure","equation_gcn","sequenceenv"),
    "cross_tree":("structure","cross_tree","venv"),
}
SOURCE_ROOTS=("optimization_finalists.py","optimized_selection.py","optimization_core_refit.py",
    "optimization_sequence_refit.py","optimization_structure_refit.py","predict.py","score_predictions.py","common.py")

def stamp():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");tmp.replace(path)
def safe_name(value):
    if not isinstance(value,str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*",value):raise ValueError("Unsafe recipe/branch name")
    return value
def inside(root,relative):
    root=Path(root).resolve();relative=Path(relative)
    if relative.is_absolute() or ".." in relative.parts:raise ValueError("Portable relative path required")
    target=(root/relative).resolve()
    if root not in target.parents:raise ValueError("Path escapes intended directory")
    return target
def runtime(name):
    path=TOP/".local"/name/"Scripts"/"python.exe"
    if not path.is_file():raise FileNotFoundError("Missing runtime: "+str(path))
    return path

def source_closure():
    """Static local-module closure, including lazy imports used by adapters."""
    found=set();todo=list(SOURCE_ROOTS)
    while todo:
        filename=todo.pop()
        if filename in found:continue
        path=ROOT/filename
        if not path.is_file():raise FileNotFoundError(path)
        found.add(filename)
        tree=ast.parse(path.read_text(encoding="utf-8"),filename=filename)
        for node in ast.walk(tree):
            names=[]
            if isinstance(node,ast.Import):names=[x.name.split('.')[0] for x in node.names]
            elif isinstance(node,ast.ImportFrom) and node.module:names=[node.module.split('.')[0]]
            for name in names:
                local=name+".py"
                if (ROOT/local).is_file() and local not in found:todo.append(local)
    return {name:sha(ROOT/name) for name in sorted(found)}

def validate_registration(protocol):
    if sha(POLICY)!=protocol["selection_policy_sha256"]:raise RuntimeError("Selection policy changed")
    if sha(FUSION/"protocol.json")!=protocol["fusion_protocol_sha256"]:raise RuntimeError("Fusion protocol changed")
    if source_closure()!=protocol["source_sha256"]:raise RuntimeError("Registered source closure changed")
    if "repair" in protocol:
        repair=protocol["repair"]
        if sha(OUT/"repair_logging"/"core_compatibility.json")!=repair["core_compatibility_sha256"]:
            raise RuntimeError("Core compatibility record changed")
        core_compatibility()
        for filename,value in repair["driver_archive_hashes"].items():
            if sha(inside(OUT/"repair_logging",filename))!=value:raise RuntimeError("Preserved driver evidence changed")

def core_compatibility():
    folder=OUT/"repair_logging";value=read(folder/"core_compatibility.json")
    if value.get("inference_ast_unchanged") is not True or value.get("recovered_model_fits")!=1:
        raise RuntimeError("Logging repair lacks required compatibility proof")
    old=folder/"optimization_core_refit_before.py";new=ROOT/"optimization_core_refit.py"
    if sha(old)!=value["old_sha256"] or sha(new)!=value["new_sha256"]:raise RuntimeError("Core compatibility source mismatch")
    evidence=inside(folder,value["recovery_evidence_path"])
    if sha(evidence)!=value["recovery_evidence_sha256"]:raise RuntimeError("P recovery evidence changed")
    trees=[ast.parse(p.read_text(encoding="utf-8")) for p in (old,new)]
    for name in ("_manifest","_raw_member","score_branch","predict","digest"):
        funcs=[next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name) for t in trees]
        if ast.dump(funcs[0],include_attributes=False)!=ast.dump(funcs[1],include_attributes=False):raise RuntimeError("Core inference AST changed")
    imports=[[ast.dump(n,include_attributes=False) for n in t.body if isinstance(n,(ast.Import,ast.ImportFrom))] for t in trees]
    if imports[0]!=imports[1]:raise RuntimeError("Core inference imports changed")
    return value

def source_compatible(filename,expected):
    actual=sha(ROOT/filename)
    if actual==expected:return True
    if filename!="optimization_core_refit.py":return False
    protocol=read(OUT/"protocol.json")
    if "repair" not in protocol:return False
    if sha(OUT/"repair_logging"/"core_compatibility.json")!=protocol["repair"]["core_compatibility_sha256"]:return False
    value=core_compatibility()
    return expected==value["old_sha256"] and actual==value["new_sha256"]

def register():
    OUT.mkdir(parents=True,exist_ok=True)
    for name in ("venv","sequenceenv","checkenv"):runtime(name)
    f=read(FUSION/"protocol.json")
    if set(f["branches"])!=set(BRANCHES):raise RuntimeError("Fusion/refit branch mapping mismatch")
    if f["number_of_recipes"]!=48:raise RuntimeError("Unexpected frozen menu")
    expected={"purpose":"Resource qualification of at most two fixed-menu finalists; no validation/test",
        "selection_policy_sha256":sha(POLICY),"fusion_protocol_sha256":sha(FUSION/"protocol.json"),
        "source_sha256":source_closure(),"branch_mapping":BRANCHES,
        "refit_cache":"Immutable runs/optimization_finalists/refits/<fusion-branch>; reuse only completed verified assets",
        "packing":"Only relative branch manifest and its referenced NPZ assets, plus integrated config.json",
        "size_rule":"Sum NPZ ZIP member uncompressed sizes and every non-NPZ file byte in the entire checkpoint",
        "maximum_uncompressed_bytes":LIMIT_BYTES,"fixture":"First467 official training questions in original order, unmodified embeddings and references; generated audit manifest in project .local",
        "timing":"One fresh checkenv prediction process plus one fresh independent scorer process, measured together by parent perf_counter",
        "maximum_combined_seconds":LIMIT_SECONDS,"process_timeout_seconds":120,
        "timing_gate":"All registered managed training studies must be complete before any resource timing; no concurrent refit subprocess",
        "resource_failure":"Oversize, measured combined time>25s, or resource-process timeout makes recipe ineligible; no recipe changes or timing retries",
        "implementation_failure":"Nonzero subprocess exit, replay/source/hash/manifest failure stops the entire run for documented repair; do not skip a recipe",
        "outputs":"Ranked resource assessments, exact prevalidation shortlist and source/config/asset hashes; zero validation/test evaluations",
        "fit_count":"Count completed new full-training learner fits; reused completed branch cache is separately recorded"}
    path=OUT/"protocol.json"
    if path.exists():
        current=read(path);existing={k:v for k,v in current.items() if k not in ("created_utc","repair")}
        if existing!=json.loads(json.dumps(expected)):raise RuntimeError("Registration changed; preserve old evidence")
        validate_registration(current);return current
    expected["created_utc"]=stamp();dump(path,expected)
    dump(OUT/"progress.json",dict(stage="ready",completed_fits=0,planned_fits=0,detail="Registered; full-refit count depends on fixed ranked resource screen; no fits launched",updated_utc=stamp()))
    return expected

def register_repair():
    """Register only the recorded logging repair and exact resume boundary."""
    folder=OUT/"repair_logging";compat=core_compatibility()
    archived=read(folder/"driver_archive_hashes.json")
    for name,value in archived.items():
        if sha(inside(folder,name))!=value:raise RuntimeError("Driver archive changed")
    prior=read(folder/"driver_protocol_before.json")
    if archived["optimization_finalists_before.py"]!=prior["source_sha256"]["optimization_finalists.py"]:
        raise RuntimeError("Archived driver source does not match old registration")
    current=source_closure()
    changed={name for name in set(prior["source_sha256"])|set(current) if prior["source_sha256"].get(name)!=current.get(name)}
    if changed!={"optimization_core_refit.py","optimization_finalists.py"}:raise RuntimeError("Unexpected sources changed in logging recovery")
    if prior["source_sha256"]["optimization_core_refit.py"]!=compat["old_sha256"]:raise RuntimeError("Wrong historical core hash")
    started=read(folder/"driver_run_started_before.json")
    if started["fusion_summary_sha256"]!=sha(FUSION/"summary.json"):raise RuntimeError("Fusion ranking changed after failure")
    oldsummary=read(folder/"driver_failed_summary.json")
    if oldsummary.get("status")!="failed" or oldsummary.get("model_fit_calls")!=10:raise RuntimeError("Unexpected failed-attempt accounting")
    if sha(OUT/"resource_assessments.json")!=archived["driver_resource_assessments_before.json"]:
        raise RuntimeError("Resource decisions changed after failure")
    protocol=dict(prior,source_sha256=current)
    protocol["repair"]={"reason":"Logging-only duplicate seconds fix; recover already-trained P; resume exact fixed ranking without repeating timing",
        "created_utc":stamp(),"old_protocol_sha256":sha(folder/"driver_protocol_before.json"),
        "core_compatibility_sha256":sha(folder/"core_compatibility.json"),"driver_archive_hashes":archived,
        "reported_model_fits_before_failure":10,"recovered_completed_model_fits":1,"actual_completed_fits_before_resume":11,
        "fusion_summary_sha256":started["fusion_summary_sha256"],"resource_assessments_sha256":sha(OUT/"resource_assessments.json"),
        "resource_evidence_sha256":{p.relative_to(OUT).as_posix():sha(p) for p in sorted((OUT/"resource_checks").rglob("*")) if p.is_file()},
        "no_prior_resource_timing_repeated":True}
    existing=read(OUT/"protocol.json")
    if "repair" in existing:
        a={k:v for k,v in existing["repair"].items() if k!="created_utc"}
        b={k:v for k,v in protocol["repair"].items() if k!="created_utc"}
        if a!=b or existing["source_sha256"]!=current:raise RuntimeError("Registered repair changed")
        validate_registration(existing);return existing
    if existing!=prior:raise RuntimeError("Unexpected pre-repair protocol")
    dump(OUT/"protocol.json",protocol);validate_registration(protocol)
    dump(OUT/"progress.json",dict(stage="ready_recovery",completed_fits=11,planned_fits=11,planned_fits_dynamic=True,
        detail="Logging repair registered; P recovered with zero refits; first resource timing preserved; awaiting explicit resume",updated_utc=stamp()))
    return protocol

def progress(stage,started,state,detail):
    dump(OUT/"progress.json",dict(stage=stage,completed_fits=state["model_fit_calls"],planned_fits=state["planned_fit_calls"],
        planned_fits_dynamic=True,detail=detail,elapsed_seconds=time.perf_counter()-started,updated_utc=stamp()))

def assert_training_quiescent():
    queue=read(TOP/"worklogs"/"optimization_queue.json")
    unresolved=[]
    for job in queue["jobs"]:
        if job.get("mode")!="managed" or job.get("directory")==OUT.name:continue
        path=ROOT/"runs"/job.get("directory","")/"progress.json"
        if not path.is_file():continue
        p=read(path)
        if p.get("planned_fits",0)>0 and p.get("stage")!="complete":unresolved.append(job["id"])
    if unresolved:raise RuntimeError("Resource timing requires all managed training complete: "+", ".join(unresolved))

def complete_manifest(branch,directory):
    adapter,name,_=BRANCHES[branch];path=directory/"manifest.json"
    if not path.is_file():raise RuntimeError("Incomplete cached refit: "+branch)
    manifest=read(path)
    if manifest.get("branch")!=name:raise RuntimeError("Cached branch name mismatch")
    if read(directory/"progress.json").get("stage")!="complete":raise RuntimeError("Cached branch is not complete")
    if adapter=="core":
        if manifest.get("format")!="mp2-core-branch-v1":raise RuntimeError("Invalid core manifest")
        files=[(m["file"],m["sha256"]) for m in manifest["members"]]
        hashes=manifest["source_sha256"]
    else:
        expected="mp2_sequence_branch_v1" if adapter=="sequence" else "mp2_structure_branch_v1"
        if manifest.get("format")!=expected:raise RuntimeError("Invalid neural manifest")
        files=[(manifest["asset_path"],manifest["asset_sha256"])]
        hashes=manifest["inference_sources_sha256"]
    for name,value in hashes.items():
        if not source_compatible(name,value):raise RuntimeError("Cached inference source changed")
    for name,value in files:
        asset=inside(directory,name)
        if asset.suffix.lower()!=".npz" or sha(asset)!=value:raise RuntimeError("Invalid cached asset")
    return path,manifest,files

def branch_fit_count(branch):
    name=BRANCHES[branch][1]
    return 5 if name.startswith("A5_") else 3 if name in ("A_D1D2","drop_one_half") else 1

def ensure_branch(branch,started,state):
    directory=inside(OUT,"refits/"+safe_name(branch))
    if directory.exists():
        path,manifest,_=complete_manifest(branch,directory)
        state["branch_records"].append(dict(branch=branch,reused=True,manifest=str(path.relative_to(ROOT)),manifest_sha256=sha(path),new_model_fits=0))
        return directory
    adapter,name,environment=BRANCHES[branch]
    command=[str(runtime(environment)),"-u",str(ROOT/("optimization_"+adapter+"_refit.py"))]
    if adapter!="core":command.append("--fit")
    command.extend(("--branch",name,"--output",str(directory)))
    state["planned_fit_calls"]+=branch_fit_count(branch)
    progress("refitting",started,state,"Full-training branch "+branch+"; fixed recipe, no validation/test")
    log=OUT/"logs"/(branch+"_refit.log");log.parent.mkdir(exist_ok=True)
    tick=time.perf_counter()
    with log.open("wb") as stream:
        result=subprocess.run(command,cwd=ROOT,env=child_environment(),stdout=stream,stderr=subprocess.STDOUT,check=False)
    if result.returncode!=0:
        partial=read(directory/"progress.json") if (directory/"progress.json").is_file() else {}
        known=int(partial.get("completed_fits",0));state["model_fit_calls"]+=known
        state["branch_records"].append(dict(branch=branch,reused=False,status="implementation_failure",known_completed_fits=known,returncode=result.returncode,log=str(log.relative_to(ROOT))))
        raise RuntimeError("Branch refit failed; preserve log and stop: "+branch)
    path,manifest,_=complete_manifest(branch,directory)
    count=int(manifest.get("model_fit_calls",1))
    if count!=branch_fit_count(branch):raise RuntimeError("Unexpected full-refit learner count")
    state["model_fit_calls"]+=count
    state["branch_records"].append(dict(branch=branch,reused=False,status="complete",new_model_fits=count,
        manifest=str(path.relative_to(ROOT)),manifest_sha256=sha(path),wall_seconds=time.perf_counter()-tick,command=command))
    dump(OUT/"branch_records.json",state["branch_records"])
    return directory

def child_environment():
    env=os.environ.copy();env["PYTHONDONTWRITEBYTECODE"]="1"
    for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):env[key]="4"
    return env

def build_checkpoint(recipe,branches):
    destination=inside(OUT,"checkpoints/"+safe_name(recipe["name"]))
    if destination.exists():raise FileExistsError("Preserve existing assembled checkpoint")
    destination.mkdir(parents=True);components=[]
    for branch,weight in recipe["weights"].items():
        if not isinstance(weight,(int,float)) or weight<=0:raise ValueError("Unexpected component weight")
        adapter,_,_=BRANCHES[branch];source=branches[branch]
        path,_,files=complete_manifest(branch,source)
        local=inside(destination,"branches/"+safe_name(branch));local.mkdir(parents=True)
        manifest=local/"manifest.json";shutil.copyfile(path,manifest)
        for filename,_ in files:
            asset=inside(local,filename);asset.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(inside(source,filename),asset)
        components.append(dict(adapter=adapter,manifest=manifest.relative_to(destination).as_posix(),manifest_sha256=sha(manifest),weight=float(weight)))
    if abs(sum(c["weight"] for c in components)-1)>1e-9:raise ValueError("Weights must sum to one")
    dump(destination/"config.json",dict(format="mp2-integrated-verifier-v1",components=components))
    return destination

def checkpoint_inventory(directory):
    directory=directory.resolve();hashes={};sizes={};total=0
    for path in sorted(directory.rglob("*")):
        if not path.is_file():continue
        if directory not in path.resolve().parents:raise RuntimeError("Checkpoint contains escaping path")
        relative=path.relative_to(directory).as_posix();hashes[relative]=sha(path)
        if path.suffix.lower()==".npz":
            with zipfile.ZipFile(path) as archive:size=sum(member.file_size for member in archive.infolist())
        else:size=path.stat().st_size
        sizes[relative]=size;total+=size
    return total,hashes,sizes

def make_fixture():
    """Construct an audit-named fixture exclusively from official training rows."""
    from common import DATA,PROTOCOL,load_inputs,read_jsonl,check_data
    import numpy as np
    bank,arrays=load_inputs("train");refs=read_jsonl(DATA/"gsm8k_train.jsonl")
    if len(bank)!=4755 or len(refs)!=len(bank):raise RuntimeError("Unexpected training bank")
    bank=bank[:467];refs=refs[:467]
    if any(b["id"]!=r["id"] or b["question"]!=r["question"] for b,r in zip(bank,refs)):raise RuntimeError("Training fixture alignment")
    source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (DATA/"candidates/gsm8k_train.jsonl",DATA/"candidates/gsm8k_train.npz",DATA/"gsm8k_train.jsonl")}
    provenance=dict(scope="first467trainingquestions_only",source_sha256=source_hashes,n=467,ids=[b["id"] for b in bank])
    if FIXTURE.exists():
        if read(FIXTURE/"provenance.json")!=provenance:raise RuntimeError("Existing fixture differs; preserve it")
        check_data(FIXTURE,("audit",));return
    (FIXTURE/"candidates").mkdir(parents=True)
    for filename,rows in (("candidates/gsm8k_audit.jsonl",bank),("gsm8k_audit.jsonl",refs)):
        inside(FIXTURE,filename).write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")
    np.savez(FIXTURE/"candidates/gsm8k_audit.npz",embeddings=arrays["embeddings"][:467].reshape(-1,1024))
    files=("candidates/gsm8k_audit.jsonl","candidates/gsm8k_audit.npz","gsm8k_audit.jsonl")
    dump(FIXTURE/"manifest.json",dict(protocol=PROTOCOL,splits={"audit":467},sha256={name:sha(FIXTURE/name) for name in files}))
    dump(FIXTURE/"provenance.json",provenance);check_data(FIXTURE,("audit",))

def resource_probe(recipe_name,checkpoint):
    assert_training_quiescent();make_fixture()
    output=inside(OUT,"resource_checks/"+safe_name(recipe_name))
    if output.exists():raise FileExistsError("No resource timing retries in this registered run")
    output.mkdir(parents=True);pred=output/"predictions.json";score=output/"score.json"
    commands=[
        [str(runtime("checkenv")),str(ROOT/"predict.py"),"--method",str(ROOT/"optimized_selection.py"),"--checkpoint",str(checkpoint),"--split","audit","--data-dir",str(FIXTURE),"--output",str(pred)],
        [str(runtime("checkenv")),str(ROOT/"score_predictions.py"),"--predictions",str(pred),"--split","audit","--data-dir",str(FIXTURE),"--output",str(score)],
    ]
    started=time.perf_counter();processes=[];timed_out=False
    for index,command in enumerate(commands):
        log=output/("prediction.log" if index==0 else "scoring.log");tick=time.perf_counter()
        with log.open("wb") as stream:
            try:result=subprocess.run(command,cwd=ROOT,env=child_environment(),stdout=stream,stderr=subprocess.STDOUT,timeout=120,check=False)
            except subprocess.TimeoutExpired:
                timed_out=True;processes.append(dict(command=command,seconds=time.perf_counter()-tick,status="timeout"));break
        processes.append(dict(command=command,seconds=time.perf_counter()-tick,returncode=result.returncode))
        if result.returncode!=0:
            dump(output/"failure.json",dict(status="implementation_failure",processes=processes,log=str(log.relative_to(ROOT))))
            raise RuntimeError("Resource process failed; do not skip recipe: "+recipe_name)
    elapsed=time.perf_counter()-started
    result=dict(recipe=recipe_name,fixture=str(FIXTURE.relative_to(TOP)),fixture_sha256=sha(FIXTURE/"manifest.json"),
        combined_seconds=elapsed,processes=processes,timeout=timed_out,eligible=not timed_out and elapsed<=LIMIT_SECONDS,
        prediction_and_scorer_completed=not timed_out and len(processes)==2,
        interpretation="Training fixture timing only; accuracy is not used for selection. Local timing does not certify reference CPU.",threads=4)
    dump(output/"resource.json",result);return result

def install_read_guard():
    from common import DATA
    official=DATA.resolve()
    def hook(event,args):
        if event!="open" or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(args[0])).resolve()
        if official in path.parents and path.name.startswith(("gsm8k_validation","gsm8k_test","gsm8k_audit")):
            raise RuntimeError("Finalist preparation cannot read official heldout data")
    sys.addaudithook(hook)

def retained_assessments(fusion,protocol):
    """Validate the previous ranked prefix without rerunning prediction/scoring."""
    assessments=read(OUT/"resource_assessments.json")
    if sha(OUT/"resource_assessments.json")!=protocol["repair"]["resource_assessments_sha256"]:
        raise RuntimeError("Preserved assessment prefix changed")
    for filename,value in protocol["repair"]["resource_evidence_sha256"].items():
        if sha(inside(OUT,filename))!=value:raise RuntimeError("Preserved timing evidence changed")
    ranked=fusion["ranked_distinct_consistent"];rows={r["name"]:r for r in fusion["results"]}
    for rank,entry in enumerate(assessments):
        if entry["rank"]!=rank or entry["name"]!=ranked[rank]:raise RuntimeError("Prior assessment is not exact ranked prefix")
        if entry["oof_mean_hard_correct"]!=rows[entry["name"]]["mean_hard_correct"]:raise RuntimeError("OOF ranking changed")
        checkpoint=inside(ROOT,entry["checkpoint"]);size,hashes,sizes=checkpoint_inventory(checkpoint)
        if size!=entry["uncompressed_bytes"] or hashes!=entry["checkpoint_sha256"] or sizes!=entry["uncompressed_files"]:
            raise RuntimeError("Previously assessed checkpoint changed")
        if size<=LIMIT_BYTES:
            probe=read(OUT/"resource_checks"/entry["name"]/"resource.json")
            if probe["combined_seconds"]!=entry["probe_seconds"] or probe["eligible"]!=entry["eligible"]:raise RuntimeError("Prior timing mismatch")
            if entry["eligible"] and (not probe["prediction_and_scorer_completed"] or probe["timeout"] or probe["combined_seconds"]>LIMIT_SECONDS):
                raise RuntimeError("Invalid preserved eligible timing")
        elif entry["eligible"]:raise RuntimeError("Preserved oversized checkpoint marked eligible")
    return assessments

def run(resume_repair=False):
    protocol=register_repair() if resume_repair else register();validate_registration(protocol)
    if (OUT/"summary.json").exists():
        previous=read(OUT/"summary.json")
        if previous.get("status")=="complete":print("Already complete; no fits or probes repeated");return
        if not resume_repair:raise RuntimeError("Preserve prior failed attempt")
    if (OUT/"run_started.json").exists() and not resume_repair:raise RuntimeError("Incomplete prior attempt; preserve before explicit repair")
    if resume_repair and (OUT/"repair_run_started.json").exists():raise RuntimeError("Preserve incomplete recovery attempt")
    fusion=read(FUSION/"summary.json")
    if fusion.get("status")!="complete" or read(FUSION/"progress.json").get("stage")!="complete":raise RuntimeError("Fusion is not complete")
    assert_training_quiescent();install_read_guard()
    state={"model_fit_calls":0,"planned_fit_calls":0,"branch_records":[]};started=time.perf_counter()
    assessments=[];candidates=[];branches={}
    if resume_repair:
        state.update(model_fit_calls=11,planned_fit_calls=11,branch_records=read(OUT/"repair_logging"/"driver_branch_records_before.json"))
        path,manifest,_=complete_manifest("P",OUT/"refits"/"P")
        state["branch_records"].append(dict(branch="P",reused=True,status="recovered_completed_fit",previously_completed_model_fits=1,
            new_model_fits=0,manifest=str(path.relative_to(ROOT)),manifest_sha256=sha(path),recovery_evidence="repair_logging/P_recovery.json"))
        assessments=retained_assessments(fusion,protocol)
        candidates=[{k:a[k] for k in ("name","checkpoint","oof_mean_hard_correct","checkpoint_sha256","uncompressed_bytes","probe_seconds")}
                    for a in assessments if a["eligible"]]
        for branch in BRANCHES:
            directory=OUT/"refits"/branch
            if directory.exists():complete_manifest(branch,directory);branches[branch]=directory
    dump(OUT/("repair_run_started.json" if resume_repair else "run_started.json"),dict(created_utc=stamp(),protocol_sha256=sha(OUT/"protocol.json"),
        fusion_summary_sha256=sha(FUSION/"summary.json"),resume_repair=resume_repair,previously_completed_model_fits=11 if resume_repair else 0))
    try:
        ranked=fusion["ranked_distinct_consistent"]
        rows={r["name"]:r for r in fusion["results"]}
        registered={r["name"]:r for r in read(FUSION/"protocol.json")["menu"]}
        if len(ranked)!=len(set(ranked)) or any(name not in registered for name in ranked):raise RuntimeError("Invalid ranked menu")
        for rank,name in enumerate(ranked):
            if rank<len(assessments):continue
            if len(candidates)==2:break
            row=rows[name];recipe=registered[name]
            if row["weights"]!=recipe["weights"] or not row["consistent_vs_F"]:raise RuntimeError("Fusion ranking differs from registered recipe")
            for branch in recipe["weights"]:
                if branch not in branches:branches[branch]=ensure_branch(branch,started,state)
            validate_registration(protocol)
            checkpoint=build_checkpoint(recipe,branches)
            total,hashes,sizes=checkpoint_inventory(checkpoint)
            assessment=dict(rank=rank,name=name,checkpoint=str(checkpoint.relative_to(ROOT)).replace("\\","/"),uncompressed_bytes=total,
                checkpoint_sha256=hashes,uncompressed_files=sizes,oof_mean_hard_correct=row["mean_hard_correct"])
            if total>LIMIT_BYTES:
                assessment.update(eligible=False,reason="uncompressed checkpoint exceeds16MiB",probe_seconds=None)
            else:
                progress("resource_check",started,state,"One fresh prediction+scorer timing: "+name)
                probe=resource_probe(name,checkpoint)
                assessment.update(eligible=probe["eligible"],reason="passed" if probe["eligible"] else "local timing exceeds25s or timed out",probe_seconds=probe["combined_seconds"])
                if checkpoint_inventory(checkpoint)[1]!=hashes:raise RuntimeError("Checkpoint changed during resource probe")
            assessments.append(assessment);dump(OUT/"resource_assessments.json",assessments)
            if assessment["eligible"]:
                candidates.append({k:assessment[k] for k in ("name","checkpoint","oof_mean_hard_correct","checkpoint_sha256","uncompressed_bytes","probe_seconds")})
            if len(candidates)==2:break
        validate_registration(protocol)
        shortlist=dict(candidates=candidates,source_sha256=protocol["source_sha256"],selection_policy_sha256=protocol["selection_policy_sha256"],created_utc=stamp())
        dump(OUT/"shortlist.json",shortlist)
        summary=dict(status="complete",candidates=candidates,count=len(candidates),model_fit_calls=state["model_fit_calls"],
            reused_completed_branch_count=sum(r.get("reused",False) for r in state["branch_records"]),
            branch_count=len(branches),assessed_recipes=len(assessments),shortlist_sha256=sha(OUT/"shortlist.json"),
            source_sha256=protocol["source_sha256"],selection_policy_sha256=protocol["selection_policy_sha256"],
            validation_evaluations=0,test_access=False,elapsed_seconds=time.perf_counter()-started,
            logging_recovery=resume_repair,recovered_completed_model_fits=1 if resume_repair else 0,
            new_model_fits_in_this_execution=state["model_fit_calls"]-(11 if resume_repair else 0),
            interpretation="Exact resource-eligible checkpoints frozen before any new validation; training fixture accuracy is not a selector")
        dump(OUT/"summary.json",summary);progress("complete",started,state,"Resource-qualified shortlist frozen; no validation/test evaluations")
        print(json.dumps({"status":"complete","count":len(candidates),"model_fit_calls":state["model_fit_calls"]}),flush=True)
    except Exception as exc:
        dump(OUT/"branch_records.json",state["branch_records"])
        dump(OUT/"failure.json",dict(error=repr(exc),model_fit_calls=state["model_fit_calls"],assessed_recipes=len(assessments),created_utc=stamp()))
        dump(OUT/"summary.json",dict(status="failed",candidates=candidates,count=len(candidates),model_fit_calls=state["model_fit_calls"],validation_evaluations=0,test_access=False,error=repr(exc)))
        progress("failed",started,state,"Implementation failure preserved; no recipe silently discarded")
        raise

def main():
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--register-only",action="store_true");group.add_argument("--run",action="store_true")
    group.add_argument("--register-repair-only",action="store_true");group.add_argument("--resume-repair",action="store_true")
    args=parser.parse_args()
    if args.register_only:
        value=register();print(json.dumps({"status":"ready","source_files":len(value["source_sha256"]),"model_fit_calls":0,"data_reads":0}))
    elif args.register_repair_only:
        value=register_repair();print(json.dumps({"status":"ready_recovery","cumulative_completed_model_fits":11,"new_fits":0,"new_resource_probes":0}))
    else:run(resume_repair=args.resume_repair)

if __name__=="__main__":main()
