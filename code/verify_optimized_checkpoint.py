"""Training-input-only contract audit for a frozen integrated checkpoint.

Explicit --checkpoint PATH --output NEW_DIRECTORY runs the audit. It never
fits a model, opens references, or evaluates validation/test accuracy.
"""
from __future__ import annotations
import os
import sys
sys.dont_write_bytecode=True
for _name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_name]="4"
import argparse
from contextlib import contextmanager
import copy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile
import numpy as np
from common import DATA,PROTOCOL,load_inputs,parse_answer,validate_choices
from selection import select

ROOT=Path(__file__).resolve().parent
TASK=ROOT.parents[1]
PERMUTATIONS=((1,0,2,3),(3,2,1,0),(2,3,0,1),(1,2,3,0))
FORBIDDEN_MODULES=("torch","sklearn","scipy","pandas","transformers","onnxruntime","lightgbm")
_STATE=None
_HOOK=False


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def dump(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def inside(root,relative):
    path=(root/relative).resolve()
    if root not in path.parents:raise ValueError("Checkpoint path escapes its directory")
    return path


def inventory(checkpoint):
    checkpoint=Path(checkpoint).resolve()
    config=read(checkpoint/"config.json")
    if config.get("format")!="mp2-integrated-verifier-v1":raise ValueError("Unexpected checkpoint format")
    weights=[float(c["weight"]) for c in config["components"]]
    if not weights or any(not np.isfinite(w) or w<=0 for w in weights) or abs(sum(weights)-1)>1e-9:
        raise ValueError("Invalid mixture weights")
    referenced={checkpoint/"config.json"}
    for entry in config["components"]:
        manifest=inside(checkpoint,entry["manifest"])
        if sha(manifest)!=entry["manifest_sha256"]:raise ValueError("Manifest checksum mismatch")
        referenced.add(manifest);metadata=read(manifest)
        if entry["adapter"]=="core":
            files=[(m["file"],m["sha256"]) for m in metadata["members"]]
        elif entry["adapter"] in ("sequence","structure"):
            files=[(metadata["asset_path"],metadata["asset_sha256"])]
        else:raise ValueError("Unknown adapter")
        for filename,digest in files:
            path=inside(manifest.parent,filename)
            if checkpoint not in path.parents or sha(path)!=digest:raise ValueError("Asset checksum/path mismatch")
            referenced.add(path)
    hashes={};sizes={}
    for path in sorted(checkpoint.rglob("*")):
        if not path.is_file():continue
        if checkpoint not in path.resolve().parents:raise ValueError("Checkpoint has an escaping file")
        relative=path.relative_to(checkpoint).as_posix();hashes[relative]=sha(path)
        if path.suffix.lower()==".npz":
            with zipfile.ZipFile(path) as archive:sizes[relative]=sum(v.file_size for v in archive.infolist())
        else:sizes[relative]=path.stat().st_size
    total=sum(sizes.values())
    if total>16*2**20:raise ValueError("Whole checkpoint exceeds16MiB")
    return dict(hashes=hashes,file_uncompressed_bytes=sizes,uncompressed_bytes=total,
        referenced_files=[str(p) for p in sorted(referenced)],
        extra_files=sorted(set(hashes)-{p.relative_to(checkpoint).as_posix() for p in referenced}))


def numpy_only():
    loaded=[name for name in FORBIDDEN_MODULES if name in sys.modules]
    if loaded:raise RuntimeError("Forbidden inference dependencies: "+str(loaded))


def install_hook():
    global _HOOK
    if _HOOK:return
    runtime_roots={Path(sys.prefix).resolve(),Path(sys.base_prefix).resolve()}
    def audit(event,args):
        state=_STATE
        if state is None:return
        if event=="import" and str(args[0]).split(".")[0] in FORBIDDEN_MODULES:
            raise RuntimeError("Training dependency imported during inference")
        if event.startswith("socket.") or event in ("subprocess.Popen","os.system"):
            state["blocked"].append(event);raise RuntimeError("Network/process access during inference")
        if event!="open" or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(args[0])).resolve()
        mode,flags=args[1],args[2] or 0
        writing=isinstance(mode,str) and any(c in mode for c in "wax+") or flags&(os.O_WRONLY|os.O_RDWR)
        if writing:raise RuntimeError("Inference attempted a file write")
        if path in state["allowed"]:
            state["asset_reads"].add(str(path));return
        source=path.suffix.lower() in (".py",".pyc",".pyd",".dll")
        project_source=path.parent==ROOT or path.parent==ROOT/"__pycache__"
        runtime_source=any(p==path or p in path.parents for p in runtime_roots)
        if source and (project_source or runtime_source):
            state["source_reads"].add(str(path));return
        state["blocked"].append(str(path))
        raise RuntimeError("Inference attempted dataset/reference/cache access: "+str(path))
    sys.addaudithook(audit);_HOOK=True


@contextmanager
def inference_guard(allowed):
    global _STATE
    if _STATE is not None:raise RuntimeError("Nested inference guard")
    install_hook();numpy_only()
    state=dict(allowed=set(allowed),asset_reads=set(),source_reads=set(),blocked=[])
    _STATE=state
    try:yield state
    finally:_STATE=None
    numpy_only()


def guarded_predict(bank,arrays,checkpoint):
    """Used by the official-runner shim; data loading occurs before this call."""
    info=inventory(checkpoint)
    with inference_guard(Path(p) for p in info["referenced_files"]):
        import optimized_selection
        choices=optimized_selection.predict(bank,arrays,checkpoint)
    validate_choices(bank,[row["id"] for row in bank],choices)
    return choices


def current_resource_check():
    path=ROOT/"runs/optimization_finalists/progress.json"
    return path.exists() and read(path).get("stage")=="resource_check"


def run(checkpoint,output):
    checkpoint=Path(checkpoint).resolve();output=Path(output).resolve()
    if current_resource_check():raise RuntimeError("Wait for the finalist resource measurement to finish")
    if output.exists():raise FileExistsError("Use a new audit output directory")
    if TASK not in output.parents:raise ValueError("Keep audit evidence inside the MP2 task directory")
    if checkpoint==output or checkpoint in output.parents:raise ValueError("Audit output must remain outside the checkpoint")
    output.mkdir(parents=True)
    started=time.perf_counter();info=inventory(checkpoint)
    digest=hashlib.sha256(json.dumps(info["hashes"],sort_keys=True).encode()).hexdigest()
    dump(output/"protocol.json",dict(checkpoint=str(checkpoint),checkpoint_inventory_sha256=digest,
        verifier_sha256=sha(__file__),training_questions=32,external_fixture_questions=12,
        permutations=PERMUTATIONS,tie_margin=1e-4,new_model_fits=0,reference_access=False,
        fixture_source="32 evenly spaced supplied TRAIN input questions; no references",
        inference_guard="Only referenced checkpoint files and Python/runtime source reads; no dataset/cache/network/process access"))
    bank,arrays=load_inputs("train")
    chosen=np.linspace(0,len(bank)-1,32,dtype=int)
    bank=copy.deepcopy([bank[int(i)] for i in chosen]);emb=arrays["embeddings"][chosen].copy()
    original_bank=copy.deepcopy(bank);original_emb=emb.copy();emb.setflags(write=False)
    allowed=[Path(p) for p in info["referenced_files"]]
    source_reads=set();asset_reads=set();checks={}
    def score(rows,values):
        if current_resource_check():raise RuntimeError("Resource measurement started; pause contract audit")
        with inference_guard(allowed) as state:
            import optimized_selection
            result,_=optimized_selection.score_candidates(rows,dict(embeddings=values),checkpoint)
        source_reads.update(state["source_reads"]);asset_reads.update(state["asset_reads"])
        assert result.shape==(len(rows),4) and np.isfinite(result).all()
        return result
    def choices(rows,values):
        with inference_guard(allowed) as state:
            import optimized_selection
            result=optimized_selection.predict(rows,dict(embeddings=values),checkpoint)
        source_reads.update(state["source_reads"]);asset_reads.update(state["asset_reads"])
        validate_choices(rows,[r["id"] for r in rows],result)
        assert isinstance(result,list) and all(type(i) is int for i in result)
        return result
    try:
        z=score(bank,emb);selected=choices(bank,emb)
        assert selected==select(bank,z,"mean")
        assert bank==original_bank and np.array_equal(emb,original_emb)
        checks["native_indices_finite_scores_nonmutation"]=True
        empty=np.empty((0,4,1024),np.float32)
        assert score([],empty).shape==(0,4) and choices([],empty)==[]
        checks["empty"]=True
        newid=[dict(r,id=f"contract-unseen-{i}") for i,r in enumerate(bank)]
        assert np.array_equal(score(newid,emb),z)
        checks["new_ID_exact"]=True
        order=np.random.default_rng(20261031).permutation(len(bank))
        rz=score([bank[int(i)] for i in order],emb[order])
        reorder_error=float(np.max(np.abs(rz-z[order])));assert reorder_error<1e-4
        assert select([bank[int(i)] for i in order],rz,"mean")==[selected[int(i)] for i in order]
        checks["question_reorder_max_error"]=reorder_error
        chunk=np.concatenate([score(bank[i:i+7],emb[i:i+7]) for i in range(0,len(bank),7)])
        batch_error=float(np.max(np.abs(chunk-z)));assert batch_error<1e-4
        assert select(bank,chunk,"mean")==selected
        checks["chunk_batch_max_error"]=batch_error
        _,margin,_=select(bank,z,"mean",return_details=True)
        answers=[parse_answer(row["candidates"][j]["text"]) for row,j in zip(bank,selected)]
        permutations=[]
        for perm in PERMUTATIONS:
            permuted=[dict(row,candidates=[dict(row["candidates"][old],index=new) for new,old in enumerate(perm)]) for row in bank]
            pz=score(permuted,emb[:,perm]);pc=select(permuted,pz,"mean")
            pa=[parse_answer(row["candidates"][j]["text"]) for row,j in zip(permuted,pc)]
            changed=[i for i,(a,b) in enumerate(zip(answers,pa)) if a!=b]
            significant=[i for i in changed if margin[i]>1e-4]
            assert not significant,(perm,significant)
            permutations.append(dict(permutation=perm,max_score_difference=float(np.max(np.abs(pz-z[:,perm]))),
                answer_changes=changed,allowed_tie_changes=changed,non_tied_answer_changes=0))
        checks["candidate_permutations"]=permutations
        synthetic=[]
        for valid_slot in (2,None):
            synthetic.append(dict(id="synthetic-"+str(valid_slot),question="What is two plus three?",candidates=[
                dict(index=i,text="2 + 3 = 5. #### 5" if i==valid_slot else "No final answer supplied.",code=None) for i in range(4)]))
        se=np.zeros((2,4,1024),np.float32);sz=score(synthetic,se);sc=choices(synthetic,se)
        assert sc[0]==2 and 0<=sc[1]<4
        checks["one_valid_and_all_invalid"]=True
        # Confirm the installed guard rejects forbidden access before any bytes
        # or network activity are reached; these are deliberate tripwires.
        tripwires=[]
        import socket
        for label,action in (("reference",lambda:open(DATA/"gsm8k_train.jsonl","rb")),
                ("prediction_cache",lambda:open(ROOT/"runs/optimization_baselines/oof_logits.npz","rb")),
                ("network",lambda:socket.getaddrinfo("example.invalid",80))):
            with inference_guard(allowed):
                try:action()
                except RuntimeError:tripwires.append(label)
                else:raise AssertionError("Guard tripwire did not reject "+label)
        checks["guard_tripwires_blocked"]=tripwires
        fixture=TASK/".local/contract-fixtures"/(digest[:16]+"-"+output.name)
        if fixture.exists():raise FileExistsError("Preserve existing external audit fixture")
        (fixture/"candidates").mkdir(parents=True)
        external=[dict(r,id=f"external-contract-{digest[:8]}-{i}") for i,r in enumerate(bank[:12])]
        (fixture/"candidates/gsm8k_audit.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in external),encoding="utf-8")
        np.savez(fixture/"candidates/gsm8k_audit.npz",embeddings=emb[:12].reshape(-1,1024))
        files=("candidates/gsm8k_audit.jsonl","candidates/gsm8k_audit.npz")
        dump(fixture/"manifest.json",dict(protocol=PROTOCOL,splits={"audit":12},sha256={n:sha(fixture/n) for n in files}))
        assert not (fixture/"gsm8k_audit.jsonl").exists()
        shim=fixture/"guarded_method.py"
        shim.write_text("import sys\nfrom pathlib import Path\nsys.path.insert(0,"+repr(str(ROOT))+")\nfrom verify_optimized_checkpoint import guarded_predict\ndef predict(bank,arrays,checkpoint):\n    return guarded_predict(bank,arrays,checkpoint)\n",encoding="utf-8")
        prediction=output/"external_predictions.json"
        command=[str(TASK/".local/checkenv/Scripts/python.exe"),"-B",str(ROOT/"predict.py"),"--method",str(shim),"--checkpoint",str(checkpoint),
            "--split","audit","--data-dir",str(fixture),"--output",str(prediction)]
        if current_resource_check():raise RuntimeError("Resource timing active; do not launch contract subprocess")
        proc=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding="utf-8",timeout=180,
            env={**os.environ,"PYTHONUTF8":"1","PYTHONDONTWRITEBYTECODE":"1"},creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        dump(output/"external_process.json",dict(command=command,returncode=proc.returncode,stdout=proc.stdout,stderr=proc.stderr,
            fixture=str(fixture),fixture_contains_references=False,method_shim_sha256=sha(shim)))
        assert proc.returncode==0,proc.stderr[-2000:]
        result=read(prediction);assert result["ids"]==[r["id"] for r in external] and result["selected"]==selected[:12]
        checks["official_input_only_new_ID_fixture"]=True
        assert inventory(checkpoint)["hashes"]==info["hashes"]
        numpy_only()
        outcome=dict(status="passed",checks=checks,checkpoint=str(checkpoint),checkpoint_inventory=info,
            input_indices=chosen.tolist(),input_questions=32,external_fixture_questions=12,
            checkpoint_unchanged=True,inference_source_files_read=sorted(source_reads),inference_asset_files_read=sorted(asset_reads),
            inference_dataset_reference_cache_reads=0,numpy_only=True,network_blocked_during_inference=True,
            new_model_fits=0,validation_access=False,test_access=False,raw_training_references_read=False,
            seconds=time.perf_counter()-started,created_utc=datetime.now(timezone.utc).isoformat(),
            interpretation="Contract/integrity evidence only; training fixture choices are not used for selecting a recipe")
        dump(output/"verification.json",outcome)
        print(json.dumps({k:v for k,v in outcome.items() if k not in ("checkpoint_inventory","checks","inference_source_files_read","inference_asset_files_read")},ensure_ascii=False))
    except Exception as error:
        dump(output/"failure.json",dict(status="failed",error=repr(error),completed_checks=checks,new_model_fits=0,
            validation_access=False,test_access=False,seconds=time.perf_counter()-started))
        raise


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();run(args.checkpoint,args.output)
