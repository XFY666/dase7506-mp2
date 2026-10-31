"""Explicit full-training refits and NumPy-only structural branch inference.

No training occurs on import. The caller must select a branch before train_branch
or --fit. No validation/test scoring, branch search or checkpoint replacement is
implemented here.
"""
from __future__ import annotations
import os
import sys
sys.dont_write_bytecode=True
for _key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_key]="4"
import argparse
from contextlib import contextmanager
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import time
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parent
FORMAT="mp2_structure_branch_v1"
BRANCHES=("node_mlp","equation_gcn","cross_tree")
EPOCH_POLICY="max(1,floor(median(six recorded inner-selected epochs)+0.5))"

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def dump(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");tmp.replace(path)

def inference_sources(name):
    if name not in BRANCHES:raise ValueError("Unknown structural branch")
    files=["optimization_structure_refit.py","common.py","selection.py","text_features.py"]
    if name=="cross_tree":files.extend(("cross_candidate_features.py","tree_model.py"))
    else:files.extend(("equation_graph_model.py","sequence_model.py","quantitative_features.py"))
    return {file:sha(ROOT/file) for file in files}

def refit_plan(name):
    """Derive a fixed refit recipe from completed training-only records."""
    if name not in BRANCHES:raise ValueError("Unknown structural branch")
    run="optimization_trees" if name=="cross_tree" else "optimization_graph"
    folder=ROOT/"runs"/run
    required=("protocol.json","summary.json","fit_records.json","source_hashes.json")
    if not all((folder/file).is_file() for file in required):raise RuntimeError("Completed source study required: "+run)
    sources=read(folder/"source_hashes.json")
    if any(sha(ROOT/file)!=value for file,value in sources.items()):raise RuntimeError("Source study implementation changed")
    summary=read(folder/"summary.json");protocol=read(folder/"protocol.json")
    expected=42 if name=="cross_tree" else 24
    if summary.get("model_fit_calls")!=expected:raise RuntimeError("Source study did not complete all registered fits")
    plan={"branch":name,"source_run":"runs/"+run,"data_scope":"full supplied training split only",
        "planned_model_fits":1,"validation_access":False,"test_access":False,"initialization_seed":7506 if name!="cross_tree" else 20261013,
        "score_scale":"std of all-four-centered full-training scores at valid candidate entries; no heldout calibration",
        "source_study_sha256":sources,"evidence_sha256":{file:sha(folder/file) for file in required},
        "refit_source_sha256":sha(__file__)}
    if name=="cross_tree":
        if protocol["tree_parameters"]!={"max_iter":200,"max_leaf_nodes":31,"learning_rate":.05,"l2_regularization":10.,"min_samples_leaf":30,"early_stopping":False,"random_state":20261013}:
            raise RuntimeError("Unexpected registered tree recipe")
        plan.update(tree_parameters=protocol["tree_parameters"],cross_feature_indices=list(range(24)),
            features="exact original75 + D1first12 + D2next12",sampling="valid candidates from hard questions, weight4/valid_count perquestion")
    else:
        rows=[r for r in read(folder/"fit_records.json") if r["arm"]==name]
        if len(rows)!=6 or {(r["repeat"],r["fold"]) for r in rows}!={(r,f) for r in range(2) for f in range(3)}:
            raise RuntimeError("Exactly six distinct outer records required")
        rows.sort(key=lambda r:(r["repeat"],r["fold"]));epochs=[int(r["selected_epochs"]) for r in rows]
        if not all(1<=e<=24 for e in epochs):raise RuntimeError("Invalid inner-selected epoch")
        plan.update(epoch_policy=EPOCH_POLICY,outer_selected_epochs=epochs,epochs=max(1,math.floor(float(np.median(epochs))+.5)),
            normalizer="all full-training valid candidates only; original graph1099 numeric inputs; clip[-8,8]",
            loss="eligible-question answer-group CE",graph=name=="equation_gcn")
    return plan

def _raw_scores(name,bank,emb,asset):
    if name=="cross_tree":
        from text_features import extract_features
        from cross_candidate_features import extract_features as cross_features
        from tree_model import tree_scores
        if not bank:return np.empty((0,4),np.float64)
        x=np.concatenate((extract_features(bank),cross_features(bank)[...,asset["cross_feature_indices"]]),axis=-1)
        return tree_scores(x.reshape(-1,x.shape[-1]),asset).reshape(len(bank),4)
    from equation_graph_model import score_candidates
    return score_candidates(bank,emb,asset)

def score_branch(bank,emb,checkpoint_manifest,*,normalize=False):
    """Fresh scores from saved assets; imports only NumPy/standard-library code."""
    if isinstance(checkpoint_manifest,dict):
        manifest=checkpoint_manifest
        directory=Path(manifest.get("directory",".")).resolve()
        if not Path(manifest["asset_path"]).is_absolute() and "directory" not in manifest:
            raise ValueError("Dictionary manifest needs directory or absolute asset_path")
    else:
        path=Path(checkpoint_manifest)
        if path.is_dir():path/= "manifest.json"
        path=path.resolve();directory=path.parent;manifest=read(path)
    name=manifest.get("branch")
    if manifest.get("format")!=FORMAT or name not in BRANCHES:raise ValueError("Unsupported structural manifest")
    if inference_sources(name)!=manifest["inference_sources_sha256"]:raise RuntimeError("Inference sources changed")
    path=Path(manifest["asset_path"])
    if not path.is_absolute():path=directory/path
    if sha(path)!=manifest["asset_sha256"]:raise RuntimeError("Asset checksum mismatch")
    with np.load(path,allow_pickle=False) as archive:asset={k:archive[k] for k in archive.files}
    scale=float(manifest["score_scale"])
    if not np.isfinite(scale) or scale<=0 or float(asset["score_scale"])!=scale:raise ValueError("Invalid score scale")
    value=_raw_scores(name,bank,emb,asset)
    if value.shape!=(len(bank),4) or not np.isfinite(value).all():raise RuntimeError("Invalid fresh scores")
    return value/scale if normalize else value

def predict(bank,arrays,checkpoint_path):
    from selection import select
    return select(bank,score_branch(bank,arrays.get("embeddings"),checkpoint_path),"mean")

@contextmanager
def training_io(output):
    from common import DATA
    active=[True];accesses=set()
    def hook(event,args):
        if not active[0] or event!="open" or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(args[0])).resolve()
        if path.name.startswith(("gsm8k_validation","gsm8k_test","gsm8k_audit")):raise RuntimeError("Training split only")
        if DATA.resolve() in path.parents:accesses.add(path.relative_to(DATA.resolve()).as_posix())
        mode,flags=args[1],args[2] or 0
        writing=(isinstance(mode,str) and any(c in mode for c in "wax+")) or bool(flags&(os.O_WRONLY|os.O_RDWR))
        if writing and output not in path.parents:raise RuntimeError("Write outside fresh refit directory")
    sys.addaudithook(hook)
    try:yield accesses
    finally:active[0]=False

def validate_training(name,bank,emb,labels,refs):
    from common import DATA,read_jsonl,load_inputs,parse_answer
    supplied=read_jsonl(DATA/"candidates/gsm8k_train.jsonl");reference=read_jsonl(DATA/"gsm8k_train.jsonl")
    if bank!=supplied or len(bank)!=4755:raise ValueError("Unmodified full training bank required")
    if refs is not None and refs!=reference:raise ValueError("Unmodified training references required")
    answers=[[parse_answer(c["text"]) for c in r["candidates"]] for r in bank]
    expected=np.array([[a is not None and a==parse_answer(r["answer"]) for a in aa] for aa,r in zip(answers,reference)],np.float32)
    if not np.array_equal(labels,expected):raise ValueError("Training labels differ from official final-answer labels")
    if name!="cross_tree":
        _,arrays=load_inputs("train")
        if emb is None or not np.array_equal(np.asarray(emb),arrays["embeddings"]):raise ValueError("Unmodified training embeddings required")
    valid=np.array([[a is not None for a in aa] for aa in answers],bool)
    hard=np.array([len(set(a for a in aa if a is not None))>=2 for aa in answers],bool)
    return expected,valid,hard

def train_branch(name,output_directory,training_bank,emb,labels,refs=None):
    """Fit exactly the caller-selected structural branch once; no split selection."""
    plan=refit_plan(name);output=Path(output_directory).resolve()
    if output.exists():raise FileExistsError("Preserve existing refit output")
    output.mkdir(parents=True);(output/"tmp").mkdir()
    previous={k:os.environ.get(k) for k in ("TMP","TEMP")};os.environ["TMP"]=os.environ["TEMP"]=str(output/"tmp")
    started=time.perf_counter();cpu=time.process_time();dump(output/"plan.json",plan)
    dump(output/"progress.json",{"stage":"fitting","completed_fits":0,"planned_fits":1,"detail":name,"updated_utc":datetime.now(timezone.utc).isoformat()})
    try:
        with training_io(output) as accesses:
            y,valid,hard=validate_training(name,training_bank,emb,labels,refs)
            n=len(training_bank)
            if name=="cross_tree":
                from sklearn.ensemble import HistGradientBoostingClassifier
                from threadpoolctl import threadpool_limits
                from text_features import extract_features
                from cross_candidate_features import extract_features as cross_features
                from tree_model import export_trees,tree_scores
                indices=np.array(plan["cross_feature_indices"],np.int32)
                x=np.concatenate((extract_features(training_bank),cross_features(training_bank)[...,indices]),axis=-1)
                mask=valid[hard];weights=np.broadcast_to(4./np.maximum(mask.sum(axis=1,keepdims=True),1),mask.shape)[mask]
                with threadpool_limits(limits=4):
                    model=HistGradientBoostingClassifier(**plan["tree_parameters"]).fit(x[hard][mask],y[hard][mask],sample_weight=weights)
                    native=model.decision_function(x.reshape(-1,x.shape[-1])).reshape(n,4)
                asset=export_trees(model);asset["cross_feature_indices"]=indices;asset["feature_count"]=np.array(x.shape[-1],np.int32)
                replay=tree_scores(x.reshape(-1,x.shape[-1]),asset).reshape(n,4);curve=None
            else:
                import torch
                from optimization_graph import train,export,native_scores,normalizer
                from equation_graph_model import graph_inputs,numpy_forward
                from sequence_model import numeric_inputs,transform_numeric
                from sequence_experiment import group_structure
                torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
                if torch.version.cuda is not None:raise RuntimeError("CPU-only runtime required")
                weights,gv,good,group_valid,eligible=group_structure(training_bank,y)
                assert np.array_equal(valid,group_valid)
                nodes,adjacency,mask,coverage=graph_inputs(training_bank,True)
                numeric=numeric_inputs(training_bank,emb);transform=normalizer(numeric,valid,np.arange(n))
                numeric=transform_numeric(numeric,transform)
                inputs=tuple(torch.from_numpy(a) for a in (nodes,adjacency,mask,numeric))
                structure=tuple(torch.from_numpy(a) for a in (weights,gv,good))
                model,curve,_=train(inputs,structure,np.flatnonzero(eligible),name=="equation_gcn",plan["epochs"])
                native=native_scores(model,inputs);asset=export(model,transform)
                replay=numpy_forward(nodes,adjacency,mask,numeric,asset)
                dump(output/"training_curve.json",curve)
                dump(output/"graph_coverage.json",{"candidates":len(coverage),"with_equations":sum(r["equation_nodes"]>0 for r in coverage),
                    "mean_equations":float(np.mean([r["equation_nodes"] for r in coverage]))})
            error=float(np.max(np.abs(replay-native)));limit=1e-9 if name=="cross_tree" else 1e-4
            if error>limit:raise RuntimeError("NumPy replay discrepancy: "+str(error))
            scale=float(max(np.std((native-native.mean(axis=1,keepdims=True))[valid]),1e-6))
            asset["score_scale"]=np.array(scale);asset_path=output/"model.npz";np.savez_compressed(asset_path,**asset)
            with zipfile.ZipFile(asset_path) as archive:uncompressed=sum(x.file_size for x in archive.infolist())
            from selection import select
            from common import parse_answer
            native_choice=select(training_bank,native,"mean");replay_choice=select(training_bank,replay,"mean")
            changes=sum(parse_answer(r["candidates"][a]["text"])!=parse_answer(r["candidates"][b]["text"]) for r,a,b in zip(training_bank,native_choice,replay_choice))
            manifest={"format":FORMAT,"branch":name,"asset_path":"model.npz","asset_sha256":sha(asset_path),"score_scale":scale,
                "inference_sources_sha256":inference_sources(name),"plan_sha256":sha(output/"plan.json"),"training_questions":n,
                "model_fit_calls":1,"full_training_refit":True,"validation_evaluations":0,"test_access":False,
                "numpy_export_max_error":error,"numpy_native_training_answer_differences":changes,
                "asset_uncompressed_bytes":uncompressed,"array_payload_bytes":sum(a.nbytes for a in asset.values()),
                "wall_seconds":time.perf_counter()-started,"cpu_seconds":time.process_time()-cpu,
                "data_files_read":sorted(accesses),"created_utc":datetime.now(timezone.utc).isoformat(),"threads":4,
                "resource_status":"Asset measured; complete fresh prediction/scoring resource qualification remains separate"}
            if any(sha(ROOT/file)!=value for file,value in plan["source_study_sha256"].items()):raise RuntimeError("Study source changed during fit")
            dump(output/"manifest.json",manifest)
            dump(output/"progress.json",{"stage":"complete","completed_fits":1,"planned_fits":1,"detail":name,"updated_utc":datetime.now(timezone.utc).isoformat()})
        return output/"manifest.json"
    except Exception as exc:
        dump(output/"failure.json",{"error":repr(exc),"seconds":time.perf_counter()-started,"validation_access":False,"test_access":False})
        raise
    finally:
        for key,value in previous.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--branch",choices=BRANCHES,required=True)
    action=parser.add_mutually_exclusive_group(required=True);action.add_argument("--plan",action="store_true");action.add_argument("--fit",action="store_true")
    parser.add_argument("--output");args=parser.parse_args()
    if args.plan:print(json.dumps(refit_plan(args.branch),indent=2));return
    if not args.output:parser.error("--fit requires --output")
    from common import load_inputs,DATA,read_jsonl,parse_answer
    bank,arrays=load_inputs("train");refs=read_jsonl(DATA/"gsm8k_train.jsonl")
    labels=np.array([[parse_answer(c["text"]) is not None and parse_answer(c["text"])==parse_answer(r["answer"]) for c in row["candidates"]] for row,r in zip(bank,refs)],np.float32)
    print(train_branch(args.branch,args.output,bank,arrays["embeddings"],labels,refs))

if __name__=="__main__":main()
