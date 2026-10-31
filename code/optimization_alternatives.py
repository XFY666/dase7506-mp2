"""Queued, registered training-only projected-neighbor and expression study.

Use --prepare to register the immutable protocol and run synthetic checks only.
Use --run from the project queue to execute the 6 outer partitions. No validation
or test access, generation, external weights, or automatic deployment is included.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_key]="4"

import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import sklearn
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from bounded_expression_features import extract_features as expression_features, FEATURE_NAMES as EXPRESSION_NAMES, self_checks as expression_checks
from common import DATA, load_inputs
from optimization_trees import load_training, evaluate, paired, dump, array_digest, PARAMS
from projected_neighbors import centered_embeddings, project, neighbor_logits, self_checks as neighbor_checks
from text_features import extract_features as base_features, FEATURE_NAMES as BASE_NAMES
from tree_model import export_trees,tree_scores

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"runs"/"optimization_alternatives"
SEEDS=(20261027,20261028)
KS=(7,21,63)
CONFIGS=[{"name":f"PLS8_neighbors_k{k}","family":"neighbors","k":k} for k in KS]+[
    {"name":"weighted_base75","family":"tree","expression_features":False},
    {"name":"weighted_base75_expression","family":"tree","expression_features":True}]
SOURCES=("optimization_alternatives.py","projected_neighbors.py","bounded_expression_features.py",
         "cross_candidate_features.py","optimization_trees.py","text_features.py","quantitative_features.py",
         "common.py","tree_model.py","selection.py")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def progress(stage,completed,started,detail,projections=0,indexes=0):
    dump(OUT/"progress.json",{"stage":stage,"completed_fits":completed,"planned_fits":12,
        "completed_projection_fits":projections,"planned_projection_fits":6,
        "completed_index_builds":indexes,"planned_index_builds":6,"detail":detail,
        "elapsed_seconds":time.perf_counter()-started,"updated_utc":datetime.now(timezone.utc).isoformat()})


def protocol():
    return {"schema_version":1,"created_utc":datetime.now(timezone.utc).isoformat(),
        "stage":"F1 projected-neighbor head and D4 bounded arithmetic evidence",
        "data_scope":["train"],"validation_access":False,"test_access":False,
        "fold_seeds":SEEDS,"folds":3,"fold_assignment":"fold[default_rng(seed).permutation(N)]=arange(N)%3",
        "independent_unit":"question, all four candidates together",
        "configurations":CONFIGS,"planned_model_fits":12,"planned_pls_fits":6,
        "planned_index_builds":6,"full_refits":0,"threads":4,
        "F1":{"projection":"PLSRegression(n_components=8,scale=False,max_iter=500,tol=1e-6), fitted only on outer-fitting hard questions and valid candidate rows; x and y centered within valid candidates of each question",
              "export":"float64 mean and rotations, used identically for fitting index and query; one shared projection/index per outer partition",
              "index":"projected valid hard fitting candidates, binary final-correctness targets, and internal question membership for calibration only",
              "metric":"Euclidean distance, weights1/max(distance,1e-3), stable distance sort, binary probability clipped[1e-4,1-1e-4] then logit",
              "neighbors":KS,"calibration":"Every fitting-side query excludes ALL index entries from its own question. Held-out scores use no exclusions and never fit the projection/index/scale.",
              "query_inputs":"Provided embeddings and parsed validity only; candidate/question IDs never used as prediction features",
              "index_size":"Report uncompressed asset bytes; must be<=16MiB per candidate head, ensemble totals checked only at later freeze"},
        "D4":{"source":"question numeric/word-number mentions only; Fraction arithmetic; no reference, labels or candidate answers guide search",
              "initial_quantities":"at most8 occurrences: all if<=8, otherwise first4+last4; occurrence masks prevent reusing one source quantity in an expression",
              "constants":[1,2,100],"operations":["add","subtract both orders","multiply","divide both orders when nonzero"],
              "max_tree_depth":3,"retained_by_depth":[160,320,480],"total_state_cap":1000,
              "attempt_slots_by_depth":[800,2000,3200],"total_attempt_slots_cap":6000,
              "value_bounds":"abs(value)<=1e9 and denominator<=1e6 for derived states",
              "search_order":"deterministic coprime spread over operand pairs, deduplicate(value,source-mask); retain by operation count, constant use, source coverage, rational complexity",
              "features":EXPRESSION_NAMES,"equality":"exact Fraction or1e-6relative/absolute tolerance for feature matching; official exact parser still controls evaluation",
              "interpretation":"Incomplete arithmetic reachability; reached values need not solve the word problem, unreached values are not proved impossible",
              "cost_probe":"First16 training questions, feature extraction only, no learner fit or model choice; report projected full cost as estimate, then process full training bank",
              "coverage":"quantities truncated, state/attempt counts and caps, candidate reachability, feature ranges/nonzero/within-question variation"},
        "trees":{"parameters":PARAMS,"sampling":"only hard fitting questions and valid rows, sample weight4/validcandidatecount perquestion","base":"exact original75 features; matched+29expression features only"},
        "score_scale":"std of all-four-centered scores on valid candidates from all outer-fitting questions, separately per model; neighbor fitting predictions leave own question out",
        "aggregation":"same-answer group mean, zero consensus bonus",
        "selection_order":["summed hard correct across repeats","lower mean normalized group CE","earlier registered config"],
        "adaptation":"No parameter/feature expansion after observing results. Repairs preserved explicitly. No validation/test/fullrefit/deployment in this stage.",
        "interpretation":"Repeated questions across two fold seeds are not independent new samples; this is conditional exploratory research following previous study selection."}


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/"protocol.json").exists():raise SystemExit("Registered protocol exists; preserve it")
    checks={"neighbors":neighbor_checks(),"expressions":expression_checks()}
    dump(OUT/"synthetic_checks.json",checks)
    dump(OUT/"protocol.json",protocol())
    hashes={name:digest(ROOT/name) for name in SOURCES}
    dump(OUT/"source_hashes.json",hashes)
    dump(OUT/"input_hashes.json",{str(p.relative_to(ROOT)):digest(p) for p in (
        DATA/"candidates"/"gsm8k_train.jsonl",DATA/"candidates"/"gsm8k_train.npz",DATA/"gsm8k_train.jsonl")})
    dump(OUT/"progress.json",{"stage":"ready","completed_fits":0,"planned_fits":12,
        "planned_projection_fits":6,"planned_index_builds":6,"elapsed_seconds":0,
        "detail":"Protocol and source/input hashes registered. Synthetic checks only; awaiting queued --run.",
        "updated_utc":datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"stage":"ready","command":"python optimization_alternatives.py --run","model_fits":12,"projection_fits":6,"index_builds":6,"new_training_or_evaluation":False}))


def score_scale(z,valid,fitting):
    values=np.asarray(z,dtype=np.float64)[fitting]
    centered=values-values.mean(axis=1,keepdims=True)
    return float(max(np.std(centered[valid[fitting]]),1e-6))


def run():
    started=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    if not (OUT/"protocol.json").exists():raise SystemExit("Run --prepare before queue execution")
    sources=json.loads((OUT/"source_hashes.json").read_text(encoding="utf-8"))
    inputs=json.loads((OUT/"input_hashes.json").read_text(encoding="utf-8"))
    if any(digest(ROOT/name)!=value for name,value in {**sources,**inputs}.items()):raise SystemExit("Registered source/input hash changed")
    if (OUT/"summary.json").exists():
        print(json.dumps({"stage":"already_complete","summary":str(OUT/"summary.json")}));return
    if (OUT/"run_started.json").exists():raise SystemExit("An incomplete attempt exists; preserve evidence before explicit recovery")
    dump(OUT/"run_started.json",{"utc":datetime.now(timezone.utc).isoformat(),"protocol_sha256":digest(OUT/"protocol.json")})
    progress("features",0,started,"Loading training inputs and performing the registered16-question expression cost probe")
    bank,answers,labels,valid,hard=load_training();input_bank,arrays=load_inputs("train")
    assert [r["id"] for r in input_bank]==[r["id"] for r in bank]
    n=len(bank);embeddings=arrays["embeddings"]
    tick=time.perf_counter();probe,probe_coverage=expression_features(bank[:16],True);probe_seconds=time.perf_counter()-tick
    dump(OUT/"cost_probe.json",{"questions":16,"model_fits":0,"seconds":probe_seconds,
        "estimated_full_feature_seconds":probe_seconds*n/16,"estimate_not_measurement":True,"coverage":probe_coverage})
    progress("features",0,started,"Extracting full-training bounded expression and original75 features")
    tick=time.perf_counter();base=base_features(bank);expr,coverage=expression_features(bank,True);feature_seconds=time.perf_counter()-tick
    assert np.array_equal(probe,expr[:16])
    perm=[2,0,3,1];changed=[dict(r,id="unused",candidates=[r["candidates"][k] for k in perm]) for r in bank[:16]]
    assert np.array_equal(expr[:16][:,perm],expression_features(changed))
    np.savez_compressed(OUT/"training_feature_cache.npz",ids=np.array([r["id"] for r in bank]),base75=base,expression=expr,feature_names=np.array(EXPRESSION_NAMES))
    dump(OUT/"feature_coverage.json",{"feature_seconds":feature_seconds,"questions":n,"features":EXPRESSION_NAMES,
        "nonzero_fraction":np.mean(expr!=0,axis=(0,1)).tolist(),"within_question_varying_fraction":np.mean(np.ptp(expr,axis=1)!=0,axis=0).tolist(),
        "mean_states_by_depth":np.mean([r["states_by_depth"] for r in coverage],axis=0).tolist(),
        "retention_capped_fraction":np.mean([r["retention_capped"] for r in coverage],axis=0).tolist(),
        "question_quantity_truncation_fraction":float(np.mean([r["truncated"] for r in coverage])),
        "max_states":max(sum(r["states_by_depth"]) for r in coverage),
        "max_attempt_slots":max(sum(r["arithmetic_attempt_slots_by_depth"]) for r in coverage),
        "training16_candidate_permutation_exact":True})
    folds=np.empty((2,n),dtype=np.int8)
    for r,seed in enumerate(SEEDS):folds[r,np.random.default_rng(seed).permutation(n)]=np.arange(n)%3
    raw=np.full((2,n,4,len(CONFIGS)),np.nan,dtype=np.float64);normal=raw.copy()
    scales=np.zeros((2,3,len(CONFIGS)));records=[];projection_records=[];fits=0;projections=0
    with threadpool_limits(limits=4):
        for repeat,seed in enumerate(SEEDS):
            for fold in range(3):
                fitting=np.where(folds[repeat]!=fold)[0];held=np.where(folds[repeat]==fold)[0]
                fit_hard=fitting[hard[fitting]];mask=valid[fit_hard]
                progress("projection",fits,started,f"PLS8 and leave-own-question-out neighbor calibration: repeat{repeat+1} fold{fold+1}",projections,projections)
                tick=time.perf_counter()
                centered=centered_embeddings(embeddings[fit_hard],mask)
                centered_labels=labels[fit_hard]-np.sum(labels[fit_hard]*mask,axis=1,keepdims=True)/np.maximum(mask.sum(axis=1,keepdims=True),1)
                pls=PLSRegression(n_components=8,scale=False,max_iter=500,tol=1e-6).fit(centered[mask],centered_labels[mask])
                asset={"projection_mean":np.array(pls._x_mean,dtype=np.float64),"projection_rotations":np.array(pls.x_rotations_,dtype=np.float64)}
                projection_seconds=time.perf_counter()-tick
                projected=project(embeddings,valid,asset)
                assert np.allclose(projected[fit_hard][mask][:128],pls.transform(centered[mask][:128]),rtol=1e-12,atol=1e-12)
                asset.update(index_vectors=projected[fit_hard][mask],index_labels=labels[fit_hard][mask].astype(np.uint8),
                             index_question=np.repeat(fit_hard,4)[mask.ravel()].astype(np.int64))
                assert not np.intersect1d(asset["index_question"],held).size
                asset_bytes=sum(a.nbytes for a in asset.values());assert asset_bytes<16*1024*1024
                path=OUT/f"repeat{repeat}_fold{fold}_PLS8_index.npz";np.savez_compressed(path,**asset)
                tick=time.perf_counter();z=np.zeros((n,4,len(KS)),dtype=np.float64)
                z[fitting]=neighbor_logits(projected[fitting].reshape(-1,8),asset,KS,np.repeat(fitting,4)).reshape(len(fitting),4,len(KS))
                z[held]=neighbor_logits(projected[held].reshape(-1,8),asset,KS).reshape(len(held),4,len(KS))
                neighbor_seconds=time.perf_counter()-tick
                projections+=1
                projection_records.append({"repeat":repeat,"fold_seed":seed,"fold":fold,"fit_questions":len(fit_hard),
                    "fit_rows":int(mask.sum()),"fit_question_sha256":array_digest(fit_hard.astype(np.int64)),
                    "held_question_sha256":array_digest(held.astype(np.int64)),"projection_seconds":projection_seconds,
                    "neighbor_score_seconds":neighbor_seconds,"asset":path.name,"asset_sha256":digest(path),
                    "uncompressed_asset_bytes":asset_bytes,"pls_iterations":list(pls.n_iter_),"own_question_excluded_in_calibration":True})
                for c,k in enumerate(KS):
                    scale=score_scale(z[:,:,c],valid,fitting);scales[repeat,fold,c]=scale
                    raw[repeat,held,:,c]=z[held,:,c];normal[repeat,held,:,c]=z[held,:,c]/scale
                    metrics,_,_=evaluate([bank[i] for i in held],[answers[i] for i in held],labels[held],hard[held],z[held,:,c]/scale)
                    records.append({"repeat":repeat,"fold":fold,"configuration":CONFIGS[c],"scale":scale,"metrics":metrics,
                        "shared_index_asset":path.name,"independent_learner_fit":False})
                dump(OUT/"projection_records.json",projection_records)
                for c in (3,4):
                    config=CONFIGS[c];progress("fitting",fits,started,f"{config['name']}: repeat{repeat+1} fold{fold+1}",projections,projections)
                    x=np.concatenate((base,expr),axis=-1) if config["expression_features"] else base
                    weights=np.broadcast_to(4./np.maximum(mask.sum(axis=1,keepdims=True),1),mask.shape)[mask]
                    tick=time.perf_counter();model=HistGradientBoostingClassifier(**PARAMS).fit(x[fit_hard][mask],labels[fit_hard][mask],sample_weight=weights)
                    seconds=time.perf_counter()-tick;tree=export_trees(model)
                    tree.update(expression_features=np.array(int(config["expression_features"]),dtype=np.int8),feature_count=np.array(x.shape[-1],dtype=np.int32))
                    path=OUT/f"repeat{repeat}_fold{fold}_{config['name']}.npz";np.savez_compressed(path,**tree)
                    scores=model.decision_function(x.reshape(-1,x.shape[-1])).reshape(n,4)
                    exported=tree_scores(x[held].reshape(-1,x.shape[-1]),tree).reshape(len(held),4)
                    error=float(np.max(np.abs(exported-scores[held])));assert error<1e-9
                    scale=score_scale(scores,valid,fitting);scales[repeat,fold,c]=scale
                    raw[repeat,held,:,c]=exported;normal[repeat,held,:,c]=exported/scale
                    metrics,_,_=evaluate([bank[i] for i in held],[answers[i] for i in held],labels[held],hard[held],exported/scale)
                    fits+=1;records.append({"repeat":repeat,"fold":fold,"configuration":config,"scale":scale,"metrics":metrics,
                        "model_fit_calls":1,"training_seconds":seconds,"fit_questions":len(fit_hard),"fit_rows":int(mask.sum()),
                        "numpy_export_max_error":error,"asset":path.name,"asset_sha256":digest(path),
                        "uncompressed_asset_bytes":sum(a.nbytes for a in tree.values()),"fit_question_sha256":array_digest(fit_hard.astype(np.int64))})
                dump(OUT/"fit_records.json",records)
                progress("fitting",fits,started,f"Completed partition repeat{repeat+1} fold{fold+1}",projections,projections)
                print(json.dumps({"repeat":repeat,"fold":fold,"trees_completed":fits,"projections_completed":projections}),flush=True)
    assert np.isfinite(raw).all() and np.isfinite(normal).all()
    choices=np.zeros((2,n,len(CONFIGS)),dtype=np.int32);correct=np.zeros_like(choices,dtype=bool);results=[]
    for c,config in enumerate(CONFIGS):
        repeated=[]
        for r in range(2):
            metrics,choices[r,:,c],correct[r,:,c]=evaluate(bank,answers,labels,hard,normal[r,:,:,c]);repeated.append(metrics)
        results.append({"configuration":config,"repeats":repeated,"summed_hard_correct":sum(x["hard_correct"] for x in repeated),
            "mean_group_ce":float(np.mean([x["group_ce_hard"] for x in repeated]))})
    for c,result in enumerate(results):
        result["paired_vs_weighted_base"]=[paired(correct[r,:,3],correct[r,:,c],hard) for r in range(2)]
        result["per_fold_vs_weighted_base"]=[{"repeat":r,"fold":f,**paired(correct[r,:,3],correct[r,:,c],hard&(folds[r]==f))} for r in range(2) for f in range(3)]
    chosen=max(range(len(CONFIGS)),key=lambda c:(results[c]["summed_hard_correct"],-results[c]["mean_group_ce"],-c))
    np.savez_compressed(OUT/"oof_logits.npz",ids=np.array([r["id"] for r in bank]),raw=raw,normalized=normal,labels=labels,valid=valid,hard=hard,
        folds=folds,scales=scales,choices=choices,correct=correct,configuration_names=np.array([x["name"] for x in CONFIGS]),fold_seeds=np.array(SEEDS))
    dump(OUT/"oof_schema.json",{"raw_normalized_axes":["repeat","question","candidate","configuration"],"shape":[2,n,4,len(CONFIGS)],"label_scope":"training only"})
    summary={"configurations":results,"training_selected_configuration":CONFIGS[chosen],"selected_config_index":chosen,
        "model_fit_calls":fits,"projection_fit_calls":projections,"retrieval_index_builds":projections,
        "full_refit_calls":0,"validation_evaluations":0,"test_access":False,"training_questions":n,"hard_questions":int(hard.sum()),
        "feature_seconds":feature_seconds,"feature_probe_seconds":probe_seconds,"elapsed_seconds":time.perf_counter()-started,
        "threads":4,"python":platform.python_version(),"numpy":np.__version__,"sklearn":sklearn.__version__,
        "sources_unchanged":all(digest(ROOT/name)==h for name,h in sources.items()),
        "source_hashes":sources,"input_hashes":inputs,"interpretation":"OOF winner selected on repeated historical training questions; no independent generalization claim"}
    assert summary["sources_unchanged"]
    dump(OUT/"results.json",results);dump(OUT/"summary.json",summary)
    progress("complete",fits,started,"Registered alternatives finished; no validation/test or fullrefit",projections,projections)
    print(json.dumps({"summary":str(OUT/"summary.json"),"winner":CONFIGS[chosen]},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    options=parser.add_mutually_exclusive_group(required=True)
    options.add_argument("--prepare",action="store_true");options.add_argument("--run",action="store_true")
    args=parser.parse_args()
    if args.prepare:prepare()
    else:run()


if __name__=="__main__":
    main()
