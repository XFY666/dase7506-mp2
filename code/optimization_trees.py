"""Seven fixed tree controls under two fresh three-fold training partitions.

No validation or test inputs are read. The only fitted objects are the 42
declared HGBT learners; all feature extractors are reference-free. This is
exploratory follow-up research, not an untouched estimate of the entire search.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_key]="4"

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from common import DATA, read_jsonl, parse_answer
from cross_candidate_features import extract_features as cross_features, FEATURE_NAMES as CROSS_NAMES, FEATURE_GROUPS, self_checks
from text_features import extract_features as base_features, FEATURE_NAMES as BASE_NAMES
from tree_model import export_trees, tree_scores
from selection import select

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"runs"/"optimization_trees"
FOLD_SEEDS=(20261027,20261028)
TREE_SEED=20261013
CONFIGS=[
    {"name":"base_all_candidates","policy":"all_candidates","blocks":[]},
    {"name":"base_all_valid","policy":"all_valid","blocks":[]},
    {"name":"base_hard_valid","policy":"hard_valid","blocks":[]},
    {"name":"base_hard_valid_question_weighted","policy":"hard_valid_question_weighted","blocks":[]},
    {"name":"weighted_D1","policy":"hard_valid_question_weighted","blocks":["D1"]},
    {"name":"weighted_D1_D2","policy":"hard_valid_question_weighted","blocks":["D1","D2"]},
    {"name":"weighted_D1_D2_D3","policy":"hard_valid_question_weighted","blocks":["D1","D2","D3"]},
]
PARAMS={"max_iter":200,"max_leaf_nodes":31,"learning_rate":.05,"l2_regularization":10.,
        "min_samples_leaf":30,"early_stopping":False,"random_state":TREE_SEED}


def dump(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def progress(stage,done,started,detail):
    dump(OUT/"progress.json",{"stage":stage,"completed_fits":done,"planned_fits":42,
        "detail":detail,"elapsed_seconds":time.perf_counter()-started,
        "updated_utc":datetime.now(timezone.utc).isoformat()})


def protocol():
    return {"schema_version":1,"scope":"Training-only B2 sample-policy controls and D1/D2/D3 feature ablations",
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "data_scope":["train"],"validation_access":False,"test_access":False,
        "unit":"Question; all four candidates kept together",
        "fold_seeds":FOLD_SEEDS,"folds":3,
        "fold_assignment":"fold[default_rng(seed).permutation(N)] = arange(N) % 3",
        "configurations":CONFIGS,"tree_parameters":PARAMS,"planned_model_fits":42,
        "learned_transform_fits":0,"full_refit_fits":0,"feature_cache":"Only deterministic train inputs; no reference fields passed to extractors",
        "base75":"Exact original text_features.extract_features shared by all arms, including its existing float32 centering; no precision wrapper or base feature edit",
        "feature_definition":{"names":CROSS_NAMES,"groups":FEATURE_GROUPS,
            "D1":"Directed cross-answer derived-number containment and early-stop proxies; question numbers and explicit final-answer markers excluded",
            "D2":"Shared non-final derived numbers supported by distinct parsed-answer groups, coverage, order and leave-own-group consensus",
            "D3":"Six question-target keyword categories crossed with the outermost operation of the last parseable equality",
            "numeric_equivalence":"Fractions represented exactly where possible; relative/absolute tolerance1e-6 for numeric mention matching only; official exact parser still controls answer grouping/labels",
            "limits":{"characters":24000,"number_mentions_per_text":128},
            "caveat":"Numeric occurrence and lexical target cues do not prove semantic implication or step validity"},
        "sample_policies":{
            "all_candidates":"all four rows from each fitting question, including parser-invalid candidates; reproduces deployed tree sampling",
            "all_valid":"valid rows from all fitting questions, unit row weight",
            "hard_valid":"valid rows from questions with at least two distinct valid answers, unit row weight",
            "hard_valid_question_weighted":"same hard/valid rows, weight4/(valid candidates in question), each question weight4"},
        "aggregation":"Official-parser same-answer group mean, zero consensus bonus",
        "score_normalization":"Raw scores centered across all four candidates; std at valid candidate entries from all outer-fitting questions; held-out questions never enter the scale",
        "reporting":"Each repeat separately, paired repair/regression and question-balanced group cross-entropy; repeated questions are not independent new observations",
        "training_selection_order":["highest summed hard-correct across both repeats","lowest mean normalized group CE","fewest added feature columns","earlier registered config"],
        "deployment":"No promotion, validation, test or full-training refit in this driver; all fold trees exported for later inspection",
        "adaptation":"No extra configurations or feature tuning after scores; implementation repairs recorded separately",
        "threads":4,"minimum_independent_evaluation":"New folds reduce split dependence, but all hypotheses and old recipes were informed by earlier research; OOF selection remains exploratory"}


def load_training():
    bank=read_jsonl(DATA/"candidates"/"gsm8k_train.jsonl")
    refs=read_jsonl(DATA/"gsm8k_train.jsonl")
    if len(bank)!=len(refs):raise ValueError("Training alignment mismatch")
    answers=[];labels=[]
    for row,ref in zip(bank,refs):
        if row["id"]!=ref["id"] or row["question"]!=ref["question"]:raise ValueError("Training alignment mismatch")
        aa=[parse_answer(c["text"]) for c in row["candidates"]]
        gold=parse_answer(ref["answer"])
        answers.append(aa);labels.append([a is not None and a==gold for a in aa])
    valid=np.array([[a is not None for a in aa] for aa in answers],dtype=bool)
    hard=np.array([len({a for a in aa if a is not None})>=2 for aa in answers],dtype=bool)
    return bank,answers,np.asarray(labels,dtype=bool),valid,hard


def group_ce(scores,answers,labels,mask):
    losses=[]
    for z,aa,y,include in zip(scores,answers,labels,mask):
        if not include:continue
        groups={a:[k for k in range(4) if aa[k]==a] for a in sorted({a for a in aa if a is not None})}
        values=np.array([np.mean(z[ix],dtype=np.float64) for ix in groups.values()])
        target=np.array([bool(y[ix].any()) for ix in groups.values()])
        m=float(values.max());logden=m+float(np.log(np.exp(values-m).sum()))
        positive=values[target];m=float(positive.max());lognum=m+float(np.log(np.exp(positive-m).sum()))
        losses.append(logden-lognum)
    return float(np.mean(losses))


def evaluate(bank,answers,labels,hard,z):
    choices=np.asarray(select(bank,z,"mean",0.),dtype=np.int32)
    correct=labels[np.arange(len(bank)),choices]
    return {"correct":int(correct.sum()),"n":len(bank),"hard_correct":int(correct[hard].sum()),
            "hard_n":int(hard.sum()),"group_ce_hard":group_ce(z,answers,labels,hard)},choices,correct


def paired(reference,current,hard):
    return {"repaired":int((~reference & current & hard).sum()),
            "regressed":int((reference & ~current & hard).sum()),
            "gain":int((current[hard].sum()-reference[hard].sum())),
            "both_correct":int((reference & current & hard).sum()),
            "both_wrong":int((~reference & ~current & hard).sum())}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checks-only",action="store_true")
    args=parser.parse_args();started=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    if args.checks_only:
        result=self_checks();dump(OUT/"synthetic_checks.json",result);print(json.dumps(result));return
    if (OUT/"protocol.json").exists():raise SystemExit("Protocol exists; preserve evidence and use a separate research copy")
    # Written before input statistics, model fitting or training-score inspection.
    dump(OUT/"protocol.json",protocol())
    source_names=["optimization_trees.py","cross_candidate_features.py","text_features.py","quantitative_features.py","common.py","tree_model.py","selection.py"]
    hashes={name:digest(ROOT/name) for name in source_names}
    dump(OUT/"source_hashes.json",hashes)
    inputs={str(p.relative_to(ROOT)):digest(p) for p in (DATA/"candidates"/"gsm8k_train.jsonl",DATA/"gsm8k_train.jsonl")}
    dump(OUT/"input_hashes.json",inputs)
    progress("checks",0,started,"Synthetic numeric/marker/directionality/permutation checks")
    checks=self_checks();dump(OUT/"synthetic_checks.json",checks)
    progress("features",0,started,"Computing train-only original75 and cross-candidate features")
    bank,answers,labels,valid,hard=load_training();n=len(bank)
    assert labels.any(axis=1).all() and (labels<=valid).all()
    feature_started=time.perf_counter();base=base_features(bank);cross=cross_features(bank)
    feature_seconds=time.perf_counter()-feature_started
    assert base.shape==(n,4,75) and cross.shape==(n,4,len(CROSS_NAMES))
    subset=np.arange(min(32,n));perm=[2,0,3,1]
    changed=[dict(bank[i],id="renamed",candidates=[bank[i]["candidates"][j] for j in perm]) for i in subset]
    assert np.array_equal(cross[subset][:,perm],cross_features(changed))
    assert np.array_equal(cross[subset][::-1],cross_features([bank[i] for i in subset[::-1]]))
    checks.update(training32_candidate_permutation_exact=True,training32_question_permutation_exact=True)
    dump(OUT/"synthetic_checks.json",checks)
    np.savez_compressed(OUT/"training_feature_cache.npz",ids=np.array([r["id"] for r in bank]),base75=base,cross=cross,
                        base_feature_names=np.array(BASE_NAMES),cross_feature_names=np.array(CROSS_NAMES))
    dump(OUT/"feature_summary.json",{"n":n,"feature_seconds":feature_seconds,"base_features":75,"cross_features":len(CROSS_NAMES),
        "names":CROSS_NAMES,"groups":FEATURE_GROUPS,"nonzero_fraction":np.mean(cross!=0,axis=(0,1)).tolist(),
        "within_question_varying_fraction":np.mean(np.ptp(cross,axis=1)!=0,axis=0).tolist(),
        "feature_min":cross.min(axis=(0,1)).tolist(),"feature_max":cross.max(axis=(0,1)).tolist()})
    folds=np.empty((2,n),dtype=np.int8)
    for repeat,seed in enumerate(FOLD_SEEDS):folds[repeat,np.random.default_rng(seed).permutation(n)]=np.arange(n)%3
    raw=np.full((2,n,4,len(CONFIGS)),np.nan,dtype=np.float64);normalized=raw.copy()
    scales=np.zeros((2,3,len(CONFIGS)),dtype=np.float64);records=[];done=0
    with threadpool_limits(limits=4):
        for repeat,seed in enumerate(FOLD_SEEDS):
            for fold in range(3):
                fitting=np.where(folds[repeat]!=fold)[0];held=np.where(folds[repeat]==fold)[0]
                for config_index,config in enumerate(CONFIGS):
                    progress("fitting",done,started,f"repeat {repeat+1}/2 fold {fold+1}/3: {config['name']}")
                    indices=[i for block in config["blocks"] for i in FEATURE_GROUPS[block]]
                    x=np.concatenate((base,cross[...,indices]),axis=-1) if indices else base
                    ids=fitting[hard[fitting]] if config["policy"].startswith("hard") else fitting
                    mask=valid[ids] if config["policy"]!="all_candidates" else np.ones((len(ids),4),dtype=bool)
                    weights=None
                    if config["policy"]=="hard_valid_question_weighted":
                        weights=np.broadcast_to(4./np.maximum(mask.sum(axis=1,keepdims=True),1),mask.shape)[mask]
                    tick=time.perf_counter();model=HistGradientBoostingClassifier(**PARAMS).fit(x[ids][mask],labels[ids][mask],sample_weight=weights)
                    fit_seconds=time.perf_counter()-tick;asset=export_trees(model)
                    asset.update(cross_feature_indices=np.asarray(indices,dtype=np.int32),feature_count=np.array(x.shape[-1],dtype=np.int32))
                    path=OUT/f"repeat{repeat}_fold{fold}_{config['name']}.npz";np.savez_compressed(path,**asset)
                    tick=time.perf_counter();z=model.decision_function(x.reshape(-1,x.shape[-1])).reshape(n,4)
                    export=tree_scores(x[held].reshape(-1,x.shape[-1]),asset).reshape(len(held),4)
                    error=float(np.max(np.abs(export-z[held])));assert error<1e-9
                    centered=z[fitting]-np.mean(z[fitting],axis=1,keepdims=True)
                    scale=float(max(np.std(centered[valid[fitting]]),1e-6));scales[repeat,fold,config_index]=scale
                    raw[repeat,held,:,config_index]=export;normalized[repeat,held,:,config_index]=export/scale
                    score_seconds=time.perf_counter()-tick
                    hb=[bank[i] for i in held];ha=[answers[i] for i in held]
                    metrics,_,_=evaluate(hb,ha,labels[held],hard[held],export/scale)
                    records.append({"repeat":repeat,"fold_seed":seed,"fold":fold,"configuration":config,
                        "features":x.shape[-1],"fit_questions":len(ids),"fit_candidate_rows":int(mask.sum()),
                        "sample_weight_sum":float(weights.sum()) if weights is not None else int(mask.sum()),
                        "outer_fit_ids_sha256":array_digest(fitting.astype(np.int64)),"actual_fit_ids_sha256":array_digest(ids.astype(np.int64)),
                        "heldout_ids_sha256":array_digest(held.astype(np.int64)),"scale":scale,"scale_fitting_questions":len(fitting),
                        "training_seconds":fit_seconds,"score_and_export_check_seconds":score_seconds,
                        "numpy_export_max_error":error,"asset":path.name,"asset_bytes":sum(v.nbytes for v in asset.values()),
                        "asset_file_sha256":digest(path),"metrics":metrics})
                    done+=1;dump(OUT/"fit_records.json",records)
                    progress("fitting",done,started,f"Completed {config['name']} repeat {repeat+1} fold {fold+1}")
                    print(json.dumps({"fit":done,"of":42,"repeat":repeat,"fold":fold,"name":config["name"],"hard_correct":metrics["hard_correct"],"hard_n":metrics["hard_n"]}),flush=True)
    assert np.isfinite(raw).all() and np.isfinite(normalized).all()
    choices=np.zeros((2,n,len(CONFIGS)),dtype=np.int32);correct=np.zeros_like(choices,dtype=bool)
    results=[]
    for c,config in enumerate(CONFIGS):
        repeats=[]
        for r in range(2):
            metrics,choices[r,:,c],correct[r,:,c]=evaluate(bank,answers,labels,hard,normalized[r,:,:,c])
            repeats.append(metrics)
        results.append({"configuration":config,"repeats":repeats,
            "summed_hard_correct":sum(m["hard_correct"] for m in repeats),"mean_group_ce":float(np.mean([m["group_ce_hard"] for m in repeats]))})
    for c,result in enumerate(results):
        result["paired_vs_all_candidates"]=[paired(correct[r,:,0],correct[r,:,c],hard) for r in range(2)]
        result["paired_vs_weighted_base"]=[paired(correct[r,:,3],correct[r,:,c],hard) for r in range(2)]
        result["per_fold_vs_weighted_base"]=[{"repeat":r,"fold":f,**paired(correct[r,:,3],correct[r,:,c],hard&(folds[r]==f))} for r in range(2) for f in range(3)]
    chosen=max(range(len(CONFIGS)),key=lambda c:(results[c]["summed_hard_correct"],-results[c]["mean_group_ce"],-len(CONFIGS[c]["blocks"]),-c))
    np.savez_compressed(OUT/"oof_logits.npz",ids=np.array([r["id"] for r in bank]),raw=raw,normalized=normalized,labels=labels,valid=valid,hard=hard,
        folds=folds,scales=scales,choices=choices,correct=correct,configuration_names=np.array([c["name"] for c in CONFIGS]),fold_seeds=np.array(FOLD_SEEDS))
    dump(OUT/"oof_schema.json",{"raw_and_normalized_shape":[2,n,4,len(CONFIGS)],"axes":["repeat","question","candidate","configuration"],
        "choices_and_correct_axes":["repeat","question","configuration"],"folds_axes":["repeat","question"],"scales_axes":["repeat","fold","configuration"],
        "label_scope":"train only","ids_role":"alignment only","score_normalization":"scale fitted without outer-heldout questions"})
    summary={"configurations":results,"training_selected_configuration":CONFIGS[chosen],"selected_config_index":chosen,
        "model_fit_calls":done,"transform_fit_calls":0,"full_refit_calls":0,"validation_evaluations":0,"test_access":False,
        "training_questions":n,"hard_questions":int(hard.sum()),"invalid_candidate_rows":int((~valid).sum()),
        "feature_seconds":feature_seconds,"fit_seconds":sum(r["training_seconds"] for r in records),"elapsed_seconds":time.perf_counter()-started,
        "threads":4,"python":platform.python_version(),"numpy":np.__version__,"sklearn":sklearn.__version__,
        "source_hashes":hashes,"input_hashes":inputs,"source_hashes_unchanged":all(digest(ROOT/name)==h for name,h in hashes.items()),
        "interpretation":"Repeated OOF questions are correlated; sums are ranking criteria, not an independent sample size. No deployment or validation decision is made."}
    assert summary["source_hashes_unchanged"]
    dump(OUT/"summary.json",summary);dump(OUT/"results.json",results)
    progress("complete",done,started,"42 fixed training fits finished; no validation/test/full refit")
    print(json.dumps({"summary":str(OUT/"summary.json"),"winner":CONFIGS[chosen],"results":[[r["configuration"]["name"],[x["hard_correct"] for x in r["repeats"]]] for r in results]},indent=2),flush=True)


if __name__=="__main__":
    main()
