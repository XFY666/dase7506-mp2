"""Registered drop-one training augmentation with matched answer-group networks.

--prepare registers the study and runs synthetic checks only. --run is intended
for the automatic project queue. No validation/test or deployment change occurs.
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
from threadpoolctl import threadpool_limits

from answer_group_experiment import groups,make_features,fit_statistics,normalized,initialize,loss_gradient,scores,candidate_scores,scale_scores,save_asset
from answer_group_model import group_model_scores
from common import DATA,load_inputs,parse_answer
from optimization_trees import load_training,evaluate,paired,dump,array_digest
from text_features import extract_features

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"runs"/"optimization_subset"
FOLD_SEEDS=(20261027,20261028)
MODEL_SEEDS=(7506,17001,17002)
CONFIG={"representation":"group","objective":"pairwise","dropout":.2,"hidden":128,"l2":.01}
ARMS=("original","drop_one_half")
EPOCHS=14
SOURCES=("optimization_subset.py","optimization_trees.py","answer_group_experiment.py","answer_group_model.py",
         "advanced_neural.py","text_features.py","common.py","selection.py")


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def progress(stage,done,started,detail):
    dump(OUT/"progress.json",{"stage":stage,"completed_fits":done,"planned_fits":36,
        "detail":detail,"elapsed_seconds":time.perf_counter()-started,"updated_utc":datetime.now(timezone.utc).isoformat()})


def remove_slot(bank,embeddings,slot):
    changed=[]
    for row in bank:
        if len(row["candidates"])!=4:raise ValueError("Expected four candidates")
        candidates=list(row["candidates"])
        candidates[slot]={"index":slot,"text":"","code":None}
        changed.append(dict(row,candidates=candidates))
    values=np.asarray(embeddings,dtype=np.float32).copy();values[:,slot]=0.
    return changed,values


def input_view(bank,embeddings):
    weight=groups(bank);text=extract_features(bank)
    features=make_features(embeddings,text,weight,"group")
    return features,weight,weight.sum(axis=2)>0,text


def label_views(weights,valid,labels):
    # Called only with the current outer fitting questions, never held-out views.
    target=np.einsum("vngc,nc->vng",weights,labels)>0
    eligible=target.any(axis=2)&(valid&~target).any(axis=2)
    return target,eligible


def synthetic_checks():
    row={"id":"x","question":"A basket has 4 apples and 6 pears. What is the total?","candidates":[
        {"index":0,"text":"4 + 6 = 10. #### 10"},{"index":1,"text":"6 + 4 = 10. #### 10"},
        {"index":2,"text":"4 * 6 = 24. #### 24"},{"index":3,"text":"6 - 4 = 2. #### 2"}]}
    bank=[row];emb=np.arange(4*1024,dtype=np.float32).reshape(1,4,1024)/1000
    original_texts=[c["text"] for c in row["candidates"]];x,w,v,_=input_view(bank,emb)
    changed,ze=remove_slot(bank,emb,0);xx,ww,vv,_=input_view(changed,ze)
    assert parse_answer(changed[0]["candidates"][0]["text"]) is None and not ze[:,0].any()
    assert np.array_equal(emb[:,1:],ze[:,1:]) and [c["text"] for c in bank[0]["candidates"]]==original_texts
    assert np.isclose(x[0,0,-2],.5) and np.isclose(xx[0,0,-2],1/3)
    targets,eligible=label_views(np.stack([w,ww]),np.stack([v,vv]),np.array([[1,1,0,0]],dtype=bool))
    assert eligible.all() and targets[:,0].sum(axis=1).tolist()==[1,1]
    perm=[2,0,3,1];permuted=[dict(row,id="other",candidates=[row["candidates"][i] for i in perm])]
    xp,wp,_,_=input_view(permuted,emb[:,perm])
    candidate=np.einsum("ngc,ngd->ncd",w>0,x)
    pcandidate=np.einsum("ngc,ngd->ncd",wp>0,xp)
    assert np.allclose(candidate[:,perm],pcandidate,atol=2e-6,rtol=1e-6)
    empty=input_view([],np.empty((0,4,1024),np.float32))
    assert empty[0].shape==(0,4,1252)
    invalid=[dict(row,candidates=[{"text":""} for _ in range(4)])]
    ix,iw,iv,_=input_view(invalid,np.zeros((1,4,1024),np.float32))
    assert np.isfinite(ix).all() and not iv.any()
    return {"removed_slot_invalid_and_zero_embedding":True,"original_input_not_modified":True,
        "remaining_embeddings_unchanged":True,"support_recomputed_half_to_third":True,
        "training_view_targets_recomputed":True,"candidate_permutation_tolerant":True,
        "empty_and_all_invalid":True}


def protocol():
    return {"created_utc":datetime.now(timezone.utc).isoformat(),"scope":"G1 matched subset augmentation",
        "data_scope":["train"],"validation_access":False,"test_access":False,
        "fold_seeds":FOLD_SEEDS,"folds":3,"fold_assignment":"fold[default_rng(seed).permutation(N)]=arange(N)%3",
        "arms":ARMS,"model_seeds":MODEL_SEEDS,"configuration":CONFIG,"epochs":EPOCHS,
        "batch_questions":128,"optimizer":{"name":"Adam","learning_rate":.001,"beta1":.9,"beta2":.999,"epsilon":1e-8},
        "augmentation":{"probability":.5,"selection":"uniform among eligible drop-one views for each sampled fitting question; fallback original if none eligible",
            "eligibility":"retains at least one correct and one incorrect VALID candidate; reference labels consulted only for current fitting-side view filtering and losses",
            "representation":"four original slots retained; removed slot empty invalid text and zero embedding; remaining embeddings untouched",
            "recomputed":"original75 text features, answer groups, support, pooled embeddings/text/max/std; no stale original-support features",
            "randomness":"Separateaugmentation RNG seed=model_seed+20261026+1000*repeat+100*fold; initialization/shuffle/dropout RNG exactly matched across arms",
            "budget":"one view per original training question; same epochs, batches and optimizer updates; no added training rows"},
        "statistics":"one mean/scale per outer partition, fitted ONLY on valid original-view groups of all fitting questions; shared across arms and all model seeds",
        "training_questions":"original difficult outer-fitting questions; held-out and easy questions never enter optimizer batches",
        "heldout":"original four-candidate view only",
        "seed_ensemble":"Each member scaled on original fitting inputs; average three normalized member scores; one fitting-only ensemble scale per arm/partition",
        "planned_model_fits":36,"normalizer_fits":6,"full_training_refits":0,"threads":4,
        "selection":"highest summed repeat hard-correct, then lower mean normalized group CE, original on exact tie; never choose an individual seed",
        "checks":"support and deletion semantics, unchanged original data, synthetic candidate permutation/empty input, matched init and RNG/update counts, all held-out NumPy export replay",
        "limitations":"Original base75 has existing float32 centering; do not claim exact bitwise permutation invariance. Repeated training questions are correlated and OOF selection is exploratory."}


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/"protocol.json").exists():raise SystemExit("Registered protocol exists")
    dump(OUT/"synthetic_checks.json",synthetic_checks())
    dump(OUT/"protocol.json",protocol())
    dump(OUT/"source_hashes.json",{name:digest(ROOT/name) for name in SOURCES})
    dump(OUT/"input_hashes.json",{str(p.relative_to(ROOT)):digest(p) for p in (
        DATA/"candidates"/"gsm8k_train.jsonl",DATA/"candidates"/"gsm8k_train.npz",DATA/"gsm8k_train.jsonl")})
    progress("ready",0,time.perf_counter(),"Synthetic checks and protocol registered; awaiting queued --run")
    print(json.dumps({"stage":"ready","command":"python optimization_subset.py --run","planned_model_fits":36,"training_started":False}))


def fit_member(x,weights,targets,valid,eligible,seed,repeat,fold,augment):
    model=initialize(x.shape[-1],CONFIG["hidden"],seed)
    initial_hash=hashlib.sha256(b"".join(p.tobytes() for p in model)).hexdigest()
    first=[np.zeros_like(p) for p in model];second=[np.zeros_like(p) for p in model]
    rng=np.random.default_rng(seed+1)
    augmentation=np.random.default_rng(seed+20261026+1000*repeat+100*fold)
    updates=0;replaced=0;requested=0;slots=np.zeros(4,dtype=int);epoch_records=[]
    n=x.shape[1]
    for epoch in range(EPOCHS):
        order=rng.permutation(n);losses=[];epoch_replaced=0
        for start in range(0,n,128):
            ix=order[start:start+128];view=np.zeros(len(ix),dtype=np.int64)
            if augment:
                draw=augmentation.random(len(ix))<.5;pick=augmentation.random(len(ix));requested+=int(draw.sum())
                for j,q in enumerate(ix):
                    allowed=np.flatnonzero(eligible[1:,q])+1
                    if draw[j] and len(allowed):
                        view[j]=allowed[min(int(pick[j]*len(allowed)),len(allowed)-1)]
                        slots[view[j]-1]+=1;replaced+=1;epoch_replaced+=1
            loss,gradients=loss_gradient(x[view,ix],weights[view,ix],targets[view,ix],valid[view,ix],model,CONFIG,rng)
            losses.append(float(loss));updates+=1
            for k,g in enumerate(gradients):
                first[k]=.9*first[k]+.1*g;second[k]=.999*second[k]+.001*g*g
                model[k]-=.001*(first[k]/(1-.9**updates))/(np.sqrt(second[k]/(1-.999**updates))+1e-8)
        epoch_records.append({"epoch":epoch+1,"mean_batch_loss":float(np.mean(losses)),"augmented_questions":epoch_replaced})
    assert all(np.isfinite(p).all() for p in model)
    return model,{"updates":updates,"initial_parameters_sha256":initial_hash,
        "shared_rng_final_sha256":hashlib.sha256(json.dumps(rng.bit_generator.state,sort_keys=True).encode()).hexdigest(),
        "augmented_question_occurrences":replaced,"augmentation_requested_occurrences":requested,
        "removed_slot_occurrences":slots.tolist(),"total_question_occurrences":n*EPOCHS,"epochs":epoch_records}


def run():
    started=time.perf_counter()
    sources=json.loads((OUT/"source_hashes.json").read_text(encoding="utf-8"))
    inputs=json.loads((OUT/"input_hashes.json").read_text(encoding="utf-8"))
    if any(digest(ROOT/name)!=h for name,h in {**sources,**inputs}.items()):raise SystemExit("Registered source/input changed")
    if (OUT/"summary.json").exists():print("Already complete; no new fits");return
    if (OUT/"run_started.json").exists():raise SystemExit("Incomplete attempt exists; preserve it before recovery")
    dump(OUT/"run_started.json",{"utc":datetime.now(timezone.utc).isoformat(),"protocol_sha256":digest(OUT/"protocol.json")})
    progress("features",0,started,"Precomputing original and4reference-free drop-one views")
    bank,answers,labels,valid,hard=load_training();input_bank,arrays=load_inputs("train");emb=arrays["embeddings"]
    assert [r["id"] for r in bank]==[r["id"] for r in input_bank]
    n=len(bank);tick=time.perf_counter()
    original,original_weights,original_valid,original_text=input_view(bank,emb)
    x=np.empty((5,n,4,1252),dtype=np.float32);weight=np.empty((5,n,4,4),dtype=np.float32);group_valid=np.empty((5,n,4),dtype=bool)
    x[0]=original;weight[0]=original_weights;group_valid[0]=original_valid
    for slot in range(4):
        dropped,changed=remove_slot(bank,emb,slot)
        x[slot+1],weight[slot+1],group_valid[slot+1],_=input_view(dropped,changed)
        progress("features",0,started,f"Prepared drop-slot{slot} input features; no reference labels used")
    feature_seconds=time.perf_counter()-tick
    np.savez_compressed(OUT/"training_view_features.npz",ids=np.array([r["id"] for r in bank]),group_features=x,group_weights=weight,group_valid=group_valid)
    folds=np.empty((2,n),dtype=np.int8)
    for r,seed in enumerate(FOLD_SEEDS):folds[r,np.random.default_rng(seed).permutation(n)]=np.arange(n)%3
    raw=np.full((2,n,4,2),np.nan,dtype=np.float64);normal=raw.copy()
    member_raw=np.full((2,n,4,2,3),np.nan,dtype=np.float64);member_normal=member_raw.copy()
    member_scales=np.zeros((2,3,2,3));ensemble_scales=np.zeros((2,3,2));records=[];normalizers=[];done=0
    with threadpool_limits(limits=4):
        for repeat in range(2):
            for fold in range(3):
                fitting=np.where(folds[repeat]!=fold)[0];held=np.where(folds[repeat]==fold)[0];active=fitting[hard[fitting]]
                mean,std=fit_statistics(x[0,fitting],group_valid[0,fitting])
                normalizers.append({"repeat":repeat,"fold":fold,"fit_question_sha256":array_digest(fitting.astype(np.int64)),
                    "fit_valid_groups":int(group_valid[0,fitting].sum()),"mean_sha256":array_digest(mean),"scale_sha256":array_digest(std)})
                active_x=normalized(x[:,active],mean,std)
                active_weights=weight[:,active];active_valid=group_valid[:,active]
                targets,eligible=label_views(active_weights,active_valid,labels[active])
                assert eligible[0].all()
                original_fit=normalized(x[0,fitting],mean,std);original_held=normalized(x[0,held],mean,std)
                original_eligible_counts=eligible[1:].sum(axis=0)
                matched={}
                for arm,arm_name in enumerate(ARMS):
                    fit_sum=np.zeros((len(fitting),4),dtype=np.float64);held_sum=np.zeros((len(held),4),dtype=np.float64)
                    for member,seed in enumerate(MODEL_SEEDS):
                        progress("fitting",done,started,f"{arm_name} repeat{repeat+1} fold{fold+1} seed{seed}")
                        tick=time.perf_counter()
                        model,record=fit_member(active_x,active_weights,targets,active_valid,eligible,seed,repeat,fold,arm==1)
                        seconds=time.perf_counter()-tick
                        fit_z=candidate_scores(scores(original_fit,model),weight[0,fitting],"group")
                        held_z=candidate_scores(scores(original_held,model),weight[0,held],"group")
                        scale=scale_scores(fit_z,valid[fitting]);member_scales[repeat,fold,arm,member]=scale
                        fit_sum+=fit_z/scale;held_sum+=held_z/scale
                        member_raw[repeat,held,:,arm,member]=held_z;member_normal[repeat,held,:,arm,member]=held_z/scale
                        path=OUT/f"repeat{repeat}_fold{fold}_{arm_name}_seed{seed}.npz";save_asset(path,model,mean,std)
                        with np.load(path,allow_pickle=False) as archive:asset={k:archive[k] for k in archive.files}
                        replay=group_model_scores([bank[i] for i in held],emb[held],asset,"group",original_text[held])
                        error=float(np.max(np.abs(replay-held_z)));assert error<1e-5
                        fingerprint=(record["updates"],record["initial_parameters_sha256"],record["shared_rng_final_sha256"])
                        if arm==0:matched[seed]=fingerprint
                        else:assert fingerprint==matched[seed]
                        metric,_,_=evaluate([bank[i] for i in held],[answers[i] for i in held],labels[held],hard[held],held_z/scale)
                        record.update(repeat=repeat,fold=fold,arm=arm_name,seed=seed,training_seconds=seconds,fit_questions=len(active),
                            scale=scale,fit_question_sha256=array_digest(active.astype(np.int64)),held_question_sha256=array_digest(held.astype(np.int64)),
                            normalizer_mean_sha256=array_digest(mean),normalizer_scale_sha256=array_digest(std),
                            eligible_drop_view_count_histogram=np.bincount(original_eligible_counts,minlength=5).tolist(),
                            asset=path.name,asset_sha256=digest(path),uncompressed_asset_bytes=sum(a.nbytes for a in asset.values()),
                            numpy_export_max_error=error,metrics=metric)
                        records.append(record);done+=1;dump(OUT/"fit_records.json",records)
                        print(json.dumps({"fit":done,"of":36,"repeat":repeat,"fold":fold,"arm":arm_name,"seed":seed,"hard_correct":metric["hard_correct"]}),flush=True)
                    fit_sum/=3;held_sum/=3;scale=scale_scores(fit_sum,valid[fitting]);ensemble_scales[repeat,fold,arm]=scale
                    raw[repeat,held,:,arm]=held_sum;normal[repeat,held,:,arm]=held_sum/scale
                progress("fitting",done,started,f"Completed matched partition repeat{repeat+1} fold{fold+1}")
    assert np.isfinite(raw).all() and np.isfinite(member_raw).all()
    choices=np.zeros((2,n,2),dtype=np.int32);correct=np.zeros_like(choices,dtype=bool);results=[]
    for arm,name in enumerate(ARMS):
        repeats=[];members=[]
        for r in range(2):
            metric,choices[r,:,arm],correct[r,:,arm]=evaluate(bank,answers,labels,hard,normal[r,:,:,arm]);repeats.append(metric)
            members.append([evaluate(bank,answers,labels,hard,member_normal[r,:,:,arm,m])[0] for m in range(3)])
        results.append({"name":name,"repeats":repeats,"member_metrics_by_repeat":members,
            "summed_hard_correct":sum(m["hard_correct"] for m in repeats),"mean_group_ce":float(np.mean([m["group_ce_hard"] for m in repeats]))})
    for arm,result in enumerate(results):
        result["paired_vs_original"]=[paired(correct[r,:,0],correct[r,:,arm],hard) for r in range(2)]
        result["per_fold_vs_original"]=[{"repeat":r,"fold":f,**paired(correct[r,:,0],correct[r,:,arm],hard&(folds[r]==f))} for r in range(2) for f in range(3)]
    chosen=max(range(2),key=lambda a:(results[a]["summed_hard_correct"],-results[a]["mean_group_ce"],-a))
    np.savez_compressed(OUT/"oof_logits.npz",ids=np.array([r["id"] for r in bank]),raw=raw,normalized=normal,member_raw=member_raw,member_normalized=member_normal,
        labels=labels,valid=valid,hard=hard,folds=folds,scales=ensemble_scales,member_scales=member_scales,choices=choices,correct=correct,
        configuration_names=np.array(ARMS),fold_seeds=np.array(FOLD_SEEDS),model_seeds=np.array(MODEL_SEEDS))
    dump(OUT/"oof_schema.json",{"raw_normalized_shape":[2,n,4,2],"axes":["repeat","question","candidate","arm"],
        "member_axes":["repeat","question","candidate","arm","seed"],"raw_definition":"mean of three member-scale-normalized scores","label_scope":"train only"})
    dump(OUT/"normalizer_records.json",normalizers)
    summary={"configurations":results,"training_selected_arm":ARMS[chosen],"selected_config_index":chosen,
        "model_fit_calls":done,"normalizer_fits":6,"full_refit_calls":0,"validation_evaluations":0,"test_access":False,
        "training_questions":n,"hard_questions":int(hard.sum()),"feature_seconds":feature_seconds,
        "fit_seconds":sum(r["training_seconds"] for r in records),"elapsed_seconds":time.perf_counter()-started,
        "matched_rng_and_updates_all_partitions":True,"maximum_numpy_export_error":max(r["numpy_export_max_error"] for r in records),
        "sources_unchanged":all(digest(ROOT/name)==h for name,h in sources.items()),"source_hashes":sources,"input_hashes":inputs,
        "threads":4,"python":platform.python_version(),"numpy":np.__version__,"interpretation":"Repeated OOF estimates are correlated, exploratory and selection-conditional; no validation/test benefit claimed"}
    assert summary["sources_unchanged"]
    dump(OUT/"results.json",results);dump(OUT/"summary.json",summary)
    progress("complete",done,started,"Matched subset study finished; no validation/test/fullrefit")
    print(json.dumps({"summary":str(OUT/"summary.json"),"winner":ARMS[chosen]},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);g=parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--prepare",action="store_true");g.add_argument("--run",action="store_true")
    args=parser.parse_args()
    prepare() if args.prepare else run()


if __name__=="__main__":main()
