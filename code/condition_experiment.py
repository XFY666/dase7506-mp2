"""Six fixed HGBT configurations selected by three-fold TRAIN OOF only.

There is intentionally no validation/test loading code in this driver. The
selected full-training NumPy artifact is saved pending separate authorization
for any later evaluation. All six predictions and fitting-side scales remain.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_key]="4"

import argparse
import copy
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

from common import parse_answer
from condition_features import extract_features, FEATURE_NAMES, FEATURE_MEANINGS, FEATURE_GROUPS
from condition_model import DESIGNS, feature_matrix, score_candidates
from experiment_utils import development_data, dump, metrics
from quantitative_model import answer_valid, stable_base_features, stable_center
from selection import select
from text_features import FEATURE_NAMES as BASE_NAMES
from tree_model import export_trees, tree_scores

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"runs"/"condition_experiments"
SEED=20261015
CONFIGS=[{"name":f"{design}__L{leaves}","design":design,"leaves":leaves}
         for design in DESIGNS for leaves in (15,31)]


def digest(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def parameters(leaves, rounds=300):
    return {"max_leaf_nodes":leaves,"max_iter":rounds,"learning_rate":.05,"l2_regularization":10.,
            "min_samples_leaf":30,"early_stopping":False,"random_state":SEED}


def protocol():
    return {"status":"Predeclared six-configuration training-only study",
        "hypothesis":"Local correspondence of quantities, objects, temporal/comparison qualifiers and requested target improves selection beyond global text/number cues.",
        "independent_unit":"Question; all four candidates stay together",
        "data_scope":["train"],"new_validation_evaluations":0,"test_access":False,
        "folds":3,"seed":SEED,"fold_mapping":"order=default_rng(20261015).permutation(N); fold[order]=arange(N)%3",
        "configurations":CONFIGS,"hgb":parameters(15),"vary_only":"design and max_leaf_nodes 15/31",
        "planned_model_fits":{"cost_probe":1,"outer_folds":18,"selected_full_refit":1,"total":20},
        "planned_pls_fits":"3 (one per outer fold, shared across two tree leaf counts), plus 1 if selected full design uses PLS",
        "round_selection":"No early stopping or stage search; exactly 300 rounds",
        "fit_rows":"Valid candidate rows from fitting-side questions containing both correct and incorrect valid answers; each eligible question total sample weight 4",
        "pls":"PLSRegression(n_components=8,scale=False,max_iter=500,tol=1e-6), valid-centered embeddings/labels on eligible fitting questions only. Export float32 means/rotations and use these for both tree fitting and deployment. Both raw and valid-centered projected coordinates are included.",
        "feature_boundary":"No learned lexical vocabulary, references, IDs or candidate index features; fixed small linguistic dictionaries only. Object labels are lexical proxies; opposite terms are fallible warnings, not entailment proof.",
        "numeric_policy":"Local ratios and relative features float64; relative means sort four candidate values before summing. Existing base75 uses the precision-repaired stable_base_features wrapper. Projected centering uses float64 accumulation and subtraction followed by float32 cast.",
        "aggregation":"mean","consensus":0.,
        "score_scale":"Fitting-side std of candidate raw scores after all-four-question mean centering, measured at fitting valid candidate entries; compute float64 and include unanimous fitting questions",
        "selection_order":["highest pooled training OOF hard_correct","lowest normalized question-balanced pair logistic loss","fewer feature dimensions","fewer leaves","earlier predefined config"],
        "cost_probe":"First 256 training questions feature extraction and synthetic semantic controls; fit one leaf15 condition HGBT for 40 rounds on first 256 training questions only. Probe is timing only, not a grid candidate and not used for model selection.",
        "adaptation_policy":"No feature edits or config expansion after OOF scores; implementation repairs are explicitly logged. Only one selected full model is exported; validation is not loaded in this program.",
        "threads":4,"asset_limit_bytes":16*1024*1024,
        "coverage_definition":"For each feature, training candidate nonzero fraction, within-question varying fraction, range and mean; descriptive, not used to remove or select features",
        "scope_caveat":"This is exploratory research informed by earlier train/validation studies. The new OOF winner is selected on the same OOF questions and is therefore a selection-conditional estimate."}


def fit_pls(embeddings, labels, valid, eligible, indices):
    ids=np.asarray(indices)[eligible[indices]]
    x=stable_center(embeddings[ids],valid[ids])[valid[ids]]
    y=stable_center(labels[ids],valid[ids])[valid[ids]]
    tick=time.perf_counter()
    model=PLSRegression(n_components=8,scale=False,max_iter=500,tol=1e-6).fit(x,y)
    asset={"pls_mean":model._x_mean.astype(np.float32),"pls_rotations":model.x_rotations_.astype(np.float32)}
    return asset,{"fit_questions":len(ids),"fit_rows":len(y),"fit_indices_sha256":digest(ids.astype(np.int64)),
        "components":8,"seconds":time.perf_counter()-tick,"iterations":[int(x) for x in model.n_iter_]}


def fit_tree(x, labels, valid, eligible, indices, leaves, rounds=300):
    ids=np.asarray(indices)[eligible[indices]]
    mask=valid[ids]
    weights=np.broadcast_to(4./np.maximum(mask.sum(axis=1,keepdims=True),1),mask.shape)[mask]
    return HistGradientBoostingClassifier(**parameters(leaves,rounds)).fit(x[ids][mask],labels[ids][mask],sample_weight=weights)


def scale_scores(z, valid, ids):
    values=np.asarray(z,dtype=np.float64)[ids]
    centered=values-np.sort(values,axis=1).mean(axis=1,keepdims=True)
    return float(max(np.std(centered[valid[ids]]),1e-6))


def pair_loss(z, labels, valid):
    losses=[]
    for values,y,v in zip(z,labels,valid):
        good=values[(y>0)&v];bad=values[(y==0)&v]
        if len(good) and len(bad):losses.append(float(np.logaddexp(0.,bad[None,:]-good[:,None]).mean()))
    return float(np.mean(losses))


def names(design):
    return list(BASE_NAMES)+(list(FEATURE_NAMES) if design!="base75" else [])+(
        [f"pls_raw_{i}" for i in range(8)]+[f"pls_centered_{i}" for i in range(8)] if design=="condition_pls8" else [])


def asset_for(model, transform, config):
    result=export_trees(model)
    if transform:result.update(transform)
    result.update(condition_model_version=np.array(1,dtype=np.int32),
        design_code=np.array(DESIGNS.index(config["design"]),dtype=np.int32),
        feature_count=np.array(len(names(config["design"])),dtype=np.int32))
    return result


def self_check():
    cases=[
      ("Maya has 6 red beads and Leo has 9 blue beads. How many blue beads does Leo have?",[
        "Maya has 6 red beads. Leo has 9 blue beads. Thus Leo has 9 blue beads. #### 9",
        "Maya has 9 red beads. Leo has 6 blue beads. Thus Leo has 6 blue beads. #### 6",
        "6 + 9 = 15. The answer is 15. #### 15", "Leo has 9 blue beads. #### 9"]),
      ("Before lunch Ada had 12 apples. After lunch Ada had 5 apples. How many apples did Ada lose?",[
        "Before lunch Ada had 12 apples. After lunch Ada had 5 apples. Ada lost 12 - 5 = 7 apples. #### 7",
        "After lunch Ada had 12 apples. Before lunch Ada had 5 apples. Ada gained 12 - 5 = 7 apples. #### 7",
        "12 - 5 = 7. #### 7", "Ada lost 7 apples. #### 7"]),
      ("Each box holds 4 pencils. Kim buys 3 boxes. How many pencils does Kim buy?",[
        "Each box holds 4 pencils. Kim buys 3 boxes. Thus 3 * 4 = 12 pencils. #### 12",
        "Each box holds 4 pencils. Kim buys 3 boxes. Thus 3 * 4 = 12 boxes. #### 12",
        "3 * 4 = 12. #### 12", "Kim buys 12 pencils. #### 12"])]
    bank=[{"id":str(i),"question":q,"candidates":[{"index":j,"text":t} for j,t in enumerate(texts)]}
          for i,(q,texts) in enumerate(cases)]
    x=extract_features(bank);idx={n:i for i,n in enumerate(FEATURE_NAMES)}
    assert x.shape==(3,4,len(FEATURE_NAMES)) and np.isfinite(x).all()
    assert extract_features([]).shape==(0,4,len(FEATURE_NAMES))
    assert x[0,0,idx["object_binding_recall_mean"]]>x[0,1,idx["object_binding_recall_mean"]]
    assert x[1,0,idx["anchor_time_preservation"]]>x[1,1,idx["anchor_time_preservation"]]
    assert x[2,0,idx["target_answer_object_recall"]]>x[2,1,idx["target_answer_object_recall"]]
    perm=np.array([2,0,3,1]);changed=copy.deepcopy(bank)
    for row in changed:
        row["id"]="irrelevant";row["candidates"]=[row["candidates"][i] for i in perm]
        for c in row["candidates"]:c["index"]=-3
    assert np.array_equal(extract_features(changed),x[:,perm])
    return {"passed":True,"feature_count":len(FEATURE_NAMES),"checks":["empty shape","finite bounded values","object swap control","temporal reversal control","target-object control","candidate permutation and ID/index mutation"],
        "synthetic_cases":[{"question":row["question"],"candidate_texts":[c["text"] for c in row["candidates"]],
          "features":{name:x[i,:,idx[name]].tolist() for name in ("object_binding_recall_mean","anchor_time_preservation","anchor_time_reversal_fraction","target_answer_object_recall")}}
          for i,row in enumerate(bank)]}


def feature_coverage(x):
    return [{"name":name,"group":next(g for g,ii in FEATURE_GROUPS.items() if j in ii),
        "meaning":FEATURE_MEANINGS[name],"nonzero_fraction":float((x[...,j]!=0).mean()),
        "within_question_varying_fraction":float((np.ptp(x[...,j],axis=1)>1e-7).mean()),
        "minimum":float(x[...,j].min()),"maximum":float(x[...,j].max()),"mean":float(x[...,j].astype(np.float64).mean())}
        for j,name in enumerate(FEATURE_NAMES)]


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--self-check",action="store_true");parser.add_argument("--probe-only",action="store_true")
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    checks=self_check()
    if args.self_check:print(json.dumps(checks,indent=2));return
    if (OUT/"summary.json").exists():raise FileExistsError("Completed finite study exists")
    dump(OUT/"protocol.json",protocol());dump(OUT/"self_check.json",checks)
    dump(OUT/"feature_definitions.json",{"feature_count":len(FEATURE_NAMES),"groups":FEATURE_GROUPS,"definitions":FEATURE_MEANINGS})
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
        ("condition_features.py","condition_model.py","condition_experiment.py","quantitative_model.py","text_features.py","tree_model.py")}
    dump(OUT/"source_hashes.json",hashes)
    dump(OUT/"environment.json",{"python":platform.python_version(),"numpy":np.__version__,"sklearn":sklearn.__version__,"threads":4,
        "command":"python condition_experiment.py","validation_loaded":False})
    started=time.perf_counter();bank,embeddings,labels,hard=development_data("train")
    valid=answer_valid(bank);eligible=((labels*valid).sum(1)>0)&(((1-labels)*valid).sum(1)>0)
    ids=np.array([r["id"] for r in bank]);n=len(bank)
    order=np.random.default_rng(SEED).permutation(n);fold=np.empty(n,dtype=np.int32);fold[order]=np.arange(n)%3
    np.savez(OUT/"fold_assignment.npz",ids=ids,fold=fold,labels=labels,valid=valid,hard=hard,eligible=eligible)
    with threadpool_limits(limits=4):
        if not (OUT/"cost_probe.json").exists():
            probe_ids=np.arange(min(256,n));pb=[bank[i] for i in probe_ids]
            tick=time.perf_counter();pc=extract_features(pb);condition_seconds=time.perf_counter()-tick
            tick=time.perf_counter();pbase=stable_base_features(pb);base_seconds=time.perf_counter()-tick
            px=np.concatenate((pbase,pc),axis=-1)
            tick=time.perf_counter();probe_model=fit_tree(px,labels[probe_ids],valid[probe_ids],eligible[probe_ids],probe_ids,15,40)
            fit_seconds=time.perf_counter()-tick
            dump(OUT/"cost_probe.json",{"questions":len(pb),"condition_seconds":condition_seconds,"base_seconds":base_seconds,
                "hgb40_fit_seconds":fit_seconds,"feature_count":pc.shape[-1],"projected_full_bank_feature_seconds":(condition_seconds+base_seconds)*n/len(pb),
                "model_fit_calls":1,"labels_scope":"First 256 training questions only; no performance-based choice","source_sha256":hashes})
            dump(OUT/"probe_feature_coverage.json",feature_coverage(pc))
            np.savez(OUT/"probe_model.npz",**export_trees(probe_model))
            print(json.dumps({"probe":json.loads((OUT/"cost_probe.json").read_text())}),flush=True)
        if args.probe_only:return
        tick=time.perf_counter();base=stable_base_features(bank);base_seconds=time.perf_counter()-tick
        tick=time.perf_counter();conditions=extract_features(bank);condition_seconds=time.perf_counter()-tick
        dump(OUT/"feature_coverage.json",feature_coverage(conditions))
        np.savez(OUT/"training_feature_cache.npz",ids=ids,base=base,condition=conditions)
        raw=np.empty((n,4,len(CONFIGS)),dtype=np.float64);normalized=np.empty_like(raw)
        scales=np.empty((3,len(CONFIGS)));records=[];transforms=[]
        for f in range(3):
            fit=np.flatnonzero(fold!=f);held=np.flatnonzero(fold==f)
            transform,info=fit_pls(embeddings,labels,valid,eligible,fit);transforms.append({"fold":f,**info})
            for ci,config in enumerate(CONFIGS):
                tr=transform if config["design"]=="condition_pls8" else None
                x=feature_matrix(base,conditions,embeddings,valid,config["design"],tr)
                tick=time.perf_counter();model=fit_tree(x,labels,valid,eligible,fit,config["leaves"]);fit_seconds=time.perf_counter()-tick
                z=model.decision_function(x.reshape(-1,x.shape[-1])).reshape(-1,4)
                scale=scale_scores(z,valid,fit);scales[f,ci]=scale
                raw[held,:,ci]=z[held];normalized[held,:,ci]=z[held]/scale
                asset=asset_for(model,tr,config);asset["score_scale"]=np.array(scale)
                np.savez(OUT/f"fold{f}_{config['name']}.npz",**asset)
                exported=tree_scores(x[held].reshape(-1,x.shape[-1]),asset).reshape(-1,4)
                error=float(np.max(np.abs(exported-z[held])));assert error<1e-8
                row={"fold":f,"config":config,"features":x.shape[-1],"fit_questions":len(fit),"heldout_questions":len(held),
                    "eligible_fit_questions":int(eligible[fit].sum()),"valid_fit_candidate_rows":int(valid[fit][eligible[fit]].sum()),
                    "fit_indices_sha256":digest(fit.astype(np.int64)),"held_indices_sha256":digest(held.astype(np.int64)),
                    "fit_seconds":fit_seconds,"scale":scale,"export_error":error,
                    "heldout_metrics":metrics(select([bank[i] for i in held],z[held],"mean"),labels[held],hard[held])}
                records.append(row);dump(OUT/"fold_records.json",records)
                print(json.dumps(row),flush=True)
            dump(OUT/"transform_records.json",transforms)
        rows=[]
        for ci,config in enumerate(CONFIGS):
            rows.append({"config_index":ci,**config,"features":len(names(config["design"])),
                "oof":metrics(select(bank,raw[:,:,ci],"mean"),labels,hard),
                "normalized_pair_loss":pair_loss(normalized[:,:,ci],labels,valid),"fold_scales":scales[:,ci].tolist()})
        chosen=min(rows,key=lambda r:(-r["oof"]["hard_correct"],r["normalized_pair_loss"],r["features"],r["leaves"],r["config_index"]))
        ci=chosen["config_index"];config=CONFIGS[ci]
        dump(OUT/"oof_results.json",rows);dump(OUT/"selected_training_recipe.json",chosen)
        common={"ids":ids,"labels":labels,"valid":valid,"hard":hard,"eligible":eligible,"fold":fold}
        np.savez(OUT/"all_oof_logits.npz",**common,raw=raw,normalized=normalized,fold_scales=scales,config_names=np.array([c["name"] for c in CONFIGS]))
        np.savez(OUT/"selected_oof_logits.npz",**common,raw=raw[:,:,ci],normalized=normalized[:,:,ci],fold_scales=scales[:,ci])
        fit=np.arange(n);transform=None;info=None
        if config["design"]=="condition_pls8":transform,info=fit_pls(embeddings,labels,valid,eligible,fit)
        x=feature_matrix(base,conditions,embeddings,valid,config["design"],transform)
        tick=time.perf_counter();model=fit_tree(x,labels,valid,eligible,fit,config["leaves"]);full_seconds=time.perf_counter()-tick
        asset=asset_for(model,transform,config)
        z=model.decision_function(x.reshape(-1,x.shape[-1])).reshape(-1,4)
        scale=scale_scores(z,valid,fit);asset["score_scale"]=np.array(scale)
        path=OUT/"selected_model.npz";np.savez(path,**asset)
        export=tree_scores(x.reshape(-1,x.shape[-1]),asset).reshape(-1,4)
        error=float(np.max(np.abs(export-z)));assert error<1e-8
        # Deployment checks are on training inputs only, after selection.
        perm=np.array([2,0,3,1]);alt=copy.deepcopy(bank)
        for r in alt:
            r["id"]="unused";r["candidates"]=[r["candidates"][i] for i in perm]
            for c in r["candidates"]:c["index"]=-99
        tick=time.perf_counter();zp=score_candidates(alt,embeddings[:,perm],asset);deployment_seconds=time.perf_counter()-tick
        pred=select(bank,z,"mean");pp=select(alt,zp,"mean")
        answer_changes=sum(parse_answer(bank[i]["candidates"][j]["text"])!=parse_answer(alt[i]["candidates"][pp[i]]["text"]) for i,j in enumerate(pred))
        check={"questions":n,"permutation":perm.tolist(),"max_score_error":float(np.max(np.abs(zp-z[:,perm]))),
            "questions_with_nonidentical_score":int(np.any(zp!=z[:,perm],axis=1).sum()),"parsed_answer_changes":answer_changes,
            "deployment_seconds":deployment_seconds,"empty_shape":list(score_candidates([],np.empty((0,4,1024),np.float32),asset).shape),
            "scope":"One full-training-bank permutation; no universal invariance claim"}
        assert answer_changes==0,check
        dump(OUT/"deployment_check.json",check)
        np.savez(OUT/"training_deployment_scores.npz",ids=ids,raw=z,permuted=zp,permutation=perm)
        selected_config={"kind":"condition_tree","module":"condition_model.py","score_function":"score_candidates(bank, embeddings, asset)",
            "asset":"selected_model.npz","design":config["design"],"parameters":parameters(config["leaves"]),"features":names(config["design"]),
            "aggregation":"mean","score_scale":scale,"normalization":"Divide raw scores by score_scale when fusing",
            "selection":"Six fixed recipes by training OOF only; validation pending separate authorization"}
        dump(OUT/"selected_config.json",selected_config)
        summary={"status":"TRAIN-only study complete; selected full model saved; validation not accessed",
            "selected":chosen,"oof_results":rows,"feature_counts":{"base":len(BASE_NAMES),"condition":len(FEATURE_NAMES)},
            "model_fit_calls":20,"transform_fit_calls":3+int(transform is not None),"new_validation_evaluations":0,
            "feature_seconds":{"base":base_seconds,"condition":condition_seconds},"full_fit_seconds":full_seconds,"full_transform":info,
            "score_scale":scale,"asset_bytes":path.stat().st_size,"asset_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "export_max_abs_error":error,"deployment_check":check,"total_seconds":time.perf_counter()-started,"source_sha256":hashes,
            "feature_coverage_path":"feature_coverage.json","inference_dependencies":["Python standard library","NumPy"],
            "interpretation":"OOF selection is exploratory; any claimed generalization gain requires the separately authorized held-out check. No new setting may be selected from that check."}
        dump(OUT/"summary.json",summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":main()

