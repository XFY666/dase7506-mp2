"""Registered training-only aggregation of three inner stopping decisions."""
from __future__ import annotations
import os
import sys
sys.dont_write_bytecode = True
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "4"
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import zipfile
import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
import advanced_neural as advanced
import nonlinear_experiment as nonlinear
import optimization_baselines as baseline
from common import DATA
from ensemble import head_scores
from experiment_utils import dump, metrics
from finalize_validation import score_scale
from selection import select
from train_selected import fit_head

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs/optimization_stopping"
BASE = ROOT / "runs/optimization_baselines"
POLICIES = ("first", "median3", "rounded_mean3")
METHODS = ("D_fixed5", "D_first", "D_median3", "D_rounded_mean3")
FOLD_SEEDS = (20261027, 20261028)
STARTED = None


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(x):
    x = np.ascontiguousarray(x)
    return hashlib.sha256(str(x.dtype).encode() + str(x.shape).encode() + x.tobytes()).hexdigest()


def epochs_for_policy(epochs):
    assert len(epochs) == 3 and min(epochs) >= 1 and max(epochs) <= 100
    return {"first": int(epochs[0]), "median3": int(np.median(epochs)),
            "rounded_mean3": max(1, int(np.floor(np.mean(epochs) + .5)))}


def source_closure():
    paths = {Path(__file__).resolve()}
    for module in list(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if name and Path(name).resolve().parent == ROOT and Path(name).suffix == ".py":
            paths.add(Path(name).resolve())
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths)}


def progress(stage, count, detail, resolved_slots=0):
    temporary = OUT / "progress.json.tmp"
    dump(temporary, dict(stage=stage, completed_fits=count, planned_fits=72,
        maximum_planned_fits=72, resolved_fit_slots=resolved_slots, detail=detail,
        elapsed_seconds=time.perf_counter()-STARTED if STARTED is not None else 0.,
        updated_utc=datetime.now(timezone.utc).isoformat()))
    os.replace(temporary, OUT / "progress.json")


def register():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "protocol.json"
    sources = source_closure()
    if path.exists():
        protocol = json.loads(path.read_text(encoding="utf-8"))
        assert protocol["source_sha256"] == sources, "Registered source changed"
        return protocol
    bp = json.loads((BASE / "protocol.json").read_text(encoding="utf-8"))
    assert bp["fold_seeds"] == list(FOLD_SEEDS)
    protocol = dict(purpose="A3 matched first/median/rounded-mean stopping aggregation for the two deployed D neural heads",
        methods=METHODS, policies=POLICIES, outer_fold_seeds=FOLD_SEEDS, outer_folds=3,
        inner_splits=3, split_seed="outer_fold_seed + 1000*outer_fold + [1,2,3]",
        split_rule="Permute all sorted outer-fitting question indices; first floor(.85*N) fit, remainder stop; then restrict supervision to hard questions",
        neural_recipes=bp["heads_D"][:2], fixed_initialization_seed=7506,
        stopping=dict(max_epochs=100, pair_patience=8, list_patience=10, min_improvement=1e-6,
            loss="Original head heldout objective loss; loss excludes regularization, as in original helpers",
            normalizer="All inner-fitting questions only, including easy; no stopping/outer-held question statistics",
            aggregation="first split best epoch; integer median of3; max(1,floor(mean(best_epochs)+.5))",
            max_epoch_difference="Pair original search used max80; this registered bounded study uses requested max100, retaining original patience8"),
        refit="Fresh fixed-seed fit on all outer-fitting questions at each policy epoch; original pair/list supervision and transforms",
        refit_cache="Within one outer partition and one head only, identical epoch requests reuse exactly one fresh refit; no cross-head/cross-fold or A2 neural reuse",
        cost=dict(inner_trajectory_fits=36, maximum_outer_refits=36, maximum_total_fits=72,
            outer_policy_readouts=36, cached_refits_counted_as_new_fits=False, text_model_new_fits=0),
        fusion="Reuse A2 outer text raw scores and its fitting-only scale; original float32 equal thirds D fusion, mean answer-group aggregation",
        inference="Original NumPy head assets plus score_scale; policy manifests identify logical asset paths and fixed external A2 text asset",
        evidence="All method comparisons descriptive training OOF; repeated questions not independent, no validation/test or final promotion",
        source_sha256=sources,
        baseline_evidence_sha256={name:digest(BASE/name) for name in ("protocol.json","oof_logits.npz","fit_records.json","verification.json")})
    dump(path, protocol)
    progress("queued", 0, "A3三次内层早停汇总已登记；最多72次拟合，等待队列启动")
    return protocol


def audit_io():
    accessed = set()
    def audit(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path.name.startswith(("gsm8k_validation", "gsm8k_test")):
            raise RuntimeError("Forbidden split access")
        if DATA.resolve() in path.parents:
            accessed.add(path.relative_to(DATA.resolve()).as_posix())
        mode, flags = args[1], args[2] or 0
        write = isinstance(mode, str) and any(c in mode for c in "wax+") or bool(flags & (os.O_WRONLY | os.O_RDWR))
        if write and path != OUT.resolve() and OUT.resolve() not in path.parents:
            raise RuntimeError(f"Write outside experiment: {path}")
    sys.addaudithook(audit)
    return accessed


def stop_fit(head, embeddings, labels, valid, pair_weights, hard, fitting, stopping):
    active = fitting[hard[fitting]]
    check = stopping[hard[stopping]]
    assert len(active) and len(check)
    recipe = head["training"]
    if head["kind"] == "nonlinear":
        nonlinear.SEED = recipe["seed"]
        rows = embeddings[fitting].reshape(-1, 1024)
        mean = rows.mean(0, dtype=np.float64).astype(np.float32)
        scale = np.maximum(rows.std(0, dtype=np.float64), 1e-8).astype(np.float32)
        model, curve, selected = nonlinear.fit((embeddings[active]-mean)/scale, pair_weights[active],
            recipe["hidden"], recipe["l2"], 100,
            validation=((embeddings[check]-mean)/scale, pair_weights[check]), patience=8)
        w1 = model[0]/scale[:,None]
        asset = dict(w1=w1, b1=model[1]-mean@w1, w2=model[2], train_mean=mean, train_scale=scale)
    else:
        assert head["kind"] == "residual"
        stats = advanced.fit_transform(embeddings[fitting], valid[fitting])
        x = advanced.transform(embeddings[active], valid[active], stats, False)
        h = advanced.transform(embeddings[check], valid[check], stats, False)
        model, curve, selected = advanced.fit(x, labels[active], valid[active], objective=recipe["objective"],
            l2=recipe["l2"], hidden=recipe["hidden"], seed=recipe["seed"], epochs=100,
            heldout=(h, labels[check], valid[check]))
        asset = advanced.export_model(model, stats, False)
    assert 1 <= selected <= len(curve) <= 100
    return asset, curve, selected


def run():
    global STARTED
    protocol = register()
    if (OUT/"run_started.json").exists():
        raise FileExistsError("A run already exists; preserve its evidence")
    STARTED = time.perf_counter()
    accessed = audit_io()
    dump(OUT/"run_started.json", dict(utc=datetime.now(timezone.utc).isoformat()))
    for name, expected in protocol["baseline_evidence_sha256"].items():
        assert digest(BASE/name) == expected
    bank, embeddings, labels, valid, hard = advanced.development("train")
    pairs = labels[:,:,None] * ((~labels)&valid)[:,None,:]
    count = pairs.sum((1,2))
    pair_weights = (pairs/np.maximum(count[:,None,None],1)).astype(np.float32)
    assert np.array_equal(count>0, hard)
    with np.load(BASE/"oof_logits.npz", allow_pickle=False) as f:
        outer = {k:f[k] for k in ("ids","raw","normalized","labels","valid","hard","folds","mixture_scores","score_scales")}
    for key, value in (("ids",np.array([r["id"] for r in bank])),("labels",labels),("valid",valid),("hard",hard)):
        assert np.array_equal(outer[key],value)
    raw = np.full((2,len(bank),4,3,2), np.nan, np.float32)
    normalized = np.full_like(raw,np.nan)
    scores = np.full((2,len(bank),4,4),np.nan,np.float64)
    scores[:,:,:,0] = outer["mixture_scores"][:,:,:,0]
    scales = np.full((2,3,3,2),np.nan,np.float64)
    records, readouts, policies = [], [], []
    resolved = 0
    with threadpool_limits(limits=4):
        for repeat, outer_seed in enumerate(FOLD_SEEDS):
            for fold in range(3):
                fitting = np.flatnonzero(outer["folds"][repeat]!=fold)
                held = np.flatnonzero(outer["folds"][repeat]==fold)
                folder = OUT/f"repeat{repeat}_fold{fold}"
                folder.mkdir()
                np.savez(folder/"partition.npz",fitting=fitting,held=held)
                splits=[]
                for split in range(3):
                    seed=outer_seed+1000*fold+split+1
                    perm=np.random.default_rng(seed).permutation(fitting)
                    cutoff=int(.85*len(perm)); fit,stop=perm[:cutoff],perm[cutoff:]
                    assert not np.intersect1d(fit,stop).size and not np.intersect1d(perm,held).size
                    np.savez(folder/f"inner{split}.npz",fitting=fit,stopping=stop)
                    splits.append((seed,fit,stop))
                for col,head in enumerate(protocol["neural_recipes"]):
                    epochs=[]
                    for split,(seed,fit,stop) in enumerate(splits):
                        tick=time.perf_counter()
                        asset,curve,selected=stop_fit(head,embeddings,labels,valid,pair_weights,hard,fit,stop)
                        path=folder/f"head{col}_inner{split}_terminal.npz"
                        np.savez(path,**asset)
                        dump(folder/f"head{col}_inner{split}_curve.json",curve)
                        records.append(dict(repeat=repeat,fold=fold,head=col,stage="inner_stopping",split=split,split_seed=seed,
                            fit_indices_sha256=array_hash(fit),stopping_indices_sha256=array_hash(stop),
                            fitting_questions=len(fit),fit_hard_questions=int(hard[fit].sum()),stopping_questions=len(stop),
                            epochs_executed=len(curve),selected_epoch=selected,seconds=time.perf_counter()-tick,
                            asset=path.relative_to(OUT).as_posix(),asset_sha256=digest(path),asset_scope="Terminal inner model, not a deployed refit"))
                        epochs.append(selected); resolved+=1
                        dump(OUT/"fit_records.json",records)
                        progress("fitting",len(records),f"划分{repeat+1}/折{fold+1}：头{col+1}内层早停{split+1}完成",resolved)
                        print(json.dumps(dict(completed_fits=len(records),resolved_slots=resolved,stage="inner",repeat=repeat,fold=fold,head=col,split=split)),flush=True)
                    selection=epochs_for_policy(epochs)
                    policy=dict(repeat=repeat,fold=fold,head=col,inner_best_epochs=epochs,selected_epochs=selection,
                        selection_scope="Three stopping sets inside outer fitting questions only")
                    dump(folder/f"head{col}_frozen_epochs.json",policy)
                    policies.append(policy)
                    cache={}
                    for pi,name in enumerate(POLICIES):
                        epoch=selection[name]; reused=epoch in cache
                        if not reused:
                            tick=time.perf_counter()
                            recipe={**head["training"],"epochs":epoch}
                            fit_bank=[bank[int(i)] for i in fitting]
                            asset=fit_head(head["kind"],recipe,fit_bank,embeddings[fitting],None,labels[fitting],valid[fitting],pair_weights[fitting],hard[fitting])
                            fit_scores=head_scores(head["kind"],fit_bank,embeddings[fitting],None,asset)
                            scale=score_scale(fit_scores,fit_bank)
                            z=head_scores(head["kind"],[bank[int(i)] for i in held],embeddings[held],None,asset)
                            asset["score_scale"]=np.array(scale)
                            path=folder/f"head{col}_epoch{epoch}.npz"
                            np.savez(path,**asset)
                            with zipfile.ZipFile(path) as archive:
                                expanded=sum(a.file_size for a in archive.infolist())
                            records.append(dict(repeat=repeat,fold=fold,head=col,stage="outer_refit",epochs=epoch,
                                fit_indices_sha256=array_hash(fitting),held_indices_sha256=array_hash(held),
                                fitting_questions=len(fitting),fit_hard_questions=int(hard[fitting].sum()),
                                score_scale=scale,seconds=time.perf_counter()-tick,asset=path.relative_to(OUT).as_posix(),
                                asset_sha256=digest(path),asset_uncompressed_bytes=expanded))
                            dump(OUT/"fit_records.json",records)
                            cache[epoch]=(z,scale,path)
                        z,scale,path=cache[epoch]
                        raw[repeat,held,:,pi,col]=z
                        normalized[repeat,held,:,pi,col]=(z/scale).astype(np.float32)
                        scales[repeat,fold,pi,col]=scale
                        readouts.append(dict(repeat=repeat,fold=fold,head=col,policy=name,epochs=epoch,reused_refit=reused,
                            asset=path.relative_to(OUT).as_posix(),score_scale=scale,
                            text_asset=f"optimization_baselines/repeat{repeat}_fold{fold}/d_text.npz",
                            text_score_scale=float(outer["score_scales"][repeat,fold,2])))
                        resolved+=1
                        dump(OUT/"readouts.json",readouts)
                        progress("fitting",len(records),f"划分{repeat+1}/折{fold+1}：头{col+1}策略{name}完成",resolved)
                    for pi in range(3):
                        if col==1:
                            components=np.stack((normalized[repeat,held,:,pi,0],normalized[repeat,held,:,pi,1],outer["normalized"][repeat,held,:,2].astype(np.float32)),axis=-1)
                            scores[repeat,held,:,pi+1]=components.mean(-1)
    assert resolved==72 and sum(r["stage"]=="inner_stopping" for r in records)==36
    assert len(records)==72-sum(r["reused_refit"] for r in readouts) and len(readouts)==36
    assert np.isfinite(scores).all() and np.isfinite(normalized).all()
    group_weights=baseline.routing.group.groups(bank)
    group_valid=group_weights.sum(2)>0
    good=np.einsum("ngc,nc->ng",group_weights,labels)>0
    choices=np.empty((2,len(bank),4),np.int8); losses=np.empty_like(choices,dtype=np.float64); results=[]
    for repeat in range(2):
        rows={}
        for col,name in enumerate(METHODS):
            score=scores[repeat,:,:,col]
            choice=np.asarray(select(bank,score,"mean"),np.int8)
            grouped=np.einsum("ngc,nc->ng",group_weights.astype(np.float64),score)
            loss=logsumexp(np.where(group_valid,grouped,-np.inf),axis=1)-(grouped*good).sum(1)
            choices[repeat,:,col]=choice; losses[repeat,:,col]=loss
            rows[name]={**metrics(choice,labels,hard),"group_ce":float(loss[hard].mean()),
                "paired_vs_fixed5":baseline.routing.paired(choice,choices[repeat,:,0],labels,hard,outer["folds"][repeat])}
        results.append(dict(repeat=repeat,fold_seed=FOLD_SEEDS[repeat],results=rows))
    np.savez(OUT/"oof_logits.npz",ids=outer["ids"],scores=scores,normalized=scores,raw_neural_heads=raw,
        normalized_neural_heads=normalized,score_scales=scales,method_names=np.array(METHODS),policy_names=np.array(POLICIES),
        labels=labels,valid=valid,hard=hard,folds=outer["folds"],choices=choices,group_losses=losses)
    dump(OUT/"policies.json",policies)
    assert all(digest(ROOT/name)==h for name,h in protocol["source_sha256"].items())
    assert all(digest(BASE/name)==h for name,h in protocol["baseline_evidence_sha256"].items())
    summary=dict(status="complete",results=results,model_fit_calls=len(records),inner_fits=36,
        outer_refits=sum(r["stage"]=="outer_refit" for r in records),outer_policy_readouts=36,
        cached_refit_readouts=sum(r["reused_refit"] for r in readouts),maximum_planned_fits=72,resolved_fit_slots=72,
        text_model_new_fits=0,seconds=time.perf_counter()-STARTED,source_hashes_unchanged=True,
        validation_accessed=False,test_accessed=False,data_files_read=sorted(accessed),
        caveat="Fixed repeated-training-OOF comparisons; no final model promotion or independent test of historical model search")
    dump(OUT/"summary.json",summary)
    progress("complete",len(records),"A3三次内层早停汇总完成；相同轮数复用记录与实际拟合计数分开保存",72)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--register-only",action="store_true")
    action.add_argument("--run",action="store_true")
    args=parser.parse_args()
    if args.register_only:
        p=register()
        print(json.dumps(dict(status="registered_not_started",maximum_fits=72,command="python -B optimization_stopping.py --run")))
    else:
        try:
            run()
        except Exception as error:
            if STARTED is not None:
                file=OUT/"fit_records.json"
                count=len(json.loads(file.read_text(encoding="utf-8"))) if file.exists() else 0
                progress("failed",count,f"{type(error).__name__}: {error}")
            raise
