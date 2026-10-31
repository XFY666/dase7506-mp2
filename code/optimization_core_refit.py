"""Explicit full-training refits and NumPy-only core-branch inference.

Importing this module never trains or opens data. ``score_branch`` accepts a
manifest.json path (or its directory). ``train_branch`` requires an unused
output directory and the supplied training bank, embeddings and labels.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zipfile
sys.dont_write_bytecode = True
import numpy as np
from answer_group_model import group_model_scores, group_weights, features_for_groups
from common import DATA
from conditional_pair import conditional_pair_scores, transform_inputs, score_features
from kernel_rank import kernel_rank_scores
from optimization_interactions import score_asset as interaction_scores
from quantitative_model import answer_valid, stable_base_features, stable_center
from selection import select
from text_features import extract_features

ROOT = Path(__file__).resolve().parent
FIVE_SEEDS = (7506, 17001, 17002, 17003, 17004)
THREE_SEEDS = FIVE_SEEDS[:3]
BRANCHES = ("A", "K", "Q", "P", "C", "A5_epoch14", "A5_ema40", "A5_noise_ema40",
            "A_D1D2", "Q_D1D2", "C_D1D2", "drop_one_half")
ALIASES = {"A14":"A", "A5noise":"A5_noise_ema40", "A5_noise":"A5_noise_ema40",
           "A5_ema":"A5_ema40", "A5_14":"A5_epoch14", "subset":"drop_one_half"}
SOURCE_FILES = ("optimization_core_refit.py", "optimization_interactions.py", "optimization_regularization.py",
    "optimization_subset.py", "optimization_trees.py", "answer_group_model.py", "answer_group_experiment.py",
    "kernel_rank.py", "conditional_pair.py", "conditional_pair_experiment.py", "condition_features.py",
    "condition_model.py", "condition_experiment.py", "quantitative_model.py", "quantitative_features.py",
    "cross_candidate_features.py", "text_features.py", "tree_model.py", "common.py", "selection.py",
    "advanced_neural.py", "experiment_utils.py")
_ACTIVE_OUTPUT = None
_AUDIT_INSTALLED = False


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(x):
    x = np.ascontiguousarray(x)
    return hashlib.sha256(str(x.dtype).encode()+str(x.shape).encode()+x.tobytes()).hexdigest()


def source_hashes():
    return {name:digest(ROOT/name) for name in SOURCE_FILES}


def _manifest(value):
    if isinstance(value, (str, os.PathLike)):
        path = Path(value)
        if path.is_dir():
            path = path/"manifest.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        directory = path.parent.resolve()
    else:
        result = dict(value)
        if "_directory" not in result:
            raise ValueError("A manifest dictionary requires _directory; alternatively pass manifest.json path")
        directory = Path(result["_directory"]).resolve()
    if result.get("format") != "mp2-core-branch-v1":
        raise ValueError("Unexpected core manifest format")
    return result, directory


def _raw_member(bank, emb, member, asset, original=None):
    family = member["family"]
    if family == "group":
        return group_model_scores(bank, emb, asset, "group", original)
    if family == "interaction":
        return interaction_scores(bank, emb, asset)
    if family == "kernel":
        return kernel_rank_scores(bank, emb, asset, original)
    if family == "pair":
        return conditional_pair_scores(bank, emb, asset)
    raise ValueError(f"Unsupported core member family: {family}")


def score_branch(bank, emb, manifest, normalize=True):
    """Return [N,4] scores, with no references or training dependencies.

    normalize=True exactly follows each OOF branch's calibration: member
    scales, uniform averaging, then subset-only ensemble scale. False returns
    the plain mean of raw member scores (not the subset OOF 'raw' field).
    """
    metadata, directory = _manifest(manifest)
    emb = np.asarray(emb, np.float32)
    if emb.shape != (len(bank),4,1024):
        raise ValueError("Expected [N,4,1024] embeddings")
    if not bank:
        return np.empty((0,4), np.float64)
    members = metadata["members"]
    if not members:
        raise ValueError("Manifest has no members")
    original = extract_features(bank) if any(m["family"] in ("group","kernel") for m in members) else None
    values = []
    for member in members:
        path = (directory/member["file"]).resolve()
        if directory not in path.parents:
            raise ValueError("Asset path escapes branch directory")
        with np.load(path, allow_pickle=False) as archive:
            asset = {k:archive[k] for k in archive.files}
        score = _raw_member(bank, emb, member, asset, original)
        if normalize:
            scale = float(member["score_scale"])
            if not np.isfinite(scale) or scale < 1e-6:
                raise ValueError("Invalid member scale")
            # Divide before float64 accumulation: group readouts retain their
            # original float32 division, as in the registered OOF experiments.
            score = score/scale
        values.append(score)
    result = np.stack(values).mean(axis=0, dtype=np.float64)
    if normalize:
        scale = float(metadata["ensemble_scale"])
        if not np.isfinite(scale) or scale < 1e-6:
            raise ValueError("Invalid ensemble scale")
        result = result/scale
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite core scores")
    return result


def predict(bank, arrays, checkpoint):
    return select(bank, score_branch(bank, arrays["embeddings"], checkpoint), "mean")


def protocol(name):
    name = ALIASES.get(name,name)
    if name not in BRANCHES:
        raise ValueError(f"Unsupported branch: {name}")
    return dict(branch=name, transform_seed=7506, neural_initialization="fixed prescribed seeds",
        group_recipe=dict(hidden=128,objective="pairwise",dropout=.2,l2=.01,batch_questions=128,learning_rate=.001),
        fixed_seeds=FIVE_SEEDS if name.startswith("A5_") else THREE_SEEDS if name in ("A_D1D2","drop_one_half") else (7506,),
        group_readout="epoch14 for A/A_D1D2/subset; A5 branches use registered40epoch checkpoint_fit and named readout",
        bag_ema=dict(epochs=40,decay=.99,starts_after_epoch=5,noise_std=.05 if name=="A5_noise_ema40" else 0.),
        kernel=dict(PCA=64,C=3.,gamma="original median-distance estimator with seed7506",objective="pair_delta",tol=1e-4,max_iter=200000),
        pair=dict(PCA=64,hidden=64,l2=.03,epochs=27),
        trees=dict(PLS=8,Q_leaves=15,C_leaves=31,rounds=300,seed=20261015,valid_hard_question_weight=4),
        subset=dict(augmentation_probability=.5,eligible_views="correct and incorrect valid candidates retained",
            full_training_augmentation_seed="model_seed+20261026, corresponding to repeat0/fold0 convention",
            recomputation="original75/groups/support/max/std recomputed after removing each slot; embeddings unchanged for retained slots",
            ensemble_scale="std of original full-fitting scores after mean of3 member-normalized scores"),
        scales="All4-candidate-centered, valid-position std on full-fitting input scores; exact family dtype preserved",
        fitting_boundary="Only supplied complete official training inputs/labels; no held-out split access or tuning",
        output="Original NumPy inference arrays, relative manifest, source/input hashes, costs and replay evidence",
        normalized_false="mean of raw member scores, without either member or ensemble calibration")


def _install_audit():
    global _AUDIT_INSTALLED
    if _AUDIT_INSTALLED:
        return
    def audit(event, args):
        if _ACTIVE_OUTPUT is None or event!="open" or not isinstance(args[0],(str,bytes,os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path.name.startswith(("gsm8k_validation","gsm8k_test")):
            raise RuntimeError("Core training cannot open validation/test files")
        mode, flags = args[1], args[2] or 0
        write = isinstance(mode,str) and any(c in mode for c in "wax+") or bool(flags&(os.O_WRONLY|os.O_RDWR))
        if write and path != _ACTIVE_OUTPUT and _ACTIVE_OUTPUT not in path.parents:
            raise RuntimeError(f"Core training write outside its branch: {path}")
    sys.addaudithook(audit)
    _AUDIT_INSTALLED = True


def train_branch(name, outdir, bank, emb, labels):
    """Explicitly fit one complete-training branch; never called on import."""
    global _ACTIVE_OUTPUT
    name = ALIASES.get(name,name)
    registered = protocol(name)
    outdir = Path(outdir).resolve()
    if outdir.exists():
        raise FileExistsError("Use a new branch directory; existing artifacts are never overwritten")
    if ROOT.parent.parent not in outdir.parents:
        raise ValueError("Keep refit outputs inside the MP2 project directory")
    if _ACTIVE_OUTPUT is not None:
        raise RuntimeError("Nested/concurrent core refits in one process are unsupported")
    for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[key]="4"
    _install_audit()
    _ACTIVE_OUTPUT = outdir
    started = time.perf_counter()
    try:
        return _train(name,outdir,bank,emb,labels,registered,started)
    finally:
        _ACTIVE_OUTPUT = None


def _train(name,outdir,bank,emb,labels,registered,started):
    # Heavy dependencies are restricted to this explicit training entry point.
    from threadpoolctl import threadpool_limits
    from common import load_bank
    import answer_group_experiment as group
    emb = np.asarray(emb,np.float32)
    labels = np.asarray(labels,dtype=np.float32)
    if emb.shape!=(len(bank),4,1024) or labels.shape!=(len(bank),4):
        raise ValueError("Expected embeddings[N,4,1024] and labels[N,4]")
    official = load_bank("train")
    if bank != official:
        raise ValueError("Full-refit inputs must be the complete unchanged official training bank")
    if not np.isfinite(emb).all() or not np.isfinite(labels).all() or not np.isin(labels,[0,1]).all():
        raise ValueError("Invalid fitting arrays")
    valid = answer_valid(bank)
    weights = group_weights(bank); gv=weights.sum(2)>0
    good = np.einsum("ngc,nc->ng",weights,labels)>0
    hard = good.any(1)&(gv&~good).any(1)
    if not np.all(good.sum(1)==1):
        raise ValueError("Expected exactly one correct answer group per training question")
    ids=np.array([r["id"] for r in bank]); allids=np.arange(len(bank),dtype=np.int64)
    outdir.mkdir(parents=True)
    hashes=source_hashes()
    registered.update(source_sha256=hashes,training_ids_sha256=array_hash(ids),embeddings_sha256=array_hash(emb),
        labels_sha256=array_hash(labels),training_questions=len(bank),hard_questions=int(hard.sum()),
        created_utc=datetime.now(timezone.utc).isoformat())
    dump(outdir/"protocol.json",registered)
    records=[];members=[];fit_outputs=[];transforms=[]
    def save_member(asset,z,family,seed,detail,seconds):
        scale=detail.pop("score_scale")
        assert np.isfinite(z).all() and np.isfinite(scale) and scale>=1e-6
        asset=dict(asset,score_scale=np.array(scale))
        file=f"member{len(members)}_seed{seed}.npz";path=outdir/file
        np.savez(path,**asset)
        with zipfile.ZipFile(path) as archive:
            size=sum(a.file_size for a in archive.infolist())
        member=dict(file=file,family=family,seed=seed,score_scale=float(scale),sha256=digest(path),uncompressed_bytes=size)
        sample=list(range(min(32,len(bank))))
        replay=_raw_member([bank[i] for i in sample],emb[sample],member,asset)
        error=float(np.max(np.abs(replay-z[sample])));assert error<2e-5
        members.append(member);fit_outputs.append(z/scale)
        # Some optimizers return their own timing. Preserve it separately from
        # the enclosing transform/training/export duration.
        detail = dict(detail)
        if "seconds" in detail:
            detail["optimizer_seconds"] = detail.pop("seconds")
        records.append({**detail,"member":len(members)-1,"seconds":seconds,"export_replay_max_error":error})
        dump(outdir/"fit_records.json",records)
        dump(outdir/"progress.json",dict(stage="fitting",completed_fits=len(records),branch=name,
            detail=f"{name}: {len(records)} member fits complete",elapsed_seconds=time.perf_counter()-started))
        print(json.dumps(dict(branch=name,completed_fits=len(records),seed=seed)),flush=True)
    with threadpool_limits(limits=4):
        if name in ("A","A5_epoch14","A5_ema40","A5_noise_ema40","A_D1D2","drop_one_half"):
            original=extract_features(bank)
            text=original
            if name=="A_D1D2":
                import cross_candidate_features as cross
                text=np.concatenate((text,cross.extract_features(bank)[:,:,cross.FEATURE_GROUPS["D1"]+cross.FEATURE_GROUPS["D2"]]),-1)
            features=features_for_groups(emb,text,weights,"group")
            mean,std=group.fit_statistics(features,gv)
            x=group.normalized(features,mean,std); active=np.flatnonzero(hard)
            transforms.append(dict(kind="group_normalizer",fit_questions=len(bank),valid_groups=int(gv.sum()),mean_sha256=array_hash(mean),scale_sha256=array_hash(std)))
            if name=="drop_one_half":
                import optimization_subset as subset
                view_x=[features];view_w=[weights];view_v=[gv]
                for slot in range(4):
                    removed,changed=subset.remove_slot(bank,emb,slot)
                    fx,fw,fv,_=subset.input_view(removed,changed)
                    view_x.append(fx);view_w.append(fw);view_v.append(fv)
                vx=group.normalized(np.stack(view_x)[:,active],mean,std)
                vw=np.stack(view_w)[:,active];vv=np.stack(view_v)[:,active]
                target,eligible=subset.label_views(vw,vv,labels[active])
                assert eligible[0].all()
            seeds=FIVE_SEEDS if name.startswith("A5_") else THREE_SEEDS if name in ("A_D1D2","drop_one_half") else (7506,)
            for seed in seeds:
                tick=time.perf_counter()
                if name.startswith("A5_"):
                    from optimization_regularization import checkpoint_fit
                    noise=.05 if name=="A5_noise_ema40" else 0.
                    snapshots,steps=checkpoint_fit(x[active],weights[active],good[active],gv[active],seed,noise)
                    key="epoch14" if name=="A5_epoch14" else "epoch40_ema"
                    model=snapshots[key];detail=dict(trajectory_epochs=40,readout=key,steps=steps,noise=noise)
                elif name=="drop_one_half":
                    model,detail=subset.fit_member(vx,vw,target,vv,eligible,seed,0,0,True)
                elif name=="A_D1D2":
                    from optimization_interactions import fit_group
                    model,steps=fit_group(x[active],weights[active],good[active],gv[active],seed)
                    detail=dict(epochs=14,steps=steps)
                else:
                    config=dict(representation="group",objective="pairwise",dropout=.2,hidden=128,l2=.01)
                    model,curve,epochs=group.fit(x[active],weights[active],good[active],gv[active],config,epochs=14)
                    detail=dict(epochs=epochs,curve=curve)
                z=group.candidate_scores(group.scores(x,model),weights,"group")
                asset=dict(w1=model[0],b1=model[1],w2=model[2],skip=model[3],mean=mean,scale=std)
                family="group"
                if name=="A_D1D2":
                    family="interaction"
                    asset.update(family_code=np.array(0,np.int32),with_cross=np.array(1,np.int8),feature_width=np.array(features.shape[-1],np.int32),interaction_version=np.array(1,np.int32))
                detail["score_scale"]=group.scale_scores(z,valid)
                save_member(asset,z,family,seed,detail,time.perf_counter()-tick)
        elif name=="K":
            from sklearn.svm import SVC
            import kernel_rank as kernel
            text=extract_features(bank);tick=time.perf_counter()
            tf=kernel.fit_transform(emb,text,7506);transforms.append(dict(kind="PCA64",seed=7506,fit_questions=len(bank)))
            x=kernel.transformed(emb,text,tf)
            design,target,weight,count=kernel.design_matrix(x,labels.astype(bool),valid,"pair_delta")
            gamma=kernel.base_gamma(design,7506)
            estimator=SVC(C=3.,kernel="rbf",gamma=gamma,cache_size=512,tol=1e-4,max_iter=200000,shrinking=True,probability=False,random_state=7506)
            estimator.fit(design,target,sample_weight=weight);assert estimator.fit_status_==0
            asset=kernel.export_model(estimator,tf,"pair_delta");z=kernel.score_transformed(x,valid,asset)
            native=kernel.native_scores(x[:32],valid[:32],estimator,"pair_delta")
            assert np.max(np.abs(native-z[:32]))<2e-6
            save_member(asset,z,"kernel",7506,dict(C=3.,gamma=gamma,fit_hard_questions=count,support_vectors=len(estimator.support_),score_scale=kernel.score_scale(z,valid)),time.perf_counter()-tick)
        elif name=="P":
            import conditional_pair_experiment as pair
            tick=time.perf_counter();text=stable_base_features(bank)
            tf,info=pair.fit_transform(emb,text,valid,64,7506);transforms.append(dict(kind="PCA64",**info))
            x=transform_inputs(emb,text,tf);active=np.flatnonzero(hard)
            model,detail=pair.fit(x[active],valid[active],weights[active].astype(np.float64),good[active],gv[active],dict(pca=64,hidden=64,l2=.03),epochs=27)
            asset=pair.exported(model,tf);z=score_features(x,valid,model)
            detail["score_scale"]=pair.score_scale(z,valid)
            save_member(asset,z,"pair",7506,detail,time.perf_counter()-tick)
        else:
            import condition_experiment as condition
            from condition_features import extract_features as conditions
            from tree_model import export_trees
            from optimization_interactions import tree_features
            import cross_candidate_features as cross
            family=name[0];with_cross=name.endswith("_D1D2");tick=time.perf_counter()
            centered=stable_center(labels[hard],valid[hard])[valid[hard]]
            assert labels.dtype==np.float32 and centered.min()<0<centered.max()
            projection,pinfo=condition.fit_pls(emb,labels,valid,hard,allids)
            assert np.any(projection["pls_rotations"]!=0)
            transforms.append(dict(kind="PLS8",**pinfo))
            base=stable_base_features(bank);condition_values=conditions(bank) if family=="C" else None
            directed=cross.extract_features(bank)[:,:,cross.FEATURE_GROUPS["D1"]+cross.FEATURE_GROUPS["D2"]] if with_cross else None
            x=tree_features(emb,base,condition_values,directed,valid,projection,family,with_cross)
            leaves=15 if family=="Q" else 31
            estimator=condition.fit_tree(x,labels,valid,hard,allids,leaves,300)
            z=estimator.decision_function(x.reshape(-1,x.shape[-1])).reshape(-1,4)
            asset={**export_trees(estimator),**projection,
                "family_code":np.array(1 if family=="Q" else 2,np.int32),"with_cross":np.array(with_cross,np.int8),
                "feature_width":np.array(x.shape[-1],np.int32),"interaction_version":np.array(1,np.int32)}
            scale=group.scale_scores(z,valid) if family=="Q" else condition.scale_scores(z,valid,allids)
            save_member(asset,z,"interaction",20261015,dict(parameters=condition.parameters(leaves,300),feature_width=x.shape[-1],score_scale=scale),time.perf_counter()-tick)
    ensemble=np.stack(fit_outputs).mean(0,dtype=np.float64)
    ensemble_scale=group.scale_scores(ensemble,valid) if name=="drop_one_half" else 1.
    manifest=dict(format="mp2-core-branch-v1",branch=name,members=members,ensemble_scale=float(ensemble_scale),
        aggregation="mean",source_sha256=hashes,protocol_sha256=digest(outdir/"protocol.json"),
        training_ids_sha256=array_hash(ids),normalization="member scale -> uniform float64 mean -> subset-only ensemble scale",
        model_fit_calls=len(records),learned_transform_fits=len(transforms))
    dump(outdir/"manifest.json",manifest)
    replay=score_branch(bank[:32],emb[:32],outdir)
    error=float(np.max(np.abs(replay-(ensemble/ensemble_scale)[:32])));assert error<2e-5
    assert source_hashes()==hashes
    dump(outdir/"transform_records.json",transforms)
    np.savez(outdir/"fitting_scores.npz",ids=ids,normalized=ensemble/ensemble_scale,
        member_normalized=np.stack(fit_outputs,-1),valid=valid,hard=hard)
    costs=dict(status="complete",branch=name,model_fit_calls=len(records),learned_transform_fits=len(transforms),
        seconds=time.perf_counter()-started,uncompressed_asset_bytes=sum(m["uncompressed_bytes"] for m in members),
        manifest_bytes=(outdir/"manifest.json").stat().st_size,export_replay_max_error=error,
        source_hashes_unchanged=True,validation_evaluations=0,test_access=False,
        artifact_scope="Payload cost is per branch; a fused release must enforce its total checkpoint size separately")
    dump(outdir/"costs.json",costs)
    dump(outdir/"progress.json",dict(stage="complete",completed_fits=len(records),branch=name,
        elapsed_seconds=costs["seconds"],detail="Full-training branch complete; no held-out evaluations"))
    # The returned handle is directly usable by score_branch; the saved JSON
    # keeps only relative asset paths so release artifacts remain portable.
    return {**manifest,"_directory":str(outdir)}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list",action="store_true")
    parser.add_argument("--branch",choices=BRANCHES)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.list:
        print(json.dumps(dict(branches=BRANCHES,training_started=False)))
    elif args.branch and args.output:
        from advanced_neural import development
        bank,emb,labels,_,_=development("train")
        train_branch(args.branch,args.output,bank,emb,labels)
    else:
        parser.error("Use --list, or explicit --branch NAME --output NEW_DIRECTORY")
