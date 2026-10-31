"""Registered fixed-seed averaging of matched group-network readouts."""
from __future__ import annotations
import os
import sys
sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
import answer_group_experiment as group
from experiment_utils import development_data, dump, metrics
from optimization_regularization import checkpoint_fit, ce
from selection import select
from text_features import extract_features

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'runs/optimization_bagging'
PRIOR = ROOT / 'runs/optimization_regularization'
SEEDS = (7506, 17001, 17002, 17003, 17004)
NAMES = ('epoch14', 'epoch40_ema', 'epoch40_ema_noise')
SIZES = (1, 2, 3, 5)
SOURCES = ('optimization_bagging.py', 'optimization_regularization.py', 'answer_group_experiment.py',
           'answer_group_model.py', 'text_features.py', 'selection.py', 'experiment_utils.py')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def register():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = dict(purpose='B1 fixed initialization averaging; no seed selected by score',
        seeds=SEEDS, bag_sizes=SIZES, readouts=NAMES, additional_seeds=SEEDS[2:],
        reused_seeds=SEEDS[:2], reused_directory='optimization_regularization',
        fresh_trajectories=36, new_readout_assets=54, retained_but_not_compared='clean epoch40',
        fold_seeds=(20261027, 20261028), folds=3,
        fit='Same registered40epoch clean/noise trajectories, normalizers, dropout, optimizer and EMA as prior regularization study',
        supervision='hard fitting questions only; normalization all fitting valid groups; held labels never enter fit',
        averaging='Normalize each seed by its own fitting-side score scale then uniform average of first k fixed seeds',
        selection='No seed cherry-picking. For later fusion use all5 seeds regardless of the highest individual or bag-size readout.',
        caveat='Same training questions and already screened recipes; descriptive stability, not independent validation',
        validation_access=False, test_access=False,
        source_sha256={name:digest(ROOT/name) for name in SOURCES},
        reused_sha256={name:digest(PRIOR/name) for name in ('protocol.json','oof_logits.npz','fit_records.json')})
    path=OUT/'protocol.json'
    if path.exists():
        assert json.loads(path.read_text(encoding='utf-8')) == json.loads(json.dumps(protocol)), 'Changed registration'
    else:
        dump(path,protocol)
        dump(OUT/'progress.json',dict(stage='queued',completed_fits=0,planned_fits=36,detail='5个固定种子平均：复用前2个，新增3个种子'))
    return protocol


def run():
    protocol=register()
    assert not (OUT/'fit_records.json').exists(), 'Preserve existing experiment'
    def guard(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(args[0])).resolve()
        if path.name.startswith(('gsm8k_validation','gsm8k_test')):raise RuntimeError('Forbidden split')
        mode,flags=args[1],args[2] or 0
        writing=isinstance(mode,str) and any(c in mode for c in 'wax+') or flags & (os.O_WRONLY|os.O_RDWR)
        if writing and OUT.resolve() not in path.parents:raise RuntimeError('Write outside experiment')
    sys.addaudithook(guard)
    started=time.perf_counter();records=[]
    def progress(stage,detail):
        dump(OUT/'progress.json',dict(stage=stage,completed_fits=len(records),planned_fits=36,detail=detail,
            elapsed_seconds=time.perf_counter()-started,updated_utc=datetime.now(timezone.utc).isoformat()))
    progress('features','准备匹配的组特征')
    bank,emb,labels,hard=development_data('train')
    weights=group.groups(bank);gv=weights.sum(2)>0;valid=weights.sum(1)>0
    good=np.einsum('ngc,nc->ng',weights,labels)>0
    features=group.make_features(emb,extract_features(bank),weights,'group')
    old=np.load(PRIOR/'oof_logits.npz',allow_pickle=False)
    ids=np.array([b['id'] for b in bank]);assert np.array_equal(ids,old['ids'])
    assert np.array_equal(labels,old['labels']) and np.array_equal(hard,old['hard'])
    folds=old['folds'];raw=np.full((2,len(bank),4,3,5),np.nan);normalized=raw.copy()
    raw[...,:2]=old['raw_members'][:,:,:, [0,2,3],:]
    normalized[...,:2]=old['normalized_members'][:,:,:, [0,2,3],:]
    with threadpool_limits(limits=4):
        for r in range(2):
            for fold in range(3):
                fitting=np.flatnonzero(folds[r]!=fold);held=np.flatnonzero(folds[r]==fold)
                mean,std=group.fit_statistics(features[fitting],gv[fitting]);x=group.normalized(features,mean,std)
                active=fitting[hard[fitting]];directory=OUT/f'repeat{r}_fold{fold}';directory.mkdir()
                np.savez(directory/'partition.npz',fit_indices=fitting,held_indices=held)
                for si,seed in enumerate(SEEDS[2:],2):
                    for noise in (0.,.05):
                        tick=time.perf_counter()
                        snapshots,steps=checkpoint_fit(x[active],weights[active],good[active],gv[active],seed,noise)
                        selected=[('epoch40_ema_noise',snapshots['epoch40_ema'])] if noise else [(n,snapshots[n]) for n in NAMES[:2]]
                        assets=[]
                        for name,model in selected:
                            col=NAMES.index(name);scores=group.candidate_scores(group.scores(x,model),weights,'group')
                            scale=group.scale_scores(scores[fitting],valid[fitting])
                            raw[r,held,:,col,si]=scores[held];normalized[r,held,:,col,si]=scores[held]/scale
                            path=directory/f'{name}_seed{seed}.npz'
                            np.savez(path,w1=model[0],b1=model[1],w2=model[2],skip=model[3],mean=mean,scale=std,score_scale=np.array(scale))
                            assets.append(dict(recipe=name,path=str(path.relative_to(OUT)),sha256=digest(path),scale=scale))
                        records.append(dict(repeat=r,fold=fold,seed=seed,noise=noise,epochs=40,steps=steps,
                            fit_seconds=time.perf_counter()-tick,fit_questions=len(active),assets=assets))
                        dump(OUT/'fit_records.json',records);progress('fitting',f'第{r+1}组第{fold+1}折，种子{seed}，噪声{noise}')
    assert np.isfinite(raw).all() and np.isfinite(normalized).all()
    readouts=[];bag_scores=[]
    for col,name in enumerate(NAMES):
        for size in SIZES:
            scores=normalized[:,:,:,col,:size].mean(-1);bag_scores.append(scores);repeats=[]
            for r in range(2):
                choice=np.array(select(bank,scores[r],'mean'));base=np.array(select(bank,normalized[r,:,:,col,0],'mean'))
                yes=labels[np.arange(len(bank)),choice].astype(bool);ref=labels[np.arange(len(bank)),base].astype(bool)
                repeats.append(dict(metrics=metrics(choice,labels,hard),group_ce=ce(scores[r],weights,good,gv,hard),
                    repairs=int((yes&~ref&hard).sum()),regressions=int((~yes&ref&hard).sum()),
                    fold_correct=[int(yes[hard&(folds[r]==f)].sum()) for f in range(3)]))
            readouts.append(dict(name=f'{name}_bag{size}',recipe=name,seeds=SEEDS[:size],repeats=repeats))
    np.savez_compressed(OUT/'oof_logits.npz',raw_members=raw,normalized_members=normalized,
        normalized=np.stack(bag_scores,axis=-1),config_names=np.array([v['name'] for v in readouts]),
        ids=ids,labels=labels,valid=valid,hard=hard,folds=folds,seeds=SEEDS,recipes=NAMES)
    assert all(digest(ROOT/n)==h for n,h in protocol['source_sha256'].items())
    assert all(digest(PRIOR/n)==h for n,h in protocol['reused_sha256'].items())
    dump(OUT/'summary.json',dict(status='complete',configs=readouts,model_fit_calls=len(records),
        new_readout_assets=sum(len(v['assets']) for v in records),reused_seed_count=2,
        full_model_fitted=False,validation_evaluations=0,test_access=False,seconds=time.perf_counter()-started,
        decision='All5 fixed seeds for subsequent fusion; bag-size comparisons are descriptive only'))
    progress('complete','36次新增轨迹拟合完成，所有固定种子和平均规模结果已保存')


if __name__=='__main__':
    parser=argparse.ArgumentParser();action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--register-only',action='store_true');action.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.register_only:register()
    else:run()
