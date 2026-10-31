"""Training-only matched length/EMA/input-noise study, two repeated folds."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
import answer_group_experiment as group
from experiment_utils import development_data, dump, metrics
from selection import select
from text_features import extract_features

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'runs/optimization_regularization'
FOLD_SEEDS = (20261027, 20261028)
SEEDS = (7506, 17001)
NAMES = ('epoch14', 'epoch40', 'epoch40_ema', 'epoch40_ema_noise')
CONFIG = dict(representation='group', objective='pairwise', dropout=.2, hidden=128, l2=.01)


def checkpoint_fit(x, weights, good, valid, seed, noise):
    model = group.initialize(x.shape[-1], 128, seed)
    first = [np.zeros_like(p) for p in model]; second = [np.zeros_like(p) for p in model]
    rng = np.random.default_rng(seed + 1); augment = np.random.default_rng(seed + 303)
    count = 0; snapshots = {}; ema = None
    for epoch in range(1, 41):
        order = rng.permutation(len(x))
        for start in range(0, len(x), 128):
            ids = order[start:start + 128]
            inputs = x[ids]
            if noise:
                inputs = inputs + augment.normal(0, noise, inputs.shape).astype(np.float32)
            _, grads = group.loss_gradient(inputs, weights[ids], good[ids], valid[ids], model, CONFIG, rng)
            count += 1
            for j, grad in enumerate(grads):
                first[j] = .9 * first[j] + .1 * grad
                second[j] = .999 * second[j] + .001 * grad * grad
                model[j] -= .001 * (first[j] / (1 - .9 ** count)) / (np.sqrt(second[j] / (1 - .999 ** count)) + 1e-8)
            if epoch > 5:
                if ema is None:
                    ema = [p.copy() for p in model]
                else:
                    for j, p in enumerate(model):
                        ema[j] *= .99; ema[j] += .01 * p
        if epoch == 14:
            snapshots['epoch14'] = [p.copy() for p in model]
    snapshots['epoch40'] = [p.copy() for p in model]
    snapshots['epoch40_ema'] = ema
    assert all(np.isfinite(p).all() for m in snapshots.values() for p in m)
    return snapshots, count


def ce(z, weights, good, valid, hard):
    grouped = np.einsum('ngc,nc->ng', weights, z)
    z = np.where(valid[hard], grouped[hard], -np.inf)
    maximum = z.max(1)
    return float(np.mean(maximum + np.log(np.exp(z-maximum[:,None]).sum(1)) - (grouped[hard] * good[hard]).sum(1)))


def main():
    if (OUT / 'protocol.json').exists():
        raise RuntimeError('Protocol/output already exists; do not overwrite a study')
    OUT.mkdir(parents=True, exist_ok=True)
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
               (Path(__file__), ROOT/'answer_group_experiment.py', ROOT/'text_features.py', ROOT/'selection.py')}
    dump(OUT/'protocol.json', dict(fold_seeds=FOLD_SEEDS, seeds=SEEDS, recipes=NAMES, configuration=CONFIG,
        experiments='A3 EMA and longer training, F2 Gaussian input noise', epochs=40, retained_epoch=14,
        ema_decay=.99, ema_after_epoch=5, input_noise_standard_deviation=.05,
        design='Each clean trajectory supplies 14/40/EMA readouts; separate noisy trajectory supplies EMA. No epoch chosen from held-out scores.',
        model_fit_calls=24, raw_score_readouts=48, within_recipe_all_fixed_seeds_averaged=True,
        boundaries='All fit statistics/scales use outer fitting questions; only train data loaded. Same fixed repeated folds as A2.',
        caveat='Two repeats reuse questions; OOF selection is exploratory. A3 stopping-fold aggregation is not tested by this fixed-length EMA comparison.',
        source_sha256=sources))
    started = time.perf_counter(); completed = 0
    def progress(stage, detail, result=None):
        value=dict(stage=stage,completed_fits=completed,planned_fits=24,detail=detail,
                   elapsed_seconds=time.perf_counter()-started,updated_utc=datetime.now(timezone.utc).isoformat())
        if result is not None:value['result']=result
        dump(OUT/'progress.json',value)
    progress('features','准备训练输入与答案组特征')
    bank, embeddings, labels, hard = development_data('train')
    weights = group.groups(bank); gv = weights.sum(2)>0
    good = np.einsum('ngc,nc->ng',weights,labels)>0
    valid = weights.sum(1)>0
    features = group.make_features(embeddings,extract_features(bank),weights,'group')
    folds = np.empty((2,len(bank)),np.int8)
    raw = np.empty((2,len(bank),4,4,2),np.float64); normalized=raw.copy()
    records=[]
    with threadpool_limits(limits=4):
        for repeat, seed in enumerate(FOLD_SEEDS):
            folds[repeat,np.random.default_rng(seed).permutation(len(bank))]=np.arange(len(bank))%3
            for fold in range(3):
                fitting=np.flatnonzero(folds[repeat]!=fold); held=np.flatnonzero(folds[repeat]==fold)
                mean,std=group.fit_statistics(features[fitting],gv[fitting]); x=group.normalized(features,mean,std)
                active=fitting[hard[fitting]]
                directory=OUT/f'repeat{repeat}_fold{fold}';directory.mkdir()
                np.savez(directory/'partition.npz',fit_indices=fitting,held_indices=held)
                for si, init in enumerate(SEEDS):
                    for noise in (0.,.05):
                        tick=time.perf_counter()
                        snapshots,steps=checkpoint_fit(x[active],weights[active],good[active],gv[active],init,noise)
                        recipes=[('epoch40_ema_noise',snapshots['epoch40_ema'])] if noise else list(snapshots.items())
                        asset_records=[]
                        for name, model in recipes:
                            column=NAMES.index(name)
                            all_scores=group.candidate_scores(group.scores(x,model),weights,'group')
                            scale=group.scale_scores(all_scores[fitting],valid[fitting])
                            raw[repeat,held,:,column,si]=all_scores[held]
                            normalized[repeat,held,:,column,si]=all_scores[held]/scale
                            asset=directory/f'{name}_seed{init}.npz'
                            np.savez(asset,w1=model[0],b1=model[1],w2=model[2],skip=model[3],mean=mean,scale=std,score_scale=np.array(scale))
                            asset_records.append(dict(recipe=name,path=str(asset.relative_to(OUT)),sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),scale=scale))
                        completed+=1
                        records.append(dict(repeat=repeat,fold=fold,seed=init,noise=noise,epochs=40,steps=steps,
                                            fit_seconds=time.perf_counter()-tick,fit_questions=len(active),assets=asset_records))
                        dump(OUT/'fit_records.json',records)
                        progress('fitting',f'第{repeat+1}组划分，第{fold+1}折，种子{init}，输入噪声{noise}')
                        print(f'fit {completed}/24',flush=True)
    combined=normalized.mean(-1); results=[]
    for col,name in enumerate(NAMES):
        repeats=[]
        for r in range(2):
            choice=np.array(select(bank,combined[r,:,:,col],'mean'))
            control=np.array(select(bank,combined[r,:,:,0],'mean'))
            yes=labels[np.arange(len(bank)),choice].astype(bool);base=labels[np.arange(len(bank)),control].astype(bool)
            repeats.append(dict(metrics=metrics(choice,labels,hard),group_ce=ce(combined[r,:,:,col],weights,good,gv,hard),
                                repairs=int((yes&~base&hard).sum()),regressions=int((~yes&base&hard).sum()),
                                fold_correct=[int(yes[hard&(folds[r]==f)].sum()) for f in range(3)]))
        results.append(dict(recipe=name,repeats=repeats,mean_hard_correct=float(np.mean([v['metrics']['hard_correct'] for v in repeats])),
                            mean_group_ce=float(np.mean([v['group_ce'] for v in repeats]))))
    selected=min(results,key=lambda v:(-v['mean_hard_correct'],v['mean_group_ce'],NAMES.index(v['recipe'])))
    np.savez_compressed(OUT/'oof_logits.npz',raw_members=raw,normalized_members=normalized,normalized=combined,
                        ids=np.array([b['id'] for b in bank]),labels=labels,valid=valid,hard=hard,folds=folds,recipes=NAMES)
    assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h for name,h in sources.items())
    summary=dict(recipes=results,selected=selected,model_fit_calls=completed,readouts=48,seeds=SEEDS,
                 elapsed_seconds=time.perf_counter()-started,validation_evaluations=0,test_access=False,
                 caveat='Fixed training-side repetitions, no independent significance claim; no full-training model fit or validation.')
    dump(OUT/'summary.json',summary)
    result='；'.join(v['recipe']+': '+str([a['metrics']['hard_correct'] for a in v['repeats']]) for v in results)
    progress('complete','24次轨迹拟合完成，全部读出保留；没有验证或测试评分',result)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
