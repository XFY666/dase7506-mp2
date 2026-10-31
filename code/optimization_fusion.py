"""Finite prospective fusion menu on existing training-only OOF scores.

This is transparent development selection, not an unbiased nested evaluation.
"""
from __future__ import annotations
import os
import sys
sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='4'
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from answer_group_model import group_weights
from experiment_utils import development_data,dump,metrics
from selection import select

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/optimization_fusion'
# Every source uses fitting-only scales from its own registered outer fit.
BRANCHES={
 'A':('optimization_baselines','head_names','A'),
 'K':('optimization_baselines','head_names','K'),
 'Q':('optimization_baselines','head_names','Q'),
 'P':('optimization_baselines','head_names','P'),
 'C':('optimization_baselines','head_names','C'),
 'A14_5':('optimization_bagging','config_names','epoch14_bag5'),
 'Aema_5':('optimization_bagging','config_names','epoch40_ema_bag5'),
 'Anoise_5':('optimization_bagging','config_names','epoch40_ema_noise_bag5'),
 'Across_3':('optimization_interactions','config_names','A_D1D2'),
 'Qcross':('optimization_interactions','config_names','Q_D1D2'),
 'Ccross':('optimization_interactions','config_names','C_D1D2'),
 'Asubset_3':('optimization_subset','configuration_names','drop_one_half'),
 'roles_cnn':('optimization_sequence','recipes','headtail_roles'),
 'allq_cnn':('optimization_sequence_aux','configs','control_allq'),
 'aux_cnn':('optimization_sequence_aux','configs','token_aux01'),
 'token_attention':('optimization_transformer','configs','token_attention'),
 'set_attention':('optimization_transformer','configs','token_set_attention'),
 'node_mlp':('optimization_graph','configuration_names','node_mlp'),
 'equation_gcn':('optimization_graph','configuration_names','equation_gcn'),
 'digit_cnn':('optimization_reference_augmentation','configs','digit_text_control'),
 'corruption_cnn':('optimization_reference_augmentation','configs','reference_corruption01'),
 'cross_tree':('optimization_trees','configuration_names','weighted_D1_D2'),
}
SUPPORT=('roles_cnn','allq_cnn','aux_cnn','token_attention','set_attention','node_mlp','equation_gcn','digit_cnn','corruption_cnn','cross_tree')


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def blend(first,second,weight):
    out={k:v*(1-weight) for k,v in first.items()}
    for k,v in second.items():out[k]=out.get(k,0.)+weight*v
    return out


def menu():
    f=dict(A=.5,K=.25,Q=.25)
    fnoise=dict(Anoise_5=.5,K=.25,Qcross=.25)
    gnoise=blend(fnoise,dict(P=2/3,Ccross=1/3),.75)
    anchors={
        'F':f,'G':blend(f,dict(P=2/3,C=1/3),.75),
        'F_bag14_cross':dict(A14_5=.5,K=.25,Qcross=.25),
        'F_ema_cross':dict(Aema_5=.5,K=.25,Qcross=.25),
        'F_noise_cross':fnoise,
        'F_group_cross':dict(Across_3=.5,K=.25,Qcross=.25),
        'G_noise_cross':gnoise,
        'F_subset_cross':dict(Asubset_3=.5,K=.25,Qcross=.25),
    }
    rows=[dict(name=n,weights=w,anchor=n,support=None,support_weight=0.) for n,w in anchors.items()]
    for name in ('F_noise_cross','G_noise_cross'):
        for support in SUPPORT:
            for alpha in (.10,.25):
                rows.append(dict(name=f'{name}__{support}_{alpha:g}',weights=blend(anchors[name],{support:1.},alpha),
                                 anchor=name,support=support,support_weight=alpha))
    assert len(rows)==48 and all(abs(sum(v['weights'].values())-1)<1e-12 for v in rows)
    return rows


def register():
    OUT.mkdir(parents=True,exist_ok=True)
    sources={n:digest(ROOT/n) for n in ('optimization_fusion.py','selection.py','answer_group_model.py','experiment_utils.py')}
    protocols={name:digest(ROOT/'runs'/name/'protocol.json') for name in sorted({v[0] for v in BRANCHES.values()})}
    protocol=dict(menu=menu(),branches=BRANCHES,fold_seeds=[20261027,20261028],number_of_recipes=48,
        score_convention='Use source normalized OOF directly, cast to float64; no pooled rescaling or fitting fusion coefficients',
        membership='Ten support branches fixed before observing any joint fusion outcomes, including matched controls and weak individual treatments',
        primary='Mean hard correct across2repeats; next lower mean group CE; then fewer positive-weight branch components; then menu order',
        screen_filter='Each repeat hard-correct >= historicalF on that repeat; descriptive consistency filter, not a significance test',
        duplicates='Identical parsed-answer group choices over both repeats share one shortlist position; all menu rows still reported',
        shortlist='Rank remaining distinct choices for full-training resource assessment; at most2 resource-eligible finalists proceed to new validation',
        resource_rule='Before any new validation, full refit/probe in ranked order; reject >16MiB uncompressed assets or >25s local fresh prediction+score on467 train inputs, reserving5s of the30s reference allowance. ReferenceCPU remains eligibility authority.',
        resource_scope='Resource failure permits next ranked fixed recipe, never new weights or model tuning',
        bootstrap='5000 question-cluster resamples of the two-repeat average paired outcome; conditional on fixed searched predictions, excludes training/search uncertainty; no acceptance test',
        statistical_scope='Historical and present screening use these train labels. This finite search is exploratory; two repeats are not independent datasets. No cross-OOF meta fitting is called nested.',
        validation='None in this program; prospective final deployment selection rule must precede the atmost2 validation evaluations',
        test='No test access, no promotion, no score-based rollback',
        source_sha256=sources,input_protocol_sha256=protocols)
    path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text(encoding='utf-8'))==json.loads(json.dumps(protocol)),'Changed registration'
    else:
        dump(path,protocol)
        dump(OUT/'progress.json',dict(stage='queued',completed_fits=0,planned_fits=0,detail='48个固定组合已登记，等待全部训练侧预测'))
    return protocol


def group_losses(score,weights,gv,good):
    z=np.einsum('ngc,nc->ng',weights,score)
    masked=np.where(gv,z,-np.inf);mx=masked.max(1)
    return mx+np.log(np.exp(masked-mx[:,None]).sum(1))-(z*good).sum(1)


def run():
    protocol=register();assert not (OUT/'summary.json').exists(),'Preserve completed menu'
    def guard(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve()
        if p.name.startswith(('gsm8k_validation','gsm8k_test')):raise RuntimeError('Forbidden split')
        mode,flags=args[1],args[2] or 0
        if ((isinstance(mode,str) and any(c in mode for c in 'wax+')) or flags&(os.O_WRONLY|os.O_RDWR)) and OUT.resolve() not in p.parents:
            raise RuntimeError('Write outside fusion evidence')
    sys.addaudithook(guard)
    started=time.perf_counter()
    bank,_,labels,hard=development_data('train');ids=np.array([r['id'] for r in bank])
    weights=group_weights(bank).astype(np.float64);gv=weights.sum(2)>0
    valid=weights.sum(1)>0;good=np.einsum('ngc,nc->ng',weights,labels)>0
    cache={};sources={};branch_scores={};folds=None
    for name,(folder,field,configuration) in BRANCHES.items():
        if folder not in cache:
            path=ROOT/'runs'/folder/'oof_logits.npz'
            assert (path.parent/'summary.json').exists(),folder+' not complete'
            with np.load(path,allow_pickle=False) as a:cache[folder]={key:a[key] for key in a.files}
            sources[folder]=dict(oof_sha256=digest(path),summary_sha256=digest(path.parent/'summary.json'))
        a=cache[folder]
        for key,expected in [('ids',ids),('labels',labels),('valid',valid),('hard',hard)]:assert np.array_equal(a[key],expected),(folder,key)
        if folds is None:folds=a['folds']
        assert np.array_equal(folds,a['folds']),folder
        column=list(a[field]).index(configuration)
        value=a['normalized'][...,column].astype(np.float64)
        assert value.shape==(2,len(bank),4) and np.isfinite(value).all()
        branch_scores[name]=value
    dump(OUT/'consumed_evidence.json',sources)
    rows=[];all_choices=[];all_losses=[];all_scores=[]
    for mi,recipe in enumerate(protocol['menu']):
        score=sum(weight*branch_scores[name] for name,weight in recipe['weights'].items())
        choices=np.stack([select(bank,score[r],'mean') for r in range(2)])
        loss=np.stack([group_losses(score[r],weights,gv,good) for r in range(2)])
        yes=labels[np.arange(len(bank))[None,:],choices].astype(bool)
        repeats=[]
        for r in range(2):
            m=metrics(choices[r],labels,hard)
            m.update(group_ce=float(loss[r,hard].mean()),fold_correct=[int(yes[r,hard&(folds[r]==f)].sum()) for f in range(3)])
            repeats.append(m)
        row=dict(**recipe,repeats=repeats,mean_hard_correct=float(yes[:,hard].sum(1).mean()),
                 mean_group_ce=float(loss[:,hard].mean()),menu_index=mi)
        all_choices.append(choices);all_losses.append(loss);all_scores.append(score);rows.append(row)
    base=labels[np.arange(len(bank))[None,:],all_choices[0]].astype(bool)
    for i,row in enumerate(rows):
        yes=labels[np.arange(len(bank))[None,:],all_choices[i]].astype(bool)
        row['paired_vs_F']=[dict(repairs=int((yes[r]&~base[r]&hard).sum()),regressions=int((~yes[r]&base[r]&hard).sum()),
                                 gain=int((yes[r,hard].astype(int)-base[r,hard]).sum())) for r in range(2)]
        row['consistent_vs_F']=all(v['gain']>=0 for v in row['paired_vs_F'])
    order=sorted(range(len(rows)),key=lambda i:(-rows[i]['mean_hard_correct'],rows[i]['mean_group_ce'],len(rows[i]['weights']),i))
    unique=[];seen=set();group_ids=np.argmax(weights>0,axis=1)
    for i in order:
        if not rows[i]['consistent_vs_F']:continue
        key=group_ids[np.arange(len(bank))[None,:],all_choices[i]].tobytes()
        if key not in seen:unique.append(i);seen.add(key)
    rng=np.random.default_rng(20261029)
    intervals={}
    for i in unique[:2]:
        yes=labels[np.arange(len(bank))[None,:],all_choices[i]].astype(bool)
        diff=(yes[:,hard].astype(float)-base[:,hard]).mean(0)
        means=np.concatenate([diff[rng.integers(0,len(diff),(250,len(diff)))].mean(1) for _ in range(20)])
        intervals[rows[i]['name']]=dict(mean_accuracy_difference=float(diff.mean()),
            conditional_question_bootstrap_95=np.quantile(means,[.025,.975]).tolist(),replicates=5000,
            interpretation='Fixed searched predictions only; excludes search/training uncertainty; not confirmatory')
    np.savez_compressed(OUT/'oof_logits.npz',ids=ids,labels=labels,valid=valid,hard=hard,folds=folds,
        normalized=np.stack(all_scores,axis=-1),choices=np.stack(all_choices,axis=-1),group_losses=np.stack(all_losses,axis=-1),
        config_names=np.array([r['name'] for r in rows]))
    summary=dict(status='complete',model_fit_calls=0,menu_readouts=48,results=rows,
        ranked_distinct_consistent=[rows[i]['name'] for i in unique],
        provisional_first_two=[rows[i] for i in unique[:2]],conditional_intervals=intervals,
        resource_eligibility='Not yet measured; ordered shortlist is provisional until complete fullfit resource checks',
        validation_evaluations=0,test_access=False,full_model_fitted=False,seconds=time.perf_counter()-started,
        source_hashes_unchanged=all(digest(ROOT/n)==h for n,h in protocol['source_sha256'].items()))
    assert summary['source_hashes_unchanged']
    dump(OUT/'summary.json',summary)
    dump(OUT/'progress.json',dict(stage='complete',completed_fits=0,planned_fits=0,
        detail='48个固定训练侧组合完成，等待全训练重拟合与资源检查',updated_utc=datetime.now(timezone.utc).isoformat(),
        result='；'.join(rows[i]['name']+': '+str([r['hard_correct'] for r in rows[i]['repeats']]) for i in unique[:2])))


if __name__=='__main__':
    p=argparse.ArgumentParser();a=p.add_mutually_exclusive_group(required=True)
    a.add_argument('--register-only',action='store_true');a.add_argument('--run',action='store_true');args=p.parse_args()
    if args.register_only:register()
    else:run()
