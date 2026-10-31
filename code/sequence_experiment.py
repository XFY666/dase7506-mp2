"""Two prespecified from-scratch CNN recipes, selected on training folds only."""
from __future__ import annotations

import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="4"

import copy
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch import nn

from common import parse_answer
from experiment_utils import development_data,dump,metrics
from sequence_model import build_vocabulary,encode_bank,numeric_inputs,transform_numeric,numpy_forward,score_candidates
from selection import select

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"runs"/"sequence_experiments"
SEED=20261015
INIT_SEED=7506
CONFIGS=("text_cnn","joint_cnn")
MAX_EPOCHS=24
PATIENCE=4
BATCH=64


def protocol():
    return {"recipes":list(CONFIGS),"outer_folds":3,"fold_seed":SEED,
        "fold_mapping":"order=default_rng(20261015).permutation(N); fold[order]=arange(N)%3",
        "inner_split":"85/15 whole questions within outer fitting set, seed20261115+fold; full selected recipe uses seed20262015",
        "initialization_seed":INIT_SEED,"vocabulary":"at most4096, fit side only; frequency then lexicographic tie; numeric literals mapped to <num>",
        "sequence":"first64 question tokens + separator + at most256 candidate head/tail tokens with gap marker; maximum322 tokens; long character input bounded by12000 head+12000 tail",
        "architecture":{"token_embedding":32,"convolution_widths":[3,5],"filters_each":32,"activation":"ReLU","pooling":"valid-token max and mean for each kernel","head":"tanh64 then scalar","joint_inputs":"1024 supplied embedding + existing75 text features, fitting-side normalized/clipped[-8,8]"},
        "initialization":"all neural parameters random, embedding N(0,.1), pad row zero; no external/pretrained weights or language model calls",
        "optimizer":{"name":"AdamW","learning_rate":.001,"weight_decay":.001,"gradient_clip":5.},
        "objective":"cross-entropy over parsed-answer GROUP MEAN scores; all valid groups vs correct answer group; equal weight per eligible question",
        "training_questions":"must contain a correct candidate and an incorrect valid candidate; vocabulary/numeric statistics may use all fitting-side input questions, never held-out inputs",
        "early_stop":{"metric":"inner heldout group-mean cross-entropy","max_epochs":MAX_EPOCHS,"patience":PATIENCE,"minimum_improvement":1e-4},
        "refit":"rebuild vocabulary/normalizers on all outer-fit questions and fresh same-seed model; train at inner-selected epoch count",
        "selection":"pooled training OOF hard accuracy, then lower normalized group loss, then text-only on exact tie",
        "aggregation":"mean","consensus_bonus":0.,"full_recipe":"After recipe selection, a new85/15 training-only split selects full-fit epoch count, then full refit; validation opened once afterward",
        "max_model_fits":14,"fold0_cost_probe":"first fold's two recipe costs retained; stop/report if estimated total exceeds one hour or NumPy inference exceeds30s for798 questions, rather than enlarging scope",
        "numpy_export":"vocabulary and numeric arrays in npz; inference imports no torch; compare scores and chosen answers to torch",
        "scale":"std of within-all-four-candidate centered fitting-question scores on valid fitting candidates only",
        "threads":4,"assets_limit_bytes":16*1024*1024,
        "official_install_sources":["https://pytorch.org/get-started/locally/","https://download.pytorch.org/whl/cpu/torch/"],
        "torch_version":"2.14.1+cpu","environment":"E:/Edge/7506/MP2/.local/sequenceenv; separate from existing project venv",
        "data_scope":["train","validation"],"test_access":False}


class WordCNN(nn.Module):
    def __init__(self,vocabulary_size,joint):
        super().__init__();self.joint=joint
        self.embedding=nn.Embedding(vocabulary_size,32,padding_idx=0)
        self.conv3=nn.Conv1d(32,32,3,padding=1)
        self.conv5=nn.Conv1d(32,32,5,padding=2)
        self.hidden=nn.Linear(128+(1099 if joint else 0),64)
        self.output=nn.Linear(64,1)
        with torch.no_grad():self.embedding.weight.normal_(0,.1);self.embedding.weight[0].zero_()

    def forward(self,codes,numeric):
        shape=codes.shape;codes=codes.reshape(-1,shape[-1]);mask=codes.ne(0)
        x=self.embedding(codes).transpose(1,2);pooled=[]
        for conv in (self.conv3,self.conv5):
            z=torch.relu(conv(x)).transpose(1,2)
            maximum=z.masked_fill(~mask[:,:,None],-1e9).amax(dim=1)
            mean=(z*mask[:,:,None]).sum(dim=1)/mask.sum(dim=1,keepdim=True).clamp_min(1)
            pooled.extend((maximum,mean))
        if self.joint:pooled.append(numeric.reshape(-1,1099))
        return self.output(torch.tanh(self.hidden(torch.cat(pooled,dim=1)))).reshape(shape[0],4)


def group_structure(bank,labels):
    weights=np.zeros((len(bank),4,4),dtype=np.float32);valid_groups=np.zeros((len(bank),4),bool)
    good=np.zeros_like(valid_groups);valid=np.zeros((len(bank),4),bool)
    for q,row in enumerate(bank):
        groups={}
        for j,c in enumerate(row["candidates"]):
            a=parse_answer(c["text"])
            if a is not None:groups.setdefault(a,[]).append(j);valid[q,j]=True
        for g,indices in enumerate(groups.values()):
            weights[q,g,indices]=1./len(indices);valid_groups[q,g]=True;good[q,g]=labels[q,indices].any()
    eligible=good.any(1)&((valid_groups&~good).any(1))
    return weights,valid_groups,good,valid,eligible


def objective(scores,weights,valid,good):
    gs=torch.bmm(weights,scores[:,:,None]).squeeze(-1)
    return (torch.logsumexp(gs.masked_fill(~valid,-1e9),dim=1)-torch.logsumexp(gs.masked_fill(~good,-1e9),dim=1)).mean()


def numpy_group_loss(scores,weights,valid,good,eligible):
    gs=np.einsum('ngc,nc->ng',weights,scores)
    def logsum(mask):
        masked=np.where(mask,gs,-np.inf);m=masked.max(axis=1)
        return m+np.log(np.exp(masked-m[:,None]).sum(axis=1))
    return float((logsum(valid)-logsum(good))[eligible].mean())


def prepare(bank,raw,valid,fit_ids):
    started=time.perf_counter();vocabulary=build_vocabulary(bank,fit_ids)
    codes,info=encode_bank(bank,vocabulary)
    values=raw[fit_ids][valid[fit_ids]].astype(np.float64)
    transform={"vocabulary":vocabulary,"numeric_mean":values.mean(0).astype(np.float32),
               "numeric_scale":np.maximum(values.std(0),1e-3).astype(np.float32)}
    numeric=transform_numeric(raw,transform)
    return codes,numeric,transform,{"fit_questions":len(fit_ids),"vocabulary_size":len(vocabulary),
             "fit_ids_sha256":hashlib.sha256(np.ascontiguousarray(fit_ids).tobytes()).hexdigest(),
             "seconds":time.perf_counter()-started,"all_bank_encoding_diagnostic":info}


def native_scores(model,codes,numeric):
    model.eval();out=[]
    with torch.no_grad():
        for start in range(0,len(codes),BATCH):out.append(model(codes[start:start+BATCH],numeric[start:start+BATCH]).numpy())
    return np.concatenate(out)


def train(codes,numeric,structure,fit_ids,joint,epochs,held_ids=None):
    weights,valid,good=structure
    torch.manual_seed(INIT_SEED);rng=np.random.default_rng(INIT_SEED+1)
    model=WordCNN(int(codes.max())+1,joint)
    # Encoded unknowns cannot increase max token ID above vocabulary size. A
    # fitting-side vocabulary may include unused last tokens in the full bank;
    # max code nevertheless covers every token actually passed to this model.
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    best=float('inf');selected=0;stale=0;curves=[];started=time.perf_counter()
    for epoch in range(1,epochs+1):
        model.train();order=rng.permutation(fit_ids);loss_sum=0.
        for start in range(0,len(order),BATCH):
            ids=order[start:start+BATCH];optimizer.zero_grad(set_to_none=True)
            z=model(codes[ids],numeric[ids]);loss=objective(z,weights[ids],valid[ids],good[ids])
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            loss_sum+=float(loss.detach())*len(ids)
        row={"epoch":epoch,"fit_loss":loss_sum/len(fit_ids),"elapsed_seconds":time.perf_counter()-started}
        if held_ids is not None:
            model.eval();total=0.
            with torch.no_grad():
                for start in range(0,len(held_ids),BATCH):
                    ids=held_ids[start:start+BATCH];z=model(codes[ids],numeric[ids])
                    total+=float(objective(z,weights[ids],valid[ids],good[ids]))*len(ids)
            current=total/len(held_ids);row['heldout_loss']=current
            if current<best-1e-4:best=current;selected=epoch;stale=0
            else:stale+=1
        curves.append(row)
        if held_ids is not None and stale>=PATIENCE:break
    return model,curves,selected if held_ids is not None else epochs


def export(model,transform):
    asset={"joint":np.array(model.joint),"token_embedding":model.embedding.weight.detach().numpy(),
           "vocabulary":transform["vocabulary"]}
    if model.joint:asset.update({k:transform[k] for k in ('numeric_mean','numeric_scale')})
    for name in ('conv3','conv5','hidden','output'):
        layer=getattr(model,name);asset[name+'_weight']=layer.weight.detach().numpy();asset[name+'_bias']=layer.bias.detach().numpy()
    return asset


def tensor_inputs(codes,numeric,structure):
    return torch.from_numpy(codes),torch.from_numpy(numeric),tuple(torch.from_numpy(v) for v in structure)


def fit_recipe(bank,raw,labels,valid,eligible,structure,fit_ids,inner_seed,joint,stem):
    order=np.random.default_rng(inner_seed).permutation(fit_ids);cut=int(.85*len(order))
    inner_ids,held_ids=order[:cut],order[cut:]
    codes,numeric,transform,info=prepare(bank,raw,valid,inner_ids)
    codes_t,numeric_t,structure_t=tensor_inputs(codes,numeric,structure)
    model,curve,epochs=train(codes_t,numeric_t,structure_t,inner_ids[eligible[inner_ids]],joint,MAX_EPOCHS,held_ids[eligible[held_ids]])
    dump(OUT/f'{stem}_early_curve.json',curve)
    early_seconds=curve[-1]['elapsed_seconds']
    codes,numeric,transform,outer_info=prepare(bank,raw,valid,fit_ids)
    codes_t,numeric_t,structure_t=tensor_inputs(codes,numeric,structure)
    model,curve,_=train(codes_t,numeric_t,structure_t,fit_ids[eligible[fit_ids]],joint,epochs)
    dump(OUT/f'{stem}_refit_curve.json',curve)
    z=native_scores(model,codes_t,numeric_t)
    asset=export(model,transform)
    scale=float(max(np.std((z[fit_ids]-z[fit_ids].mean(1,keepdims=True))[valid[fit_ids]]),1e-6))
    asset['score_scale']=np.array(scale)
    np.savez(OUT/f'{stem}.npz',**asset)
    tick=time.perf_counter();replay=numpy_forward(codes[:64],numeric[:64],asset)
    numpy_seconds=time.perf_counter()-tick;error=float(np.max(abs(replay-z[:64])))
    if error>1e-4:raise AssertionError(('export',stem,error))
    if select(bank[:64],replay,'mean')!=select(bank[:64],z[:64],'mean'):raise AssertionError(('choice export',stem))
    record={'name':stem,'joint':joint,'selected_epochs':epochs,'early_fit_seconds':early_seconds,
            'refit_seconds':curve[-1]['elapsed_seconds'],'inner_transform':info,'outer_transform':outer_info,
            'score_scale':scale,'asset_bytes':(OUT/f'{stem}.npz').stat().st_size,
            'numpy_64_question_seconds':numpy_seconds,'export_max_abs_error':error}
    dump(OUT/f'{stem}_record.json',record);print(json.dumps(record),flush=True)
    return model,asset,z,record


def main():
    started=time.perf_counter();OUT.mkdir(exist_ok=True)
    if (OUT/'summary.json').exists():raise FileExistsError('Completed finite study exists')
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    assert torch.version.cuda is None
    dump(OUT/'protocol.json',protocol())
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('sequence_model.py','sequence_experiment.py','text_features.py')}
    dump(OUT/'environment.json',{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,
                              'torch_cuda_version':torch.version.cuda,'threads':torch.get_num_threads(),'source_sha256':hashes})
    bank,embeddings,labels,hard=development_data('train')
    weights,groups,good,valid,eligible=group_structure(bank,labels);structure=(weights,groups,good)
    tick=time.perf_counter();raw=numeric_inputs(bank,embeddings);feature_seconds=time.perf_counter()-tick
    order=np.random.default_rng(SEED).permutation(len(bank));fold=np.empty(len(bank),dtype=np.int32);fold[order]=np.arange(len(bank))%3
    np.savez(OUT/'fold_assignment.npz',ids=np.array([r['id'] for r in bank]),fold=fold,labels=labels,valid=valid,eligible=eligible,hard=hard)
    oof={name:np.full((len(bank),4),np.nan,dtype=np.float32) for name in CONFIGS}
    normalized={name:np.full((len(bank),4),np.nan,dtype=np.float32) for name in CONFIGS};records=[]
    for f in range(3):
        fit_ids=np.flatnonzero(fold!=f);held_ids=np.flatnonzero(fold==f)
        for name in CONFIGS:
            model,asset,z,record=fit_recipe(bank,raw,labels,valid,eligible,structure,fit_ids,SEED+100+f,name=='joint_cnn',f'fold{f}_{name}')
            oof[name][held_ids]=z[held_ids];normalized[name][held_ids]=z[held_ids]/record['score_scale']
            record['fold']=f;record['heldout_metrics']=metrics(select([bank[i] for i in held_ids],z[held_ids],'mean'),labels[held_ids],hard[held_ids])
            records.append(record);dump(OUT/'fold_records.json',records)
        if f==0:
            elapsed=time.perf_counter()-started
            estimate=sum(r['early_fit_seconds']+r['refit_seconds'] for r in records)*3.6
            max_inference_estimate=max(r['numpy_64_question_seconds'] for r in records)*798/64
            probe={'elapsed_seconds':elapsed,'estimated_full_study_fit_seconds':estimate,
                   'estimated_numpy_798_seconds':max_inference_estimate,'scope':'first planned fold retained, no settings selected from this probe'}
            dump(OUT/'fold0_cost_probe.json',probe)
            if estimate>3600 or max_inference_estimate>30:raise RuntimeError('Prespecified cost probe limit; report without expanding scope')
        np.savez(OUT/'partial_oof.npz',**oof,completed_folds=np.arange(f+1))
    rows=[]
    for name in CONFIGS:
        rows.append({'name':name,'metrics':metrics(select(bank,oof[name],'mean'),labels,hard),
                     'normalized_group_loss':numpy_group_loss(normalized[name],weights,groups,good,eligible)})
    winner=min(rows,key=lambda r:(-r['metrics']['hard_correct'],r['normalized_group_loss'],CONFIGS.index(r['name'])))
    dump(OUT/'training_selection.json',{'candidates':rows,'selected':winner,'validation_not_loaded':True})
    name=winner['name']
    np.savez(OUT/'selected_oof_logits.npz',ids=np.array([r['id'] for r in bank]),raw=oof[name],normalized=normalized[name],
             fold=fold,labels=labels,valid=valid,eligible=eligible,hard=hard,
             fold_scales=np.array([r['score_scale'] for r in records if r['name'].endswith(name)]))
    np.savez(OUT/'all_oof_logits.npz',**oof)
    model,asset,z,full=fit_recipe(bank,raw,labels,valid,eligible,structure,np.arange(len(bank)),SEED+1000,name=='joint_cnn','selected_model')
    assert (OUT/'selected_model.npz').stat().st_size<=16*1024*1024
    dump(OUT/'selected_config.json',{'recipe':name,'epochs':full['selected_epochs'],'score_scale':full['score_scale'],
          'aggregation':'mean','module':'sequence_model.py','checkpoint':'selected_model.npz','selected_before_validation':True})
    # Only now load validation. It does not select the recipe or the epoch count.
    vb,ve,vy,vh=development_data('validation')
    tick=time.perf_counter();vz=score_candidates(vb,ve,asset);inference_seconds=time.perf_counter()-tick
    vc,_=encode_bank(vb,asset['vocabulary'])
    vn=transform_numeric(numeric_inputs(vb,ve),asset) if bool(asset['joint']) else np.zeros((len(vb),4,0),dtype=np.float32)
    native=native_scores(model,torch.from_numpy(vc),torch.from_numpy(vn))
    error=float(np.max(abs(vz-native)));assert error<1e-4
    choices=select(vb,vz,'mean');assert choices==select(vb,native,'mean')
    p=np.array([2,0,3,1]);pb=copy.deepcopy(vb[:32])
    for row in pb:
        row['id']='not-used';row['candidates']=[row['candidates'][i] for i in p]
        for c in row['candidates']:c['index']=999
    pz=score_candidates(pb,ve[:32,p],asset);perr=float(np.max(abs(pz-vz[:32,p])))
    assert perr<1e-4
    np.savez(OUT/'validation_logits.npz',ids=np.array([r['id'] for r in vb]),raw=vz,normalized=vz/full['score_scale'],labels=vy,hard=vh,
             valid=np.array([[parse_answer(c['text']) is not None for c in r['candidates']] for r in vb]),choices=np.array(choices))
    summary={'protocol':protocol(),'selected':winner,'validation':metrics(choices,vy,vh),'full_model':full,
             'asset_bytes':(OUT/'selected_model.npz').stat().st_size,'validation_numpy_inference_seconds':inference_seconds,
             'validation_export_max_abs_error':error,'permutation_and_changed_id_max_score_error':perr,
             'torch_vs_numpy_validation_choice_differences':0,'feature_seconds':feature_seconds,
             'source_sha256':hashes,'total_seconds':time.perf_counter()-started,
             'selection_caveat':'Winner training OOF is used for recipe selection; final independent validation report is bounded to the selected recipe. No external/pretrained resources or test access.'}
    dump(OUT/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
