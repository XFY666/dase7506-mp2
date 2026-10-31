"""NumPy inference for the trained answer-group scorer; no fitting imports."""
import json
from pathlib import Path
import numpy as np
from common import parse_answer
from selection import select
from text_features import extract_features


def group_weights(bank):
    weights=np.zeros((len(bank),4,4),np.float32)
    for i,row in enumerate(bank):
        answers={}
        for j,candidate in enumerate(row['candidates']):
            answer=parse_answer(candidate['text'])
            if answer is not None:
                answers.setdefault(answer,[]).append(j)
        for k,indices in enumerate(answers.values()):
            weights[i,k,indices]=1/len(indices)
    return weights


def features_for_groups(embeddings,text,weights,representation):
    membership=weights>0
    count=membership.sum(axis=2)
    ng=(count>0).sum(axis=1)
    support=np.stack((np.log(np.maximum(count,1)),count/np.maximum(count.sum(axis=1,keepdims=True),1),
                      np.broadcast_to(ng[:,None],count.shape)),axis=2).astype(np.float32)
    if representation=='candidate':
        return np.concatenate((embeddings,text,np.einsum('ngc,ngd->ncd',membership,support)),axis=2).astype(np.float32)
    mean_e=np.einsum('ngc,ncd->ngd',weights,embeddings)
    mean_t=np.einsum('ngc,ncd->ngd',weights,text)
    maxima=np.zeros_like(mean_t)
    deviation=np.zeros_like(mean_t)
    for i in range(len(weights)):
        for g in range(4):
            members=membership[i,g]
            if members.any():
                maxima[i,g]=text[i,members].max(axis=0)
                deviation[i,g]=text[i,members].std(axis=0,dtype=np.float64)
    return np.concatenate((mean_e,mean_t,maxima,deviation,support),axis=2).astype(np.float32)


def group_model_scores(bank,embeddings,asset,representation,features=None):
    if not bank:
        return np.empty((0,4),np.float32)
    if features is None:
        features=extract_features(bank)
    weights=group_weights(bank)
    x=features_for_groups(embeddings,features,weights,representation)
    x=np.clip((x-asset['mean'])/asset['scale'],-12,12).astype(np.float32)
    raw=np.tanh(x@asset['w1']+asset['b1'])@asset['w2']+x@asset['skip']
    return raw if representation=='candidate' else np.einsum('ngc,ng->nc',weights>0,raw)


def predict(bank,arrays,checkpoint):
    checkpoint=Path(checkpoint)
    config=json.loads((checkpoint/'selected_config.json').read_text(encoding='utf-8'))
    with np.load(checkpoint/'selected.npz',allow_pickle=False) as asset:
        raw=group_model_scores(bank,np.asarray(arrays['embeddings'],np.float32),asset,config['configuration']['representation'])
    return select(bank,raw,'mean')
