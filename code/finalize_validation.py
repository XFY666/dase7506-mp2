"""Finite train-only-normalized fusion grid, with validation-only selection."""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '4'
import json
import argparse
import shutil
import time
from pathlib import Path
import numpy as np
from common import parse_answer
from experiment_utils import development_data, metrics, dump
from selection import select
from submission import text_scores


def score_scale(scores, bank):
    centered = scores-scores.mean(axis=1,keepdims=True)
    valid = np.array([[parse_answer(c['text']) is not None for c in b['candidates']] for b in bank])
    return float(max(np.std(centered[valid]),1e-6))


def main():
    start = time.perf_counter()
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('runs/linear_fusion_bundle'))
    args=parser.parse_args()
    root = Path('runs'); out = args.output; out.mkdir(parents=True,exist_ok=True)
    tb,te,ty,th = development_data('train')
    vb,ve,vy,vh = development_data('validation')
    text_records = json.loads((root/'text_experiments/results.json').read_text())
    best_text = max(text_records,key=lambda r:(r['metrics']['hard_correct'],r['metrics']['correct'],-r['asset_bytes']))
    emb_summary = json.loads((root/'embedding_experiments/summary.json').read_text())
    emb_key = emb_summary['best']['key']
    text_path = root/'text_experiments'/f"{best_text['name']}.npz"
    emb_path = root/'embedding_experiments/best_embedding.npz'
    with np.load(emb_path,allow_pickle=False) as asset:
        train_emb = te@asset['coefficient']+asset['intercept']
        val_emb = ve@asset['coefficient']+asset['intercept']
    with np.load(root/'text_experiments/development_features.npz',allow_pickle=False) as asset:
        tf, vf = asset['train'], asset['validation']
    with np.load(text_path,allow_pickle=False) as asset:
        train_text = text_scores(tb,te,asset,tf)
        val_text = text_scores(vb,ve,asset,vf)
    es,ts = score_scale(train_emb,tb),score_scale(train_text,tb)
    records = []
    for alpha in (0.0,.25,.5,.75,1.0):
        for aggregation in ('max','mean'):
            for bonus in (0.0,.25):
                config = {'text_weight':alpha,'aggregation':aggregation,'consensus_bonus':bonus,
                    'embedding_scale':es,'text_scale':ts,'embedding_model':emb_key,
                    'text_model':best_text['name'],'feature_version':1}
                s = (1-alpha)*val_emb/es+alpha*val_text/ts
                count = (0<alpha<1)*2+(alpha in (0,1))
                byte_count = (emb_path.stat().st_size if alpha<1 else 0)+(text_path.stat().st_size if alpha>0 else 0)
                records.append({'config':config,'metrics':metrics(select(vb,s,aggregation,bonus),vy,vh),
                                'model_count':int(count),'model_asset_bytes':byte_count})
    best = max(records,key=lambda r:(r['metrics']['hard_correct'],r['metrics']['correct'],-r['model_asset_bytes']))
    dump(root/'fusion_results.json',records)
    dump(out/'config.json',best['config'])
    if best['config']['text_weight'] < 1:
        shutil.copy2(emb_path,out/'embedding.npz')
    if best['config']['text_weight'] > 0:
        shutil.copy2(text_path,out/'text.npz')
    dump(root/'validation_selection.json',{'best':best,'best_text':best_text,
        'embedding':emb_summary['best'],'normalization':'within-question centered score std over valid training candidates',
        'selection':'hard validation correct, full validation correct, fewer asset bytes',
        'fusion_configs':len(records),'seconds':time.perf_counter()-start})
    print(json.dumps(best,indent=2))


if __name__ == '__main__':
    main()
