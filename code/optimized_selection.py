"""Portable offline ensemble with explicit components and NumPy inference."""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='4'
import hashlib
import json
from pathlib import Path
import numpy as np
from selection import select


def _safe_path(root,relative):
    path=(root/relative).resolve()
    if root not in path.parents:raise ValueError('Checkpoint member must stay inside checkpoint')
    return path


def score_candidates(bank,arrays,checkpoint):
    directory=Path(checkpoint).resolve()
    config=json.loads((directory/'config.json').read_text(encoding='utf-8'))
    if config.get('format')!='mp2-integrated-verifier-v1':raise ValueError('Unknown checkpoint format')
    emb=np.asarray(arrays['embeddings'],np.float32)
    if emb.shape!=(len(bank),4,1024) or not np.isfinite(emb).all():raise ValueError('Expected finite aligned embeddings[N,4,1024]')
    components=config['components']
    if not components or abs(sum(float(x['weight']) for x in components)-1)>1e-9:raise ValueError('Weights must sum to one')
    score=np.zeros((len(bank),4),np.float64)
    for entry in components:
        weight=float(entry['weight'])
        if weight<=0 or not np.isfinite(weight):raise ValueError('Invalid component weight')
        manifest=_safe_path(directory,entry['manifest'])
        if hashlib.sha256(manifest.read_bytes()).hexdigest()!=entry['manifest_sha256']:raise ValueError('Changed component manifest')
        metadata=json.loads(manifest.read_text(encoding='utf-8'))
        if entry['adapter']=='core':
            from optimization_core_refit import score_branch
            for member in metadata['members']:
                path=_safe_path(manifest.parent,member['file'])
                if hashlib.sha256(path.read_bytes()).hexdigest()!=member['sha256']:raise ValueError('Changed model asset')
        elif entry['adapter']=='sequence':
            from optimization_sequence_refit import score_branch
        elif entry['adapter']=='structure':
            from optimization_structure_refit import score_branch
        else:raise ValueError('Unsupported component adapter')
        fresh=score_branch(bank,emb,manifest,normalize=True)
        if fresh.shape!=score.shape or not np.isfinite(fresh).all():raise ValueError('Invalid component scores')
        score+=weight*fresh.astype(np.float64)
    return score,config


def predict(bank,arrays,checkpoint):
    if checkpoint is None:raise ValueError('A checkpoint directory is required')
    if not bank:return []
    score,_=score_candidates(bank,arrays,checkpoint)
    return select(bank,score,'mean')
