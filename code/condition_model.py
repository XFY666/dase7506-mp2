"""NumPy-only scoring for the bounded condition-correspondence tree study."""
from pathlib import Path

import numpy as np

from condition_features import extract_features as condition_features
from quantitative_model import answer_valid, stable_base_features, stable_center
from selection import select
from tree_model import tree_scores

DESIGNS=("base75","condition","condition_pls8")


def feature_matrix(base, conditions, embeddings, valid, design, asset=None):
    pieces=[base]
    if design!="base75":pieces.append(conditions)
    if design=="condition_pls8":
        e=np.asarray(embeddings,dtype=np.float32)
        projected=(e-asset["pls_mean"])@asset["pls_rotations"]
        pieces.extend((projected,stable_center(projected,valid)))
    result=np.concatenate(pieces,axis=-1).astype(np.float32)
    if not np.isfinite(result).all():raise ValueError("Nonfinite model feature")
    return result


def score_candidates(bank, embeddings, asset):
    """Raw [N,4] scores. Normalize externally by asset['score_scale'] if fusing."""
    if np.shape(embeddings)!=(len(bank),4,1024):raise ValueError("Expected [N,4,1024] embeddings")
    if int(asset.get("condition_model_version",-1))!=1:raise ValueError("Unexpected condition asset version")
    design=DESIGNS[int(asset["design_code"])]
    if not bank:return np.empty((0,4),dtype=np.float64)
    base=stable_base_features(bank)
    conditions=condition_features(bank) if design!="base75" else None
    x=feature_matrix(base,conditions,embeddings,answer_valid(bank),design,asset)
    if x.shape[-1]!=int(asset["feature_count"]):raise ValueError("Feature schema mismatch")
    return tree_scores(x.reshape(-1,x.shape[-1]),asset).reshape(-1,4)


def predict(bank, arrays, checkpoint_path):
    path=Path(checkpoint_path)
    if path.is_dir():path=path/"selected_model.npz"
    with np.load(path,allow_pickle=False) as a:asset={k:a[k] for k in a.files}
    return select(bank,score_candidates(bank,arrays["embeddings"],asset),"mean",0.)

