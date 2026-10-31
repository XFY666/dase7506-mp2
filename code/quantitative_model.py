"""NumPy-only deployment of PLS8 + HGBT, with stable candidate centering.

The frozen baseline feature extractor is left unchanged. This new branch
recomputes its five centered features with float64 accumulation, then casts
back to float32. This avoids candidate-order rounding reaching tree thresholds.
"""
from pathlib import Path

import numpy as np

from common import parse_answer
from selection import select
from text_features import extract_features, FEATURE_NAMES
from tree_model import tree_scores

_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}
_CENTERED = (("relative_log_chars", "log_chars"), ("relative_log_words", "log_words"),
             ("relative_equality_correct_fraction", "equality_correct_fraction"),
             ("relative_question_number_coverage", "question_number_coverage"),
             ("relative_number_count", "log_numbers"))


def answer_valid(bank):
    return np.array([[parse_answer(c["text"]) is not None for c in row["candidates"]] for row in bank])


def stable_center(x, valid):
    x = np.asarray(x)
    values = x.astype(np.float64)
    weights = valid.astype(np.float64)
    if x.ndim == 3:
        weights = weights[..., None]
    mean = (values*weights).sum(axis=1, keepdims=True)/np.maximum(weights.sum(axis=1, keepdims=True), 1)
    return (values-mean).astype(x.dtype)


def stable_base_features(bank):
    base = extract_features(bank)
    for target, source in _CENTERED:
        x = base[..., _INDEX[source]].astype(np.float64)
        base[..., _INDEX[target]] = (x-x.mean(axis=1, keepdims=True)).astype(np.float32)
    return base


def feature_matrix(base, embeddings, valid, asset):
    embeddings = np.asarray(embeddings, dtype=np.float32)
    projected = (embeddings-asset["pls_mean"])@asset["pls_rotations"]
    return np.concatenate((base, projected, stable_center(projected, valid)), axis=-1).astype(np.float32)


def score_candidates(bank, embeddings, asset):
    """Return unnormalized [N,4] scores, using no fitted/reference data outside asset."""
    if int(asset.get("precision_fix_version", -1)) != 1 or asset["pls_rotations"].shape != (1024,8):
        raise ValueError("Expected precision-repaired PLS8 checkpoint")
    if np.shape(embeddings) != (len(bank),4,1024):
        raise ValueError("Expected embeddings with shape [N,4,1024]")
    x = feature_matrix(stable_base_features(bank), embeddings, answer_valid(bank), asset)
    return tree_scores(x.reshape(-1,91), asset).reshape(-1,4)


def predict(bank, arrays, checkpoint_path):
    path = Path(checkpoint_path)
    if path.is_dir(): path = path/"selected_model.npz"
    with np.load(path, allow_pickle=False) as archive:
        asset = {key: archive[key] for key in archive.files}
    return select(bank, score_candidates(bank, arrays["embeddings"], asset), "mean", 0.)
