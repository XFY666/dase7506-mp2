"""NumPy deployment for a content-conditioned antisymmetric comparison model."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from common import parse_answer
from selection import select
from text_features import extract_features, FEATURE_NAMES

PAIRS = tuple((i, j) for i in range(4) for j in range(i+1, 4))
PARAMETERS = ("difference_weight", "mean_weight", "hidden_bias", "output_weight", "difference_skip")
_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}
_RELATIVE = (("relative_log_chars", "log_chars"), ("relative_log_words", "log_words"),
             ("relative_equality_correct_fraction", "equality_correct_fraction"),
             ("relative_question_number_coverage", "question_number_coverage"),
             ("relative_number_count", "log_numbers"))


def stable_text_features(bank):
    """Existing text75 features, with stable arithmetic for their five means."""
    x = extract_features(bank)
    for target, source in _RELATIVE:
        values = x[..., _INDEX[source]].astype(np.float64)
        x[..., _INDEX[target]] = (values-values.mean(axis=1, keepdims=True)).astype(np.float32)
    return x


def valid_candidates(bank):
    return np.asarray([[parse_answer(c["text"]) is not None for c in q["candidates"]]
                       for q in bank], dtype=bool).reshape(-1, 4)


def transform_inputs(embeddings, text, asset):
    e = np.asarray(embeddings, np.float64)@asset["embedding_projection"]+asset["embedding_bias"]
    t = np.asarray(text, np.float64)*asset["text_multiplier"]+asset["text_bias"]
    return np.clip(np.concatenate((e, t), axis=2), -12., 12.)


def pair_inputs(x, valid):
    difference = np.stack([x[:, i]-x[:, j] for i, j in PAIRS], axis=1)
    mean = np.stack([.5*(x[:, i]+x[:, j]) for i, j in PAIRS], axis=1)
    pair_valid = np.stack([valid[:, i]&valid[:, j] for i, j in PAIRS], axis=1)
    return difference, mean, pair_valid


def pair_values(difference, mean, model):
    wd, wm, bias, output, skip = model
    delta = difference@wd
    condition = mean@wm+bias
    plus, minus = np.tanh(delta+condition), np.tanh(-delta+condition)
    return .5*(plus-minus)@output+difference@skip


def borda(pair_score, pair_valid, valid):
    values = np.where(pair_valid, pair_score, 0.)
    scores = np.zeros(valid.shape, np.float64)
    for p, (i, j) in enumerate(PAIRS):
        scores[:, i] += values[:, p]
        scores[:, j] -= values[:, p]
    return scores/np.maximum(valid.sum(1, keepdims=True)-1, 1)


def score_features(x, valid, model):
    if not len(x):
        return np.empty((0, 4), np.float64)
    d, m, pair_valid = pair_inputs(x, valid)
    return borda(pair_values(d, m, model), pair_valid, valid)


def conditional_pair_scores(bank, embeddings, asset, features=None):
    if not bank:
        return np.empty((0, 4), np.float64)
    if features is None:
        features = stable_text_features(bank)
    x = transform_inputs(embeddings, features, asset)
    return score_features(x, valid_candidates(bank), [asset[name] for name in PARAMETERS])


def predict(bank, arrays, checkpoint):
    path = Path(checkpoint)
    if path.is_dir():
        path = path/"model.npz"
    with np.load(path, allow_pickle=False) as asset:
        scores = conditional_pair_scores(bank, arrays["embeddings"], asset)
    return select(bank, scores, "mean")
