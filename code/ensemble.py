"""Offline candidate scoring with small NumPy checkpoints.

Each configuration explicitly names its heads, learned score scales and weights.
No training routines, references or cached predictions are needed at inference.
"""
import json
from pathlib import Path

import numpy as np

from selection import select
from text_features import extract_features
from tree_model import tree_scores


def text_scores(bank, embeddings, asset, features=None):
    if features is None:
        features = extract_features(bank)
    x = features[:, :, asset["feature_indices"]]
    if "pca_components" in asset:
        z = (embeddings - asset["pca_mean"]) @ asset["pca_components"].T
        x = np.concatenate((x, z, z-z.mean(axis=1, keepdims=True)), axis=2)
    return tree_scores(x.reshape(-1, x.shape[-1]), asset).reshape(len(bank), 4)


def head_scores(kind, bank, embeddings, features, asset):
    if kind == "nonlinear":
        return np.tanh(embeddings @ asset["w1"] + asset["b1"]) @ asset["w2"]
    if kind == "residual":
        if bool(asset["contextual"]):
            raise ValueError("This deployment supports the selected raw residual head only")
        output = embeddings @ asset["skip_raw"] + asset["b2"]
        return np.tanh(embeddings @ asset["w1_raw"] + asset["b1"]) @ asset["w2"] + output
    if kind == "joint":
        x = np.concatenate((embeddings, features), axis=2)
        return np.tanh(x @ asset["w1"] + asset["b1"]) @ asset["w2"] + x @ asset["skip"] + asset["b2"]
    if kind == "linear":
        return embeddings @ asset["coefficient"] + asset["intercept"]
    if kind == "trees":
        return text_scores(bank, embeddings, asset, features)
    if kind == "oblivious":
        from oblivious_model import catboost_scores
        return catboost_scores(embeddings, features, asset)
    raise ValueError(f"Unknown model kind: {kind}")


def score_candidates(bank, arrays, checkpoint):
    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    embeddings = np.asarray(arrays["embeddings"], dtype=np.float32)
    if embeddings.shape != (len(bank), 4, 1024):
        raise ValueError("Expected aligned embeddings [N,4,1024]")
    heads = config["heads"]
    needs_text = any(h["kind"] in ("joint", "trees", "oblivious") for h in heads)
    features = extract_features(bank) if needs_text and bank else None
    scores = np.zeros((len(bank), 4), dtype=np.float64)
    if bank:
        for head in heads:
            if head["scale"] <= 0 or head["weight"] < 0:
                raise ValueError("Weights must be nonnegative and scales positive")
            with np.load(checkpoint / head["file"], allow_pickle=False) as asset:
                raw = head_scores(head["kind"], bank, embeddings, features, asset)
            scores += head["weight"] * raw / head["scale"]
    return scores, config


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError("A checkpoint directory is required")
    if not bank:
        return []
    scores, config = score_candidates(bank, arrays, checkpoint)
    return select(bank, scores, config["aggregation"], config.get("consensus_bonus", 0.0))
