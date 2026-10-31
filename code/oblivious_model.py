"""NumPy-only inference for numeric symmetric CatBoost trees.

Schema: flattened split_features(int32), split_borders(float64), leaf_values
(float64), split_offsets/leaf_offsets(int32, one extra terminal offset), scalar
scale/bias(float64), and optional pca_mean/pca_components(float32). Conversion
copies every split and leaf without rounding, quantization, or retraining.
"""
from __future__ import annotations

import numpy as np


def compact_model(model, pca=None):
    """Convert an already loaded numeric CatBoost JSON dictionary to arrays."""
    mapping = {f["feature_index"]: f["flat_feature_index"]
               for f in model["features_info"]["float_features"]}
    features, borders, leaves, split_offsets, leaf_offsets = [], [], [], [0], [0]
    for tree in model["oblivious_trees"]:
        splits = tree.get("splits") or []
        if len(tree["leaf_values"]) != 2 ** len(splits):
            raise ValueError("Only scalar-output symmetric trees are supported")
        for split in splits:
            if split["split_type"] != "FloatFeature":
                raise ValueError("Only numeric CatBoost splits are supported")
            features.append(mapping[split["float_feature_index"]])
            borders.append(split["border"])
        leaves.extend(tree["leaf_values"])
        split_offsets.append(len(features))
        leaf_offsets.append(len(leaves))
    scale, bias = model["scale_and_bias"]
    if len(bias) != 1:
        raise ValueError("Only scalar-output models are supported")
    asset = {
        "split_features": np.asarray(features, dtype=np.int32),
        "split_borders": np.asarray(borders, dtype=np.float64),
        "split_offsets": np.asarray(split_offsets, dtype=np.int32),
        "leaf_offsets": np.asarray(leaf_offsets, dtype=np.int32),
        "leaf_values": np.asarray(leaves, dtype=np.float64),
        "scale": np.asarray(scale, dtype=np.float64),
        "bias": np.asarray(bias[0], dtype=np.float64),
    }
    if pca is not None:
        asset["pca_mean"] = np.asarray(pca["mean"])
        asset["pca_components"] = np.asarray(pca["components"])
    return asset


def compact_scores(x, asset):
    """Return raw scores for an arbitrary number of feature rows."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("Feature rows must be a two-dimensional array")
    total = np.zeros(len(x), dtype=np.float64)
    for tree in range(len(asset["split_offsets"]) - 1):
        start, stop = asset["split_offsets"][tree:tree + 2]
        leaf = np.zeros(len(x), dtype=np.int64)
        for depth, split in enumerate(range(int(start), int(stop))):
            leaf |= (x[:, asset["split_features"][split]] > asset["split_borders"][split]).astype(np.int64) << depth
        total += asset["leaf_values"][asset["leaf_offsets"][tree] + leaf]
    return float(asset["scale"]) * total + float(asset["bias"])


def catboost_scores(embeddings, text_features, asset):
    """Score [N,4,1024] embeddings with matching [N,4,75] text features."""
    z = (embeddings - asset["pca_mean"]) @ asset["pca_components"].T
    x = np.concatenate((text_features, z, z - z.mean(axis=1, keepdims=True)), axis=2)
    return compact_scores(x.reshape(-1, x.shape[-1]), asset).reshape(x.shape[:2])
