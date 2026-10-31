"""NumPy-only compact token and candidate-set attention inference."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import numpy as np

from common import parse_answer
from selection import select
from sequence_model import numeric_inputs, transform_numeric
from optimization_sequence_model import sequences, encode_sequences


def positions(length=322, width=32):
    row = np.arange(length, dtype=np.float64)[:, None]
    scale = np.exp(-np.log(10000.)*np.arange(0, width, 2, dtype=np.float64)/width)
    result = np.empty((length, width), np.float32)
    result[:, 0::2] = np.sin(row*scale)
    result[:, 1::2] = np.cos(row*scale)
    return result


def layer_norm(x, weight, bias):
    centered = x-x.mean(axis=-1, keepdims=True)
    variance = (centered*centered).mean(axis=-1, keepdims=True)
    return centered/np.sqrt(variance+np.float32(1e-5))*weight+bias


def attention_block(x, valid, asset, prefix, heads=4):
    b, length, width = x.shape
    valid = np.asarray(valid, bool).copy()
    valid[~valid.any(axis=1)] = True
    packed = x@asset[prefix+"_qkv_weight"].T+asset[prefix+"_qkv_bias"]
    q, k, value = np.split(packed, 3, axis=-1)
    def reshape(a):
        return a.reshape(b, length, heads, width//heads).transpose(0, 2, 1, 3)
    q, k, value = map(reshape, (q, k, value))
    logits = (q@k.transpose(0, 1, 3, 2))*np.float32((width//heads)**-.5)
    logits = np.where(valid[:, None, None, :], logits, np.float32(-1e9))
    probability = np.exp(logits-logits.max(axis=-1, keepdims=True))
    probability /= probability.sum(axis=-1, keepdims=True)
    attended = (probability@value).transpose(0, 2, 1, 3).reshape(b, length, width)
    projected = attended@asset[prefix+"_projection_weight"].T+asset[prefix+"_projection_bias"]
    x = layer_norm(x+projected, asset[prefix+"_norm1_weight"], asset[prefix+"_norm1_bias"])
    hidden = np.maximum(x@asset[prefix+"_ff1_weight"].T+asset[prefix+"_ff1_bias"], 0)
    residual = hidden@asset[prefix+"_ff2_weight"].T+asset[prefix+"_ff2_bias"]
    return layer_norm(x+residual, asset[prefix+"_norm2_weight"], asset[prefix+"_norm2_bias"])


def numpy_forward(codes, numeric, valid, asset, batch_candidates=16):
    n, _, length = codes.shape
    if n == 0:
        return np.empty((0, 4), np.float32)
    flat = codes.reshape(-1, length)
    numeric_flat = numeric.reshape(-1, 1099)
    features = np.empty((len(flat), 64), np.float32)
    position = positions(length, 32)
    for start in range(0, len(flat), batch_candidates):
        ids = flat[start:start+batch_candidates]
        mask = ids != 0
        x = asset["token_embedding"][ids]*np.float32(np.sqrt(32.))+position
        x = attention_block(x, mask, asset, "token")
        maximum = np.where(mask[:, :, None], x, np.float32(-1e9)).max(axis=1)
        mean = (x*mask[:, :, None]).sum(axis=1)/np.maximum(mask.sum(axis=1, keepdims=True), 1)
        pooled = np.concatenate((maximum, mean.astype(np.float32), numeric_flat[start:start+len(ids)]), axis=1)
        features[start:start+len(ids)] = np.tanh(pooled@asset["hidden_weight"].T+asset["hidden_bias"])
    features = features.reshape(n, 4, 64)
    if bool(asset["set_attention"]):
        features = attention_block(features, valid, asset, "set")
    return (features@asset["output_weight"].T+asset["output_bias"]).reshape(n, 4).astype(np.float32)


def valid_answers(bank):
    return np.array([[parse_answer(c["text"]) is not None for c in row["candidates"]] for row in bank], bool).reshape(-1, 4)


def score_candidates(bank, embeddings, asset):
    if not bank:
        return np.empty((0, 4), np.float32)
    codes, _ = encode_sequences(sequences(bank, "headtail_roles"), asset["vocabulary"])
    numeric = transform_numeric(numeric_inputs(bank, embeddings), asset)
    return numpy_forward(codes, numeric, valid_answers(bank), asset)


def predict(bank, arrays, checkpoint_path):
    path = Path(checkpoint_path)
    if path.is_dir():
        path /= "selected_model.npz"
    with np.load(path, allow_pickle=False) as data:
        asset = {k: data[k] for k in data.files}
    return select(bank, score_candidates(bank, arrays["embeddings"], asset), "mean")


def readiness(directory):
    directory = Path(directory)
    records = []
    with np.load(directory/"readiness_inputs.npz", allow_pickle=False) as data:
        codes, numeric, valid = (data[k] for k in ("codes", "numeric", "valid"))
        for name in ("token_attention", "token_set_attention"):
            with np.load(directory/f"{name}_readiness_asset.npz", allow_pickle=False) as saved:
                asset = {k: saved[k] for k in saved.files}
            z = numpy_forward(codes, numeric, valid, asset)
            expected = data[name]
            error = float(np.max(abs(z-expected)))
            assert error < 2e-5, (name, error)
            perm = np.array([2, 0, 3, 1])
            zp = numpy_forward(codes[:, perm], numeric[:, perm], valid[:, perm], asset)
            perr = float(np.max(abs(zp-z[:, perm])))
            assert perr < 2e-5
            assert numpy_forward(codes[:0], numeric[:0], valid[:0], asset).shape == (0, 4)
            split = np.concatenate((numpy_forward(codes[:1], numeric[:1], valid[:1], asset),
                                    numpy_forward(codes[1:], numeric[1:], valid[1:], asset)))
            berr = float(np.max(abs(split-z)))
            assert berr < 2e-5
            records.append({"recipe": name, "torch_numpy_max_abs_error": error,
                "permutation_max_abs_error": perr, "batch_max_abs_error": berr, "empty": True})
    result = {"numpy": np.__version__, "torch_imported": "torch" in sys.modules,
              "sklearn_imported": "sklearn" in sys.modules, "checks": records,
              "model_fits": 0, "data_access": "synthetic arrays only"}
    assert not result["torch_imported"] and not result["sklearn_imported"]
    (directory/"numpy_readiness.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-assets", type=Path, required=True)
    readiness(parser.parse_args().check_assets)
