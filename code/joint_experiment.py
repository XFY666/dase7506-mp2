"""Train a joint embedding/text scorer on question-grouped development data.

The comparison tests whether learning interactions beats combining independently
trained heads. Every normalizer and stopping decision uses training data only.
"""
import os

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "4"

import hashlib
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from advanced_neural import development, fit, prediction
from experiment_utils import dump, metrics
from finalize_validation import score_scale
from selection import select

OUT = Path("runs/joint_experiments")
SEED = 7506


def statistics(x):
    rows = x.reshape(-1, x.shape[-1])
    return (rows.mean(axis=0, dtype=np.float64).astype(np.float32),
            np.maximum(rows.std(axis=0, dtype=np.float64), 1e-6).astype(np.float32))


def export(model, mean, scale):
    w1, bias, w2, skip = model
    raw_w1, raw_skip = w1 / scale[:, None], skip / scale
    return {"w1": raw_w1, "b1": bias - mean @ raw_w1, "w2": w2,
            "skip": raw_skip, "b2": np.asarray(-mean @ raw_skip, dtype=np.float32)}


def joint_scores(embeddings, features, asset):
    x = np.concatenate((embeddings, features), axis=2)
    return np.tanh(x @ asset["w1"] + asset["b1"]) @ asset["w2"] + x @ asset["skip"] + asset["b2"]


def main():
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {"seed": SEED, "features": "1024 supplied embeddings + 75 structured text features",
                "hidden": [64, 128], "objectives": ["pairwise", "listwise"], "l2": [0.001, 0.01],
                "architecture": "tanh hidden layer with linear skip, joint standardized inputs",
                "optimizer": "Adam", "learning_rate": 0.001, "batch_questions": 128,
                "stopping": "85/15 training-question split, loss, max100/patience10; full-training refit",
                "normalizer": "training partition only; full training for final refit",
                "selection": "hard validation correct, full correct, smaller checkpoint",
                "split_access": ["train", "validation"], "fits": 8}
    dump(OUT / "protocol.json", protocol)
    tb, te, ty, tv, _ = development("train")
    vb, ve, vy, vv, vh = development("validation")
    with np.load("runs/text_experiments/development_features.npz") as values:
        tf, vf = values["train"], values["validation"]
    x = np.concatenate((te, tf), axis=2).astype(np.float32)
    eligible = ty.any(axis=1) & ((~ty) & tv).any(axis=1)
    order = np.random.default_rng(SEED).permutation(len(x))
    cut = int(0.85 * len(x))
    train_ids, hold_ids = order[:cut], order[cut:]
    inner_mean, inner_scale = statistics(x[train_ids])
    train_ids, hold_ids = train_ids[eligible[train_ids]], hold_ids[eligible[hold_ids]]
    inner_x = (x[train_ids] - inner_mean) / inner_scale
    heldout_x = (x[hold_ids] - inner_mean) / inner_scale
    mean, scale = statistics(x)
    full_x = (x[eligible] - mean) / scale
    val_x = (np.concatenate((ve, vf), axis=2) - mean) / scale
    records, scores = [], {}
    for hidden in protocol["hidden"]:
        for objective in protocol["objectives"]:
            for l2 in protocol["l2"]:
                key = f"joint_H{hidden}_{objective}_L2{l2:g}"
                tick = time.perf_counter()
                with threadpool_limits(limits=4):
                    _, curve, epoch = fit(inner_x, ty[train_ids], tv[train_ids], objective=objective,
                                         l2=l2, hidden=hidden, heldout=(heldout_x, ty[hold_ids], tv[hold_ids]))
                    model, refit_curve, _ = fit(full_x, ty[eligible], tv[eligible], objective=objective,
                                              l2=l2, hidden=hidden, epochs=epoch)
                    asset = export(model, mean, scale)
                    raw = joint_scores(ve, vf, asset)
                    error = float(np.max(np.abs(raw - prediction(val_x, model))))
                    assert np.allclose(raw, prediction(val_x, model), rtol=1e-5, atol=3e-5), error
                    training_scale = score_scale(joint_scores(te, tf, asset), tb)
                path = OUT / (key + ".npz")
                np.savez(path, **asset)
                scores[key] = raw
                dump(OUT / (key + "_curves.json"), {"early_stop": curve, "refit": refit_curve})
                for agg in ("max", "mean"):
                    row = {"name": key, "hidden": hidden, "objective": objective, "l2": l2,
                           "epoch": epoch, "aggregation": agg, "training_scale": training_scale,
                           "metrics": metrics(select(vb, raw, agg), vy, vh), "bytes": path.stat().st_size,
                           "export_error": error, "seconds": time.perf_counter() - tick}
                    records.append(row)
                    print(row, flush=True)
                dump(OUT / "results.json", records)
    best = max(records, key=lambda r: (r["metrics"]["hard_correct"], r["metrics"]["correct"], -r["bytes"]))
    np.savez(OUT / "validation_logits.npz", **scores)
    dump(OUT / "summary.json", {"protocol": protocol, "best": best, "seconds": time.perf_counter() - start,
                               "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})


if __name__ == "__main__":
    main()
