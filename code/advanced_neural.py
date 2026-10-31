"""Context-aware residual MLP direction screen using development splits only.

The preregistered screen changes objective, contextual input, and regularization.
Optimization and early stopping access training questions only; validation scores
select the finite directions after full-training refits. No test file is opened.
"""
import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "4"

import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import expit, logsumexp
from threadpoolctl import threadpool_limits

from common import DATA, load_inputs, parse_answer, read_jsonl
from selection import select

OUT = Path(__file__).resolve().parent / "runs" / "advanced_neural"
SEED = 7506


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def development(split):
    if split not in ("train", "validation"):
        raise ValueError("Only development splits are allowed")
    bank, arrays = load_inputs(split)
    references = read_jsonl(DATA / f"gsm8k_{split}.jsonl")
    assert len(bank) == len(references)
    valid, labels, hard = [], [], []
    for row, ref in zip(bank, references):
        assert row["id"] == ref["id"] and row["question"] == ref["question"]
        answers = [parse_answer(c["text"]) for c in row["candidates"]]
        correct = parse_answer(ref["answer"])
        valid.append([a is not None for a in answers])
        labels.append([a is not None and a == correct for a in answers])
        hard.append(len({a for a in answers if a is not None}) >= 2)
    return bank, arrays["embeddings"], np.asarray(labels), np.asarray(valid), np.asarray(hard)


def context_residual(embeddings, valid):
    # Validity is input-derived; no reference information enters the context.
    count = valid.sum(axis=1, keepdims=True)
    pooled = np.sum(embeddings * valid[:, :, None], axis=1, keepdims=True) / np.maximum(count[:, :, None], 1)
    pooled = np.where((count == 0)[:, :, None], embeddings.mean(axis=1, keepdims=True), pooled)
    return (embeddings - pooled).astype(np.float32)


def fit_transform(embeddings, valid):
    raw = embeddings.reshape(-1, embeddings.shape[-1])
    residual = context_residual(embeddings, valid)
    return {
        "raw_mean": raw.mean(axis=0, dtype=np.float64).astype(np.float32),
        "raw_scale": np.maximum(raw.std(axis=0, dtype=np.float64), 1e-6).astype(np.float32),
        "residual_scale": np.maximum(residual[valid].std(axis=0, dtype=np.float64), 1e-6).astype(np.float32),
    }


def transform(embeddings, valid, statistics, contextual):
    raw = (embeddings - statistics["raw_mean"]) / statistics["raw_scale"]
    if not contextual:
        return raw
    residual = context_residual(embeddings, valid) / statistics["residual_scale"]
    return np.concatenate((raw, residual), axis=2)


def initialize(input_dim, hidden, seed):
    rng = np.random.default_rng(seed)
    return [rng.normal(0, 1 / np.sqrt(input_dim), (input_dim, hidden)).astype(np.float32),
            np.zeros(hidden, dtype=np.float32),
            rng.normal(0, 1 / np.sqrt(hidden), hidden).astype(np.float32),
            np.zeros(input_dim, dtype=np.float32)]


def prediction(x, model):
    w1, bias, w2, skip = model
    return np.tanh(x @ w1 + bias) @ w2 + x @ skip


def objective_gradient(scores, labels, valid, objective):
    if objective == "pairwise":
        mask = labels[:, :, None] * ((~labels) & valid)[:, None, :]
        weight = mask / np.maximum(mask.sum(axis=(1, 2))[:, None, None], 1)
        weight = weight.astype(scores.dtype)
        difference = scores[:, None, :] - scores[:, :, None]
        loss = np.sum(np.logaddexp(0, difference) * weight) / len(scores)
        prob = expit(difference) * weight / len(scores)
        gradient = prob.sum(axis=1) - prob.sum(axis=2)
    elif objective == "listwise":
        all_scores = np.where(valid, scores, -np.inf)
        good_scores = np.where(labels, scores, -np.inf)
        all_logz = logsumexp(all_scores, axis=1, keepdims=True)
        good_logz = logsumexp(good_scores, axis=1, keepdims=True)
        loss = np.mean(all_logz - good_logz)
        gradient = (np.exp(all_scores - all_logz) - np.exp(good_scores - good_logz)) / len(scores)
    else:
        raise ValueError(objective)
    return float(loss), gradient


def loss_gradient(x, labels, valid, model, objective, l2, with_gradient=True):
    w1, bias, w2, skip = model
    hidden = np.tanh(x @ w1 + bias)
    scores = hidden @ w2 + x @ skip
    loss, score_gradient = objective_gradient(scores, labels, valid, objective)
    if not with_gradient:
        return loss
    hidden_gradient = score_gradient[:, :, None] * w2 * (1 - hidden * hidden)
    gradients = [x.reshape(-1, x.shape[-1]).T @ hidden_gradient.reshape(-1, len(w2)) + l2 * w1,
                 hidden_gradient.sum(axis=(0, 1)),
                 hidden.reshape(-1, len(w2)).T @ score_gradient.ravel() + l2 * w2,
                 x.reshape(-1, x.shape[-1]).T @ score_gradient.ravel() + l2 * skip]
    return loss, gradients


def fit(x, labels, valid, *, objective, l2, hidden=128, seed=SEED, epochs=100, heldout=None):
    model = initialize(x.shape[-1], hidden, seed)
    first = [np.zeros_like(p) for p in model]
    second = [np.zeros_like(p) for p in model]
    rng = np.random.default_rng(seed + 1)
    best_loss, selected_epoch, stale, steps = float("inf"), 0, 0, 0
    curves = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(x))
        sum_loss = 0.0
        for start in range(0, len(order), 128):
            ids = order[start:start + 128]
            loss, grads = loss_gradient(x[ids], labels[ids], valid[ids], model, objective, l2)
            sum_loss += loss * len(ids)
            steps += 1
            for k, g in enumerate(grads):
                first[k] *= 0.9
                first[k] += 0.1 * g
                second[k] *= 0.999
                second[k] += 0.001 * g * g
                model[k] -= 0.001 * (first[k] / (1 - 0.9 ** steps)) / (np.sqrt(second[k] / (1 - 0.999 ** steps)) + 1e-8)
        record = {"epoch": epoch, "train_batch_loss": sum_loss / len(x), "elapsed_seconds": time.perf_counter() - started}
        if heldout is not None:
            loss = loss_gradient(*heldout, model, objective, l2, with_gradient=False)
            record["heldout_loss"] = loss
            if loss < best_loss - 1e-6:
                best_loss, selected_epoch, stale = loss, epoch, 0
            else:
                stale += 1
        curves.append(record)
        if heldout is not None and stale >= 10:
            break
    assert all(np.isfinite(p).all() for p in model)
    return model, curves, selected_epoch if heldout is not None else epochs


def export_model(model, statistics, contextual):
    w1, b1, w2, skip = model
    raw_w = w1[:1024] / statistics["raw_scale"][:, None]
    raw_skip = skip[:1024] / statistics["raw_scale"]
    result = {"w1_raw": raw_w, "b1": b1 - statistics["raw_mean"] @ raw_w,
              "w2": w2, "skip_raw": raw_skip,
              "b2": np.asarray(-statistics["raw_mean"] @ raw_skip, dtype=np.float32),
              "contextual": np.asarray(contextual), **statistics}
    if contextual:
        result["w1_residual"] = w1[1024:] / statistics["residual_scale"][:, None]
        result["skip_residual"] = skip[1024:] / statistics["residual_scale"]
    return result


def exported_scores(embeddings, valid, asset):
    hidden = embeddings @ asset["w1_raw"] + asset["b1"]
    output = embeddings @ asset["skip_raw"] + asset["b2"]
    if bool(asset["contextual"]):
        residual = context_residual(embeddings, valid)
        hidden += residual @ asset["w1_residual"]
        output += residual @ asset["skip_residual"]
    return np.tanh(hidden) @ asset["w2"] + output


def evaluate(bank, scores, labels, hard, aggregation):
    choice = select(bank, scores, aggregation, 0.0)
    correct = labels[np.arange(len(labels)), choice]
    return {"aggregation": aggregation, "correct": int(correct.sum()), "n": len(correct),
            "accuracy": float(100 * correct.mean()), "hard_correct": int(correct[hard].sum()),
            "hard_n": int(hard.sum()), "hard_accuracy": float(100 * correct[hard].mean())}


def check_gradients():
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(3, 4, 5))
    labels = np.array([[1, 0, 0, 0], [0, 1, 1, 0], [1, 1, 0, 0]], dtype=bool)
    valid = np.array([[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 0, 1]], dtype=bool)
    model = [p.astype(np.float64) for p in initialize(5, 3, SEED)]
    records = []
    for objective in ("pairwise", "listwise"):
        _, gradients = loss_gradient(x, labels, valid, model, objective, 0.0)
        for parameter, index in ((0, (2, 1)), (1, (1,)), (2, (2,)), (3, (3,))):
            step = 1e-5
            model[parameter][index] += step
            upper = loss_gradient(x, labels, valid, model, objective, 0.0, False)
            model[parameter][index] -= 2 * step
            lower = loss_gradient(x, labels, valid, model, objective, 0.0, False)
            model[parameter][index] += step
            error = abs((upper - lower) / (2 * step) - gradients[parameter][index])
            assert error < 1e-6, (objective, parameter, error)
            records.append({"objective": objective, "parameter": parameter, "absolute_error": float(error)})
    return records


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {"seed": SEED, "objectives": ["pairwise", "listwise"],
                "inputs": ["raw", "raw_contextual"], "l2_grid": [0.001, 0.01],
                "hidden": 128, "architecture": "tanh hidden layer plus trainable linear skip",
                "optimizer": "Adam", "learning_rate": 0.001, "batch_questions": 128,
                "early_stopping": "85/15 split of training QUESTIONS, heldout objective loss, max100epochs/patience10/min_improvement1e-6",
                "listwise_loss": "logsumexp(valid candidates) minus logsumexp(correct candidates)",
                "pairwise_loss": "mean correct-versus-incorrect valid pair logistic per question",
                "eligible_training": "questions containing both a correct candidate and an incorrect valid candidate",
                "context": "candidate embedding minus within-question mean of valid candidates; no-valid fallback mean of all candidates",
                "normalizers": "raw mean/std from fitting-side candidates, residual std from fitting-side valid candidates; refit using all train only",
                "final_refit": "fresh same-seed initialization on all eligible train questions at selected epoch count",
                "selection": "hard validation correct then all validation correct then smaller asset arrays",
                "aggregations": ["max", "mean"], "number_of_fits": 8, "split_access": ["train", "validation"], "threads": 4}
    write_json(OUT / "preregistered_grid.json", protocol)
    gradient_checks = check_gradients()
    write_json(OUT / "gradient_checks.json", gradient_checks)
    tb, train, ty, tv, _ = development("train")
    vb, validation, vy, vv, vh = development("validation")
    eligible = ty.any(axis=1) & ((~ty) & tv).any(axis=1)
    permutation = np.random.default_rng(SEED).permutation(len(train))
    cutoff = int(0.85 * len(permutation))
    fit_all, hold_all = permutation[:cutoff], permutation[cutoff:]
    fit_ids, hold_ids = fit_all[eligible[fit_all]], hold_all[eligible[hold_all]]
    early_stats = fit_transform(train[fit_all], tv[fit_all])
    final_stats = fit_transform(train, tv)
    records, assets, logits = [], {}, {"validation_labels": vy, "validation_hard": vh}
    training_seconds = 0.0
    with threadpool_limits(limits=4):
        for input_name in protocol["inputs"]:
            contextual = input_name == "raw_contextual"
            early_x = transform(train[fit_ids], tv[fit_ids], early_stats, contextual)
            heldout_x = transform(train[hold_ids], tv[hold_ids], early_stats, contextual)
            final_x = transform(train[eligible], tv[eligible], final_stats, contextual)
            val_x = transform(validation, vv, final_stats, contextual)
            for objective in protocol["objectives"]:
                for l2 in protocol["l2_grid"]:
                    key = f"{input_name}_{objective}_H128_L2{l2:g}"
                    tick = time.perf_counter()
                    _, curve, epoch = fit(early_x, ty[fit_ids], tv[fit_ids], objective=objective, l2=l2,
                                          heldout=(heldout_x, ty[hold_ids], tv[hold_ids]))
                    early_seconds = time.perf_counter() - tick
                    tick = time.perf_counter()
                    model, refit_curve, _ = fit(final_x, ty[eligible], tv[eligible], objective=objective, l2=l2, epochs=epoch)
                    refit_seconds = time.perf_counter() - tick
                    training_seconds += early_seconds + refit_seconds
                    asset = export_model(model, final_stats, contextual)
                    assets[key] = asset
                    path = OUT / f"{key}.npz"
                    np.savez(path, **asset)
                    inference_start = time.perf_counter()
                    values = exported_scores(validation, vv, asset)
                    inference_seconds = time.perf_counter() - inference_start
                    reference = prediction(val_x, model)
                    export_error = float(np.max(np.abs(values - reference)))
                    assert np.allclose(values, reference, atol=2e-5, rtol=1e-5), export_error
                    logits[key] = values
                    for aggregation in protocol["aggregations"]:
                        record = {"key": key, "input": input_name, "objective": objective, "hidden": 128,
                                  "l2": l2, "seed": SEED, "selected_epoch": epoch,
                                  "early_stop_epochs_executed": len(curve),
                                  "early_stop_best_loss": curve[epoch - 1]["heldout_loss"],
                                  "early_stop_seconds": early_seconds, "full_refit_seconds": refit_seconds,
                                  "validation_scorer_seconds": inference_seconds,
                                  "checkpoint_bytes": path.stat().st_size,
                                  "asset_uncompressed_array_bytes": sum(a.nbytes for a in asset.values()),
                                  "export_max_absolute_error": export_error,
                                  **evaluate(vb, values, vy, vh, aggregation)}
                        records.append(record)
                        print(json.dumps(record), flush=True)
                    write_json(OUT / f"{key}_curves.json", {"early_stop": curve, "full_refit": refit_curve})
                    write_json(OUT / "results.json", records)
                    np.savez(OUT / "validation_logits.npz", **logits)
    best = max(records, key=lambda r: (r["hard_correct"], r["correct"], -r["asset_uncompressed_array_bytes"]))
    np.savez(OUT / "best_advanced.npz", **assets[best["key"]])
    summary = {"protocol": protocol, "best": best, "total_training_seconds": training_seconds,
               "total_seconds": time.perf_counter() - started, "fit_questions": len(fit_all),
               "heldout_questions": len(hold_all), "fit_eligible": len(fit_ids), "heldout_eligible": len(hold_ids),
               "full_refit_eligible": int(eligible.sum()),
               "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "fixed_parser_sha256": hashlib.sha256((Path(__file__).resolve().parent / "common.py").read_bytes()).hexdigest(),
               "finite_difference_max_error": max(r["absolute_error"] for r in gradient_checks)}
    write_json(OUT / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
