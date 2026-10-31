"""Small nonlinear pairwise scorer; training/validation access only.

The finite architectural grid is declared before fitting. Early stopping uses a
held-out QUESTION split inside training, followed by a full-training refit.
"""
import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "4"

import json
from pathlib import Path
import time

import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

from common import DATA, load_inputs, parse_answer, read_jsonl

OUT = Path(__file__).resolve().parent / "runs" / "nonlinear_experiments"
SEED = 7506


def supervision(bank, split):
    if split not in ("train", "validation"):
        raise ValueError("Experiments may only open train/validation")
    refs = read_jsonl(DATA / f"gsm8k_{split}.jsonl")
    assert len(bank) == len(refs)
    answers, labels = [], []
    for row, ref in zip(bank, refs):
        assert row["id"] == ref["id"] and row["question"] == ref["question"]
        aa = [parse_answer(c["text"]) for c in row["candidates"]]
        gold = parse_answer(ref["answer"])
        answers.append(aa)
        labels.append([int(a is not None and a == gold) for a in aa])
    labels = np.asarray(labels, dtype=np.float32)
    valid = np.array([[a is not None for a in aa] for aa in answers], dtype=np.float32)
    # Axis 1 is the correct candidate, axis 2 the incorrect valid candidate.
    pair_mask = labels[:, :, None] * ((1 - labels) * valid)[:, None, :]
    count = pair_mask.sum(axis=(1, 2))
    weights = pair_mask / np.maximum(count[:, None, None], 1)
    hard = np.array([len({a for a in aa if a is not None}) >= 2 for aa in answers])
    return answers, labels, weights, count > 0, hard


def initialize(hidden):
    rng = np.random.default_rng(SEED)
    return [rng.normal(0, 1 / np.sqrt(1024), (1024, hidden)).astype(np.float32),
            np.zeros(hidden, dtype=np.float32),
            rng.normal(0, 1 / np.sqrt(hidden), hidden).astype(np.float32)]


def scores(x, model):
    w1, b1, w2 = model
    return np.tanh(x @ w1 + b1) @ w2


def loss_gradient(x, weights, model, l2, gradient=True):
    w1, b1, w2 = model
    hidden = np.tanh(x @ w1 + b1)
    output = hidden @ w2
    difference = output[:, None, :] - output[:, :, None]
    loss = float(np.sum(np.logaddexp(0, difference) * weights) / len(x))
    if not gradient:
        return loss
    probabilities = expit(difference) * weights / len(x)
    output_gradient = probabilities.sum(axis=1) - probabilities.sum(axis=2)
    hidden_gradient = output_gradient[:, :, None] * w2 * (1 - hidden * hidden)
    gradients = [x.reshape(-1, 1024).T @ hidden_gradient.reshape(-1, len(w2)) + l2 * w1,
                 hidden_gradient.sum(axis=(0, 1)),
                 hidden.reshape(-1, len(w2)).T @ output_gradient.ravel() + l2 * w2]
    return loss, gradients


def fit(x, weights, hidden, l2, epochs, validation=None, patience=8):
    model = initialize(hidden)
    first = [np.zeros_like(a) for a in model]
    second = [np.zeros_like(a) for a in model]
    rng = np.random.default_rng(SEED + 1)
    updates = 0
    best_loss, best_epoch, stale = float("inf"), 0, 0
    curve = []
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(x))
        batch_losses = []
        for start in range(0, len(x), 128):
            idx = order[start:start + 128]
            loss, grads = loss_gradient(x[idx], weights[idx], model, l2)
            batch_losses.append(loss)
            updates += 1
            for j, gradient in enumerate(grads):
                first[j] *= 0.9
                first[j] += 0.1 * gradient
                second[j] *= 0.999
                second[j] += 0.001 * gradient * gradient
                model[j] -= 0.003 * (first[j] / (1 - 0.9 ** updates)) / (np.sqrt(second[j] / (1 - 0.999 ** updates)) + 1e-8)
        record = {"epoch": epoch, "train_batch_mean_pair_loss": float(np.mean(batch_losses))}
        if validation is not None:
            heldout_loss = loss_gradient(validation[0], validation[1], model, l2, gradient=False)
            record["heldout_question_pair_loss"] = heldout_loss
            if heldout_loss < best_loss - 1e-6:
                best_loss, best_epoch, stale = heldout_loss, epoch, 0
            else:
                stale += 1
        curve.append(record)
        if validation is not None and stale >= patience:
            break
    return model, curve, best_epoch if validation is not None else epochs


def evaluate(logits, answers, labels, hard, aggregation):
    choices = []
    for row, parsed in zip(logits, answers):
        groups = {}
        for j, answer in enumerate(parsed):
            if answer is not None:
                groups.setdefault(answer, []).append(j)
        if not groups:
            choices.append(int(np.argmax(row)))
            continue
        groups = list(groups.values())
        group_scores = [float(np.max(row[g]) if aggregation == "max" else np.mean(row[g])) for g in groups]
        group = groups[int(np.argmax(group_scores))]
        choices.append(group[int(np.argmax(row[group]))])
    correct = labels[np.arange(len(labels)), choices].astype(bool)
    return {"aggregation": aggregation, "n": len(labels), "correct": int(correct.sum()),
            "accuracy": float(100 * correct.mean()), "hard_n": int(hard.sum()),
            "hard_correct": int(correct[hard].sum()), "hard_accuracy": float(100 * correct[hard].mean())}


def main():
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {"seed": SEED, "hidden_grid": [16, 64], "l2_grid": [0.001, 0.01],
                "objective": "equal-question-weight correct-versus-incorrect valid-candidate pairwise logistic",
                "activation": "tanh", "optimizer": "Adam", "learning_rate": 0.003,
                "batch_questions": 128, "max_epochs": 80, "patience": 8,
                "early_stopping": "85/15 split of training QUESTIONS; heldout mean pairwise logistic loss; min improvement 1e-6",
                "normalizer": "early-stop fit normalizer from fit questions only; final refit normalizer from all training questions",
                "full_refit": "fresh same-seed initialization on all training questions at selected epoch count",
                "selection": "hard validation accuracy, then complete validation accuracy, then smallest assets",
                "aggregations": ["max", "mean"], "threads": 4, "split_access": ["train", "validation"]}
    (OUT / "preregistered_grid.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    train_bank, train_arrays = load_inputs("train")
    val_bank, val_arrays = load_inputs("validation")
    _, train_y, train_weights, eligible, _ = supervision(train_bank, "train")
    val_answers, val_y, _, _, val_hard = supervision(val_bank, "validation")
    raw_train = train_arrays["embeddings"]
    raw_val = val_arrays["embeddings"]
    permutation = np.random.default_rng(SEED).permutation(len(train_bank))
    cutoff = int(0.85 * len(permutation))
    fit_ids, heldout_ids = permutation[:cutoff], permutation[cutoff:]
    preliminary_mean = raw_train[fit_ids].reshape(-1, 1024).mean(axis=0, dtype=np.float64).astype(np.float32)
    preliminary_scale = np.maximum(raw_train[fit_ids].reshape(-1, 1024).std(axis=0, dtype=np.float64), 1e-8).astype(np.float32)
    fit_eligible = fit_ids[eligible[fit_ids]]
    heldout_eligible = heldout_ids[eligible[heldout_ids]]
    preliminary_x = (raw_train[fit_eligible] - preliminary_mean) / preliminary_scale
    heldout_x = (raw_train[heldout_eligible] - preliminary_mean) / preliminary_scale
    train_mean = raw_train.reshape(-1, 1024).mean(axis=0, dtype=np.float64).astype(np.float32)
    train_scale = np.maximum(raw_train.reshape(-1, 1024).std(axis=0, dtype=np.float64), 1e-8).astype(np.float32)
    full_x = (raw_train[eligible] - train_mean) / train_scale
    outputs, all_logits, checkpoints = [], {"validation_labels": val_y, "validation_hard": val_hard}, {}
    with threadpool_limits(limits=4):
        for hidden in protocol["hidden_grid"]:
            for l2 in protocol["l2_grid"]:
                key = f"pairwise_tanh_H{hidden}_L2{l2:g}"
                fit_start = time.perf_counter()
                _, curve, epoch = fit(preliminary_x, train_weights[fit_eligible], hidden, l2, 80,
                                      validation=(heldout_x, train_weights[heldout_eligible]))
                preliminary_seconds = time.perf_counter() - fit_start
                refit_start = time.perf_counter()
                model, refit_curve, _ = fit(full_x, train_weights[eligible], hidden, l2, epoch)
                refit_seconds = time.perf_counter() - refit_start
                raw_w1 = model[0] / train_scale[:, None]
                raw_b1 = model[1] - train_mean @ raw_w1
                checkpoint = {"w1": raw_w1, "b1": raw_b1, "w2": model[2],
                              "train_mean": train_mean, "train_scale": train_scale}
                checkpoints[key] = checkpoint
                np.savez(OUT / f"{key}.npz", **checkpoint)
                all_logits[key] = scores(raw_val, [raw_w1, raw_b1, model[2]])
                for aggregation in protocol["aggregations"]:
                    record = {"key": key, "hidden": hidden, "l2": l2, "selected_epoch": epoch,
                              "early_stop_epochs_executed": len(curve),
                              "early_stop_best_loss": curve[epoch - 1]["heldout_question_pair_loss"],
                              "early_stop_seconds": preliminary_seconds, "refit_seconds": refit_seconds,
                              "asset_uncompressed_array_bytes": sum(a.nbytes for a in checkpoint.values()),
                              **evaluate(all_logits[key], val_answers, val_y, val_hard, aggregation)}
                    outputs.append(record)
                    print(json.dumps(record), flush=True)
                (OUT / f"{key}_curves.json").write_text(json.dumps({"early_stop": curve, "full_refit": refit_curve}, indent=2), encoding="utf-8")
                (OUT / "results.json").write_text(json.dumps(outputs, indent=2), encoding="utf-8")
                np.savez(OUT / "validation_logits.npz", **all_logits)
    best = max(outputs, key=lambda r: (r["hard_correct"], r["correct"], -r["asset_uncompressed_array_bytes"]))
    np.savez(OUT / "best_nonlinear.npz", **checkpoints[best["key"]])
    summary = {"protocol": protocol, "best": best, "total_seconds": time.perf_counter() - start,
               "fit_training_questions": len(fit_ids), "early_stop_heldout_questions": len(heldout_ids),
               "fit_eligible_questions": len(fit_eligible), "heldout_eligible_questions": len(heldout_eligible),
               "full_refit_eligible_questions": int(eligible.sum()),
               "candidate_rows_never_split_independently": True,
               "comparison_linear_hard_correct": 289, "comparison_linear_hard_n": 355,
               "comparison_existing_fusion_hard_correct": 304}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
