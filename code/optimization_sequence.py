"""Four matched sequence-input experiments on two repeated training-only CVs."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "4"

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

import sequence_experiment as base
import sequence_model as original
import optimization_sequence_model as variants
from experiment_utils import development_data, dump, metrics
from selection import select

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs/optimization_sequence"
SEEDS = (20261027, 20261028)
RECIPES = variants.RECIPES
PLANNED_FITS = len(SEEDS) * 3 * len(RECIPES) * 2


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ahash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def atomic_dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def progress(stage, completed, detail, started):
    item = dict(stage=stage, completed_fits=completed, planned_fits=PLANNED_FITS,
                detail=detail, elapsed_seconds=time.perf_counter()-started,
                updated_utc=datetime.now(timezone.utc).isoformat())
    atomic_dump(OUT / "progress.json", item)
    print(json.dumps(item), flush=True)


def protocol():
    return {
        "purpose": "E1 and G2 input-token matched comparisons; no validation or test stage",
        "recipes": list(RECIPES), "fold_seeds": list(SEEDS), "folds": 3,
        "fold_assignment": "fold[default_rng(seed).permutation(N)] = arange(N) % 3",
        "inner_split": "85/15 of outer fitting questions, default_rng(outer_seed+100+fold)",
        "initialization_seed": 7506, "max_epochs": 24, "patience": 4,
        "minimum_early_improvement": 1e-4, "learners": "historical joint WordCNN, unchanged architecture/optimizer/objective",
        "architecture": "random token embedding32; ReLU Conv3/5 each32; mean/max pools; concatenate1024+75 numeric inputs; tanh64; scalar",
        "optimizer": "AdamW lr.001 weight_decay.001 gradient_clip5; batch64",
        "objective": "question-balanced answer-group-mean cross entropy on eligible fitting questions",
        "head64": "original first64 question tokens and <num> numeric replacement",
        "headtail": "first32+last32 question tokens if longer than64; original <num> replacement",
        "headtail_roles": "same head/tail; one combined numeric token tracks first-occurrence question number q0..q31/qx or derived, own-answer and different-rival-answer membership; combined flags preserve collisions",
        "role_numerics": "Exact Fraction for unsigned literals, simple a/b fractions and numeric LaTeX fractions; unary signs remain separate tokens; no reference answers are read by tokenizer; derived means not a question literal, not a correctness claim",
        "headtail_format": "same head/tail and <num>; remove markup dollars/backticks/bold/headings and text style wrappers; no mathematical operator rewriting",
        "candidate_truncation": "unchanged first128 + gap + last128 if over256; sequence322",
        "vocabulary": "max4096, fitted only on fitting-side token frequencies; role tokens fixed a priori; no external data or pretrained weights",
        "numeric_inputs": "unchanged1024 provided embeddings+historical stable75 features; valid fitting candidates determine means/stds; clipping[-8,8]",
        "normalizers_and_stopping": "inner-fit vocab/statistics for early fit, fresh outer-fit vocab/statistics and fresh model for outer refit; heldout labels never in model training/transforms",
        "score_scale": "within-question-centered raw scores on valid outer-fitting candidates only",
        "screen_selection": "highest average correct across two repeated OOF evaluations; lower mean normalized groupCE; recipe order",
        "uncertainty": "Same2114 hard questions repeated; never treat4228 as independent. This is exploratory selection, not unbiased full-search generalization.",
        "max_model_fits": PLANNED_FITS, "full_training_refit": "not in this stage",
        "later_E2_C2": "not executed; requires separate preregistration and train-reference-only auxiliary labels",
        "threads": 4, "source_files_read_only": True,
        "cost_probe": "Record first completed recipe cost projection; hard stop at3600seconds elapsed before starting another fit; unfinished screen cannot select a winner",
    }


def self_checks():
    q = " ".join(f"word{i}" for i in range(80)) + " How many red apples remain?"
    bank = [{"id": "unused", "question": q, "candidates": [
        {"index": j, "text": t} for j, t in enumerate([
            "There are 5 apples. 5 - 2 = 3. #### 3", "5 + 2 = 7. #### 7",
            "\\frac{1}{2} + 2.5 = 3. #### 3", "The number is 2. #### 2"])]}]
    original_sequences = list(original.question_candidate_tokens(bank[0]))
    assert original_sequences == list(variants.question_candidate_tokens(bank[0], "head64"))
    changed = copy.deepcopy(bank)
    perm = [2, 0, 3, 1]
    for row in changed:
        row["id"] = "changed"
        row["candidates"] = [row["candidates"][j] for j in perm]
        for c in row["candidates"]:
            c["index"] = 999
    for recipe in RECIPES:
        one = variants.sequences(bank, recipe)
        other = variants.sequences(changed, recipe)
        assert other[0] == [one[0][j] for j in perm]
        assert max(map(len, one[0])) <= 322
        if recipe != "head64":
            assert "remain" in one[0][0][:64]
        vocab = variants.vocabulary(one, [0], recipe)
        encoded, _ = variants.encode_sequences(one, vocab)
        assert encoded.shape == (1, 4, 322)
        assert variants.encode_sequences([], vocab)[0].shape == (0, 4, 322)
    assert variants.number("1/2") == variants.number("0.5")
    token = variants.role_sequence(["3"], {variants.number("3"): 0}, variants.number("3"), {variants.number("3")})
    assert token == ["<num:q0:self:rival>"]
    return {"baseline_tokens_exact": True, "question_tail_present": True,
            "candidate_permutation_and_id_independence": True, "empty_shape": True,
            "fraction_equivalence": True, "combined_role_collision": True}


def prepare(sequence_bank, recipe, raw, valid, fit_ids):
    tick = time.perf_counter()
    vocab = variants.vocabulary(sequence_bank, fit_ids, recipe)
    codes, info = variants.encode_sequences(sequence_bank, vocab)
    values = raw[fit_ids][valid[fit_ids]].astype(np.float64)
    transform = {"vocabulary": vocab, "numeric_mean": values.mean(0).astype(np.float32),
                 "numeric_scale": np.maximum(values.std(0), 1e-3).astype(np.float32)}
    numeric = original.transform_numeric(raw, transform)
    return codes, numeric, transform, {
        "fit_questions": len(fit_ids), "vocabulary_size": len(vocab),
        "fit_indices_sha256": ahash(fit_ids), "seconds": time.perf_counter()-tick,
        "encoding_diagnostic": info}


def fit_one(bank, sequence_bank, recipe, raw, labels, valid, eligible, structure,
            fit_ids, held_ids, inner_seed, stem, completed, started):
    if time.perf_counter()-started > 3600:
        raise RuntimeError("Recorded one-hour stage bound; no incomplete-screen selection")
    order = np.random.default_rng(inner_seed).permutation(fit_ids)
    cut = int(.85 * len(order))
    inner, early_held = order[:cut], order[cut:]
    assert not np.intersect1d(fit_ids, held_ids).size
    np.savez_compressed(OUT / f"{stem}_indices.npz", fit=fit_ids, held=held_ids,
                        inner=inner, early_held=early_held)
    progress("early_fit", completed, stem, started)
    codes, numeric, transform, info = prepare(sequence_bank, recipe, raw, valid, inner)
    ct, nt, st = base.tensor_inputs(codes, numeric, structure)
    model, curve, epochs = base.train(ct, nt, st, inner[eligible[inner]], True, 24,
                                     early_held[eligible[early_held]])
    completed += 1
    dump(OUT / f"{stem}_early_curve.json", curve)
    early_seconds = curve[-1]["elapsed_seconds"]
    progress("outer_refit", completed, f"{stem}; selected_epochs={epochs}", started)
    codes, numeric, transform, outer = prepare(sequence_bank, recipe, raw, valid, fit_ids)
    ct, nt, st = base.tensor_inputs(codes, numeric, structure)
    model, curve, _ = base.train(ct, nt, st, fit_ids[eligible[fit_ids]], True, epochs)
    completed += 1
    dump(OUT / f"{stem}_refit_curve.json", curve)
    z = base.native_scores(model, ct, nt)
    asset = base.export(model, transform)
    scale = float(max(np.std((z[fit_ids]-z[fit_ids].mean(1, keepdims=True))[valid[fit_ids]]), 1e-6))
    asset.update(score_scale=np.array(scale), token_recipe=np.array(recipe))
    np.savez(OUT / f"{stem}.npz", **asset)
    check = held_ids[:32]
    tick = time.perf_counter()
    replay = original.numpy_forward(codes[check], numeric[check], asset)
    numpy_seconds = time.perf_counter()-tick
    error = float(np.max(np.abs(replay-z[check])))
    assert error < 1e-4
    subset = [bank[i] for i in check]
    assert select(subset, replay, "mean") == select(subset, z[check], "mean")
    rec = {"stem": stem, "recipe": recipe, "selected_epochs": epochs,
           "early_fit_seconds": early_seconds, "refit_seconds": curve[-1]["elapsed_seconds"],
           "inner_transform": info, "outer_transform": outer, "score_scale": scale,
           "fit_indices_sha256": ahash(fit_ids), "held_indices_sha256": ahash(held_ids),
           "asset_bytes": (OUT / f"{stem}.npz").stat().st_size,
           "asset_sha256": digest(OUT / f"{stem}.npz"),
           "numpy_32_question_seconds": numpy_seconds, "numpy_export_max_abs_error": error}
    dump(OUT / f"{stem}_record.json", rec)
    progress("outer_complete", completed, stem, started)
    return z, rec, completed


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Preserve existing recorded study")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    assert torch.version.cuda is None
    checks = self_checks()
    dump(OUT / "input_checks.json", checks)
    hashes = {name: digest(ROOT / name) for name in (
        "optimization_sequence.py", "optimization_sequence_model.py", "sequence_experiment.py",
        "sequence_model.py", "common.py", "text_features.py", "selection.py", "experiment_utils.py")}
    dump(OUT / "protocol.json", protocol())
    dump(OUT / "source_evidence.json", hashes)
    dump(OUT / "environment.json", {"python": platform.python_version(), "numpy": np.__version__,
         "torch": torch.__version__, "threads": torch.get_num_threads(), "platform": platform.platform()})
    progress("prepare_train_only", 0, "load training bank and deterministic features", started)
    bank, embeddings, labels, hard = development_data("train")
    weights, gv, good, valid, eligible = base.group_structure(bank, labels)
    assert np.array_equal(hard, eligible)
    structure = weights, gv, good
    raw_inputs = original.numeric_inputs(bank, embeddings)
    sequence_banks = {recipe: variants.sequences(bank, recipe) for recipe in RECIPES}
    # Confirm the reconstructed baseline matches the original tokenizer on the complete training bank.
    assert all(sequence_banks["head64"][i] == list(original.question_candidate_tokens(row)) for i, row in enumerate(bank))
    folds = np.empty((len(SEEDS), len(bank)), np.int8)
    for r, seed in enumerate(SEEDS):
        folds[r, np.random.default_rng(seed).permutation(len(bank))] = np.arange(len(bank)) % 3
    ids = np.array([row["id"] for row in bank])
    np.savez_compressed(OUT / "fold_assignment.npz", ids=ids, folds=folds, labels=labels, valid=valid, hard=hard)
    raw = np.full((len(SEEDS), len(bank), 4, len(RECIPES)), np.nan, np.float32)
    normalized = raw.copy()
    records = []
    completed = 0
    for r, seed in enumerate(SEEDS):
        for f in range(3):
            fit_ids = np.flatnonzero(folds[r] != f)
            held_ids = np.flatnonzero(folds[r] == f)
            for c, recipe in enumerate(RECIPES):
                stem = f"repeat{r}_fold{f}_{recipe}"
                z, record, completed = fit_one(bank, sequence_banks[recipe], recipe, raw_inputs,
                    labels, valid, eligible, structure, fit_ids, held_ids, seed+100+f, stem,
                    completed, started)
                raw[r, held_ids, :, c] = z[held_ids]
                normalized[r, held_ids, :, c] = z[held_ids]/record["score_scale"]
                record.update(repeat=r, fold=f, metrics=metrics(select([bank[i] for i in held_ids], z[held_ids], "mean"), labels[held_ids], hard[held_ids]))
                records.append(record)
                atomic_dump(OUT / "fit_records.json", records)
                np.savez_compressed(OUT / "partial_oof.npz", ids=ids, raw=raw, normalized=normalized,
                    labels=labels, valid=valid, hard=hard, folds=folds, recipes=np.array(RECIPES))
                if completed == 2:
                    dump(OUT / "cost_probe.json", {"recorded_first_recipe_fits": 2,
                         "first_recipe_fit_seconds": record["early_fit_seconds"]+record["refit_seconds"],
                         "rough_all_fits_seconds": (record["early_fit_seconds"]+record["refit_seconds"])*len(SEEDS)*3*len(RECIPES),
                         "not_a_scope_or_recipe_selection": True})
    assert np.isfinite(raw).all() and np.isfinite(normalized).all()
    rows = []
    for c, recipe in enumerate(RECIPES):
        repeats = []
        for r in range(len(SEEDS)):
            choices = np.asarray(select(bank, raw[r, :, :, c], "mean"))
            baseline_choices = np.asarray(select(bank, raw[r, :, :, 0], "mean"))
            correct = labels[np.arange(len(bank)), choices].astype(bool)
            reference = labels[np.arange(len(bank)), baseline_choices].astype(bool)
            repeats.append({"seed": SEEDS[r], "metrics": metrics(choices, labels, hard),
                "group_ce": base.numpy_group_loss(normalized[r, :, :, c], weights, gv, good, eligible),
                "repairs_vs_head64": int((hard & correct & ~reference).sum()),
                "regressions_vs_head64": int((hard & reference & ~correct).sum())})
        rows.append({"recipe": recipe, "repeats": repeats,
            "mean_hard_correct": float(np.mean([x["metrics"]["hard_correct"] for x in repeats])),
            "mean_group_ce": float(np.mean([x["group_ce"] for x in repeats]))})
    selected = min(rows, key=lambda row: (-row["mean_hard_correct"], row["mean_group_ce"], RECIPES.index(row["recipe"])))
    np.savez_compressed(OUT / "oof_logits.npz", ids=ids, raw=raw, normalized=normalized,
                       labels=labels, valid=valid, hard=hard, folds=folds, recipes=np.array(RECIPES))
    assert all(digest(ROOT / name) == value for name, value in hashes.items())
    summary = {"recipes": rows, "selected": selected, "model_fit_calls": completed,
        "vocabulary_fits": completed, "numeric_normalizer_fits": completed,
        "fold_seeds": list(SEEDS), "hard_questions": int(hard.sum()),
        "total_seconds": time.perf_counter()-started, "validation_evaluations": 0, "test_access": False,
        "max_fold_asset_bytes": max(r["asset_bytes"] for r in records),
        "max_numpy_export_error": max(r["numpy_export_max_abs_error"] for r in records),
        "full_model_fitted": False, "source_sha256": hashes,
        "interpretation": "Repeated observations of the same training questions; exploratory family selection, not independent trials or untouched generalization estimate"}
    dump(OUT / "summary.json", summary)
    progress("complete", completed, f"selected {selected['recipe']}; no validation/test", started)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
