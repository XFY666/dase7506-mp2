"""Fixed D/T/F/G recipes on two new training-only three-fold partitions.

This is a stability study, not a new independent dataset or a selection test.
All 48 learners are fresh fits; no validation/test stage is implemented.
"""
from __future__ import annotations

import os
import sys
sys.dont_write_bytecode = True
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "4"

import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
import platform
import time
import zipfile

import numpy as np
import sklearn
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

from advanced_neural import development
from common import DATA
from ensemble import head_scores
from experiment_utils import dump, metrics
from finalize_validation import score_scale
from selection import select
from train_selected import fit_head
import selective_routing as routing

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "optimization_baselines"
FOLD_SEEDS = (20261027, 20261028)
HEADS = ("d_pair", "d_list", "d_text", "A", "K", "Q", "P", "C")
MODELS = ("D", "T", "F", "G")
STARTED = None


def progress(stage, count, detail):
    payload = {"stage": stage, "completed_fits": count, "planned_fits": 48, "detail": detail,
               "elapsed_seconds": time.perf_counter() - STARTED if STARTED is not None else 0.,
               "updated_utc": datetime.now(timezone.utc).isoformat()}
    path = OUT / "progress.json"
    temporary = path.with_suffix(".json.tmp")
    dump(temporary, payload)
    os.replace(temporary, path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def asset_bytes(path):
    with zipfile.ZipFile(path) as archive:
        return sum(item.file_size for item in archive.infolist())


def mixture_scores(normalized):
    # Preserve the historical D/T float32 fusion arithmetic; the five-base
    # routing helper already preserves each source's normalization dtype.
    original = normalized[..., :3].astype(np.float32)
    d = original.mean(axis=-1)
    t = original[..., [0, 2]].mean(axis=-1)
    f = .5 * normalized[..., 3] + .25 * normalized[..., 4] + .25 * normalized[..., 5]
    g = .25 * f + .5 * normalized[..., 6] + .25 * normalized[..., 7]
    return np.stack((d, t, f, g), axis=-1)


def group_losses(data, scores):
    grouped = np.einsum("ngc,nc->ng", data["weights"].astype(np.float64), scores)
    logden = logsumexp(np.where(data["group_valid"], grouped, -np.inf), axis=1)
    return logden - (grouped * data["good"]).sum(axis=1)


def protected_io():
    reads = set()
    root = OUT.resolve()
    def audit(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path.name.startswith(("gsm8k_validation", "gsm8k_test")):
            raise RuntimeError(f"Forbidden split access: {path}")
        if DATA.resolve() in path.parents:
            reads.add(path.relative_to(DATA.resolve()).as_posix())
        mode = args[1]
        flags = args[2] or 0
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or bool(flags & (os.O_WRONLY | os.O_RDWR))
        if writing and root != path and root not in path.parents:
            # Libraries do not need persistent writes for these CPU fits.
            raise RuntimeError(f"Write outside owned experiment directory: {path}")
    sys.addaudithook(audit)
    return reads


def source_closure():
    paths = {Path(__file__).resolve(), ROOT / "runs/model_bundle/config.json"}
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path:
            path = Path(path).resolve()
            if path.suffix == ".py" and path.parent == ROOT:
                paths.add(path)
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths)}


def main():
    global STARTED
    if OUT.exists() and any(p.name != "attempt0_boolean_labels" for p in OUT.iterdir()):
        raise FileExistsError("Preserve the registered run; use a separately reviewed new experiment for another run")
    OUT.mkdir(parents=True, exist_ok=True)
    accessed = protected_io()
    routing.OUT = OUT
    started = time.perf_counter()
    STARTED = started
    progress("preparing", 0, "Registering fixed recipes and source hashes")
    def routing_dump(path, value):
        dump(path, value)
        if Path(path) == OUT / "fit_records.json":
            last = value[-1]
            progress("fitting", len(value), f"{last['partition']}: fitted {last['head']}")
    routing.dump = routing_dump
    config = json.loads((ROOT / "runs/model_bundle/config.json").read_text(encoding="utf-8"))
    assert len(config["heads"]) == 3
    assert [head["kind"] for head in config["heads"]] == ["nonlinear", "residual", "trees"]
    source_hashes = source_closure()
    protocol = {
        "fold_seeds": FOLD_SEEDS, "fold_mapping": "fold[rng.permutation(N)] = arange(N) % 3; all4755questions",
        "head_order": HEADS, "model_order": MODELS, "heads_D": config["heads"],
        "five_base_recipes": {"A": {**routing.GROUP_CONFIG, "epochs": 14},
            "K": {"pca": 64, "C": 3., "gamma_multiplier": 1., "objective": "pair_delta"},
            "Q": {"PLS": 8, "HGBT_leaves": 15, "rounds": 300},
            "P": {**routing.PAIR_CONFIG, "epochs": 27},
            "C": {"PLS": 8, "HGBT_leaves": 31, "rounds": 300, "feature_design": "condition_pls8"}},
        "mixtures": {"D": "mean(d_pair,d_list,d_text)", "T": "mean(d_pair,d_text)",
                     "F": ".5A+.25K+.25Q", "G": ".25F+.5P+.25C"},
        "transform_seeds": "fold_seed + fold_index; fixed learner random seeds preserved in source recipes",
        "normalization": "fit-side score scales only; D/T normalize to float32 before averaging to match historical recipe",
        "question_grouping": "All four candidates travel together; learned statistics/transforms/scales see outer fit only",
        "planned_model_fits": 48, "planned_PCA_fits": 12, "planned_PLS_fits": 12,
        "selection": "none; report all four fixed models, both repetitions and all paired comparisons",
        "statistical_scope": "Same questions repeated; no doubled sample size, no naive pooled McNemar significance, no noninferiority claim",
        "validation_access": False, "test_access": False, "threads": 4,
        "historical_protocols": "Unchanged; this prospectively registered stability study does not retroactively pass old gates",
        "implementation_repair": "Preserved initial invalid boolean-label attempt; fixed float32 labels for PLS helper before any OOF results. All 48 learners freshly refitted with unchanged recipes/seeds.",
        "source_sha256": source_hashes,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__, "platform": platform.platform()},
    }
    dump(OUT / "protocol.json", protocol)
    dump(OUT / "source_evidence.json", {"source_sha256": source_hashes,
        "training_data_sha256": {str(p.relative_to(ROOT)): digest(p) for p in
             (DATA / "gsm8k_train.jsonl", DATA / "candidates/gsm8k_train.jsonl", DATA / "candidates/gsm8k_train.npz")}})
    bank, emb, labels, valid, hard = development("train")
    assert len(bank) == 4755 and hard.sum() == 2114
    with threadpool_limits(limits=4):
        data, preparation = routing.prepare(bank, emb, labels.astype(np.float32), hard)
        assert np.array_equal(valid, data["valid"])
        from quantitative_model import stable_center
        assert data["labels"].dtype == np.float32
        centered_target = stable_center(data["labels"][hard], valid[hard])[valid[hard]]
        assert np.std(centered_target) > 0 and np.any(centered_target < 0) and np.any(centered_target > 0)
        dump(OUT / "preparation.json", preparation)
        progress("fitting", 0, "Training-only deterministic features prepared; starting 6 outer partitions")
        folds = np.stack([routing.folds_for(len(bank), seed) for seed in FOLD_SEEDS])
        raw = np.full((2, len(bank), 4, len(HEADS)), np.nan, dtype=np.float64)
        normalized = np.full_like(raw, np.nan)
        scales = np.full((2, 3, len(HEADS)), np.nan)
        records, transforms = [], []
        for repeat, fold_seed in enumerate(FOLD_SEEDS):
            for fold in range(3):
                fit_ids = np.flatnonzero(folds[repeat] != fold)
                held_ids = np.flatnonzero(folds[repeat] == fold)
                directory = OUT / f"repeat{repeat}_fold{fold}"
                directory.mkdir()
                fit_bank = [bank[int(i)] for i in fit_ids]
                held_bank = [bank[int(i)] for i in held_ids]
                y, v = labels[fit_ids], valid[fit_ids]
                pairs = y[:, :, None] * ((~y) & v)[:, None, :]
                counts = pairs.sum(axis=(1, 2))
                pair_weight = (pairs / np.maximum(counts[:, None, None], 1)).astype(np.float32)
                for column, head in enumerate(config["heads"]):
                    tick = time.perf_counter()
                    asset = fit_head(head["kind"], head["training"], fit_bank, emb[fit_ids], data["original"][fit_ids], y, v, pair_weight, counts > 0)
                    fit_score = head_scores(head["kind"], fit_bank, emb[fit_ids], data["original"][fit_ids], asset)
                    scale = score_scale(fit_score, fit_bank)
                    held_score = head_scores(head["kind"], held_bank, emb[held_ids], data["original"][held_ids], asset)
                    raw[repeat, held_ids, :, column] = held_score
                    normalized[repeat, held_ids, :, column] = (held_score / scale).astype(np.float32)
                    scales[repeat, fold, column] = scale
                    path = directory / f"{HEADS[column]}.npz"
                    np.savez(path, **asset)
                    row = {"partition": directory.name, "repeat": repeat, "fold": fold, "head": HEADS[column],
                           "kind": head["kind"], "recipe": head["training"], "score_scale": scale,
                           "fit_questions": len(fit_ids), "heldout_questions": len(held_ids),
                           "fit_indices_sha256": array_digest(fit_ids.astype(np.int64)),
                           "held_indices_sha256": array_digest(held_ids.astype(np.int64)),
                           "asset_sha256": digest(path), "asset_uncompressed_bytes": asset_bytes(path),
                           "total_seconds": time.perf_counter() - tick}
                    records.append(row)
                    dump(OUT / "fit_records.json", records)
                    progress("fitting", len(records), f"{directory.name}: fitted {HEADS[column]}")
                    print(json.dumps({"partition": directory.name, "head": HEADS[column], "completed_learner_fits": len(records), "seconds": row["total_seconds"]}), flush=True)
                xraw, xnorm = routing.fit_partition(data, fit_ids, held_ids, fold_seed + fold, directory, records, transforms)
                for projected in ("Q", "C"):
                    with np.load(directory / f"{projected}.npz", allow_pickle=False) as asset:
                        assert np.any(asset["pls_rotations"] != 0), "PLS export must retain a nonzero learned projection"
                raw[repeat, held_ids, :, 3:] = xraw
                normalized[repeat, held_ids, :, 3:] = xnorm
                scales[repeat, fold, 3:] = [r["score_scale"] for r in records[-5:]]
                # Keep one complete record of all eight heads for this partition.
                dump(directory / "all_heads.json", {"repeat": repeat, "fold": fold, "fold_seed": fold_seed,
                    "head_order": HEADS, "learners": records[-8:], "score_scales": scales[repeat, fold].tolist()})
                np.savez(directory / "all_scores.npz", held_indices=held_ids, names=np.array(HEADS),
                         raw=raw[repeat, held_ids], normalized=normalized[repeat, held_ids],
                         score_scales=scales[repeat, fold])
    assert len(records) == 48 and len(transforms) == 24
    progress("summarizing", len(records), "48 learner fits complete; writing paired OOF metrics and assets")
    assert np.isfinite(raw).all() and np.isfinite(normalized).all()
    mixtures = mixture_scores(normalized)
    choices = np.empty((2, len(bank), len(MODELS)), np.int8)
    losses = np.empty((2, len(bank), len(MODELS)), np.float64)
    repeat_results, paired_results = [], []
    for repeat, fold_seed in enumerate(FOLD_SEEDS):
        results = {}
        for column, name in enumerate(MODELS):
            score = mixtures[repeat, :, :, column]
            choices[repeat, :, column] = select(bank, score, "mean")
            losses[repeat, :, column] = group_losses(data, score)
            result = metrics(choices[repeat, :, column], labels, hard)
            result["group_ce"] = float(losses[repeat, hard, column].mean())
            result["folds"] = []
            for fold in range(3):
                mask = hard & (folds[repeat] == fold)
                correct = labels[np.arange(len(bank)), choices[repeat, :, column]]
                result["folds"].append({"fold": fold, "hard_n": int(mask.sum()), "hard_correct": int(correct[mask].sum()),
                                        "group_ce": float(losses[repeat, mask, column].mean())})
            results[name] = result
        repeat_results.append({"repeat": repeat, "fold_seed": fold_seed, "results": results})
        for ref, cand in itertools.combinations(range(len(MODELS)), 2):
            paired_results.append({"repeat": repeat, "candidate": MODELS[cand], "reference": MODELS[ref],
                "paired": routing.paired(choices[repeat, :, cand], choices[repeat, :, ref], labels, hard, folds[repeat]),
                "group_ce_difference": results[MODELS[cand]]["group_ce"] - results[MODELS[ref]]["group_ce"]})
    np.savez(OUT / "oof_logits.npz", ids=np.array([r["id"] for r in bank]),
        raw=raw, normalized=normalized, normalized_heads=normalized, head_names=np.array(HEADS),
        mixture_scores=mixtures, model_names=np.array(MODELS), choices=choices, group_losses=losses,
        labels=labels, valid=valid, hard=hard, folds=folds, fold_seeds=np.array(FOLD_SEEDS), score_scales=scales)
    dump(OUT / "oof_schema.json", {
        "raw": "[repeat=2,question=4755,candidate=4,head=8] float64", "normalized": "same shape; fitting-side scales",
        "normalized_heads": "alias of normalized", "head_names": HEADS, "model_names": MODELS,
        "mixture_scores": "[2,4755,4,4] float64", "choices": "[2,4755,4] int8",
        "group_losses": "[2,4755,4] float64; report hard subset CE", "folds": "[2,4755] int8",
        "score_scales": "[repeat=2,fold=3,head=8] float64", "ids": "4755 alignment-only IDs",
        "labels_valid": "[4755,4] bool", "hard": "[4755] bool", "mixture_formula": protocol["mixtures"]})
    for relative, expected in source_hashes.items():
        assert digest(ROOT / relative) == expected, relative
    summary = {
        "status": "complete", "repeats": repeat_results, "paired_comparisons": paired_results,
        "model_fit_calls": len(records), "PCA_fit_calls": sum(r["kind"] == "PCA" for r in transforms),
        "PLS_fit_calls": sum(r["kind"] == "PLS" for r in transforms),
        "training_questions": len(bank), "hard_questions": int(hard.sum()), "fold_seeds": FOLD_SEEDS,
        "models": MODELS, "validation_accessed": False, "test_accessed": False,
        "data_files_read": sorted(accessed), "source_hashes_unchanged": True,
        "seconds": time.perf_counter() - started,
        "preserved_invalid_attempt": json.loads((OUT / "attempt0_boolean_labels/incident.json").read_text()) if (OUT / "attempt0_boolean_labels/incident.json").exists() else None,
        "asset_bytes": {name: [sum(r["asset_uncompressed_bytes"] for r in records
                if r["partition"] == f"repeat{repeat}_fold{fold}" and r["head"] in members)
            for repeat in range(2) for fold in range(3)] for name, members in {
                "D": HEADS[:3], "T": (HEADS[0], HEADS[2]), "F": ("A", "K", "Q"), "G": ("A", "K", "Q", "P", "C")}.items()},
        "asset_scope": "Per outer partition sum of uncompressed NPZ members, excluding config; not a final full-training deployment size",
        "caveat": "Historically chosen fixed recipes; repetitions reuse the same questions. Descriptive stability evidence, not independent duplicated sample size or unbiased search evaluation. No model promotion."}
    dump(OUT / "summary.json", summary)
    progress("complete", len(records), "All 48 fits and OOF readouts complete; no validation/test access")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if STARTED is not None:
            records_path = OUT / "fit_records.json"
            count = len(json.loads(records_path.read_text())) if records_path.exists() else 0
            progress("failed", count, f"{type(error).__name__}: {error}")
        raise
