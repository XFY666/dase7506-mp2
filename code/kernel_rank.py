"""Exact RBF candidate and antisymmetric comparison models, selected on train OOF.

Only one training-selected configuration is evaluated on validation. All
normalizers, PCA transforms and bandwidths are refitted inside each question
fold. The selected export uses NumPy and the existing deterministic text parser.
"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "4"
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from common import parse_answer
from selection import select
from text_features import extract_features

SEED = 20261015
OUT = Path(__file__).resolve().parent / "runs/kernel_rank"


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def valid_mask(bank):
    return np.array([[parse_answer(c["text"]) is not None for c in q["candidates"]] for q in bank])


def transformed(embeddings, features, asset):
    z = np.asarray(embeddings, dtype=np.float64) @ asset["projection"] + asset["projection_bias"]
    if len(asset["text_multiplier"]):
        t = np.asarray(features, dtype=np.float64) * asset["text_multiplier"] + asset["text_bias"]
        z = np.concatenate((z, t), axis=2)
    return z


def fit_transform(embeddings, features, seed):
    from sklearn.decomposition import PCA
    raw = embeddings.reshape(-1, 1024).astype(np.float64)
    mean = raw.mean(0)
    std = raw.std(0)
    std = np.where(std > 1e-6, std, 1.)
    normalized = (raw-mean)/std
    pca = PCA(n_components=64, svd_solver="randomized", iterated_power=4,
              random_state=seed).fit(normalized)
    projected = (normalized-pca.mean_) @ pca.components_.T
    pc_std = np.maximum(projected.std(0), 1e-6)
    projection = pca.components_.T / std[:, None] / pc_std[None, :] / 8.
    bias = -(mean @ projection) - (pca.mean_ @ pca.components_.T) / pc_std / 8.
    t = features.reshape(-1, features.shape[-1]).astype(np.float64)
    tmean, tstd = t.mean(0), t.std(0)
    tstd = np.where(tstd > 1e-6, tstd, 1.)
    multiplier = 1 / tstd / np.sqrt(t.shape[1])
    return {"projection": projection, "projection_bias": bias,
            "text_multiplier": multiplier, "text_bias": -tmean*multiplier}


def representation(transform, name):
    result = {k: v.copy() for k, v in transform.items()}
    if name == "pca64":
        result["text_multiplier"] = np.empty(0, dtype=np.float64)
        result["text_bias"] = np.empty(0, dtype=np.float64)
    return result


def design_matrix(x, labels, valid, objective):
    eligible = labels.any(1) & ((~labels) & valid).any(1)
    if objective == "candidate":
        mask = eligible[:, None] & valid
        weights = np.zeros(labels.shape, dtype=np.float64)
        for q in np.flatnonzero(eligible):
            good, bad = labels[q], (~labels[q]) & valid[q]
            weights[q, good] = .5 / good.sum()
            weights[q, bad] = .5 / bad.sum()
        w = weights[mask]
        return np.ascontiguousarray(x[mask]), labels[mask].astype(np.int32), w/w.mean(), int(eligible.sum())
    positive, weights = [], []
    for q in np.flatnonzero(eligible):
        good = np.flatnonzero(labels[q])
        bad = np.flatnonzero((~labels[q]) & valid[q])
        for i in good:
            for j in bad:
                positive.append(x[q, i]-x[q, j])
                weights.append(1/(len(good)*len(bad)))
    delta = np.asarray(positive)
    weights = np.tile(weights, 2).astype(np.float64)
    return (np.ascontiguousarray(np.concatenate((delta, -delta))),
            np.r_[np.ones(len(delta), np.int32), np.zeros(len(delta), np.int32)],
            weights/weights.mean(), int(eligible.sum()))


def base_gamma(x, seed):
    rng = np.random.default_rng(seed)
    i, j = rng.integers(0, len(x), size=(2, 8192))
    distance = np.sum((x[i]-x[j])**2, axis=1)
    return 1/max(float(np.median(distance)), 1e-8)


def export_model(model, transform, objective):
    centers = model.support_vectors_.copy()
    coefficient = model.dual_coef_.ravel().copy()
    if objective == "pair_delta":
        # Algebraically exact antisymmetrization. Antipodal training supports
        # can share one center with coefficient (a_plus-a_minus)/2.
        nonzero = centers != 0
        first = np.argmax(nonzero, axis=1)
        orientation = np.where(centers[np.arange(len(centers)), first] < 0, -1., 1.)
        centers *= orientation[:, None]
        centers[centers == 0] = 0.  # Canonicalize signed zero before deduplication.
        centers, inverse = np.unique(centers, axis=0, return_inverse=True)
        coefficient = np.bincount(inverse, weights=.5*coefficient*orientation,
                                  minlength=len(centers))
    return {**transform, "centers": centers, "coefficient": coefficient,
            "center_norm": np.sum(centers*centers, axis=1),
            "gamma": np.asarray(model._gamma), "intercept": np.asarray(model.intercept_[0]),
            "pair_delta": np.asarray(objective == "pair_delta")}


def kernel_values(queries, asset, antisymmetric=False):
    output = np.empty(len(queries), dtype=np.float64)
    centers, coefficient = asset["centers"], asset["coefficient"]
    gamma = float(asset["gamma"])
    for start in range(0, len(queries), 256):
        q = np.ascontiguousarray(queries[start:start+256], dtype=np.float64)
        dot = q @ centers.T
        norms = np.sum(q*q, axis=1)[:, None] + asset["center_norm"][None, :]
        positive = np.exp(-gamma*np.maximum(norms-2*dot, 0))
        if antisymmetric:
            positive -= np.exp(-gamma*np.maximum(norms+2*dot, 0))
            output[start:start+len(q)] = positive @ coefficient
        else:
            output[start:start+len(q)] = positive @ coefficient + float(asset["intercept"])
    return output


def tournament(x, valid, decision):
    pairs = [(i, j) for i in range(4) for j in range(i+1, 4)]
    delta = np.stack([x[:, i]-x[:, j] for i, j in pairs], axis=1)
    pair_valid = np.stack([valid[:, i] & valid[:, j] for i, j in pairs], axis=1)
    values = np.zeros(pair_valid.shape, dtype=np.float64)
    values[pair_valid] = decision(delta[pair_valid])
    scores = np.zeros(valid.shape, dtype=np.float64)
    for p, (i, j) in enumerate(pairs):
        scores[:, i] += values[:, p]
        scores[:, j] -= values[:, p]
    return scores / np.maximum(valid.sum(1, keepdims=True)-1, 1)


def score_transformed(x, valid, asset):
    if bool(asset["pair_delta"]):
        return tournament(x, valid, lambda q: kernel_values(q, asset, True))
    return kernel_values(x.reshape(-1, x.shape[-1]), asset).reshape(x.shape[:2])


def kernel_rank_scores(bank, embeddings, asset, features=None):
    if not bank:
        return np.zeros((0, 4))
    if features is None:
        features = extract_features(bank) if len(asset["text_multiplier"]) else np.empty((len(bank), 4, 0))
    return score_transformed(transformed(embeddings, features, asset), valid_mask(bank), asset)


def predict(bank, arrays, checkpoint):
    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint/"config.json").read_text())
    with np.load(checkpoint/"model.npz", allow_pickle=False) as asset:
        scores = kernel_rank_scores(bank, arrays["embeddings"], asset)
    return select(bank, scores, config["aggregation"])


def score_scale(scores, valid):
    centered = scores-scores.mean(1, keepdims=True)
    return float(max(np.std(centered[valid]), 1e-6))


def native_scores(x, valid, model, objective):
    if objective == "pair_delta":
        return tournament(x, valid, lambda q: .5*(model.decision_function(q)-model.decision_function(-q)))
    return model.decision_function(x.reshape(-1, x.shape[-1])).reshape(x.shape[:2])


def main():
    from sklearn.svm import SVC
    from threadpoolctl import threadpool_limits
    from experiment_utils import development_data, metrics
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.output
    if (out/"protocol.json").exists():
        raise SystemExit("Experiment already exists; preserve its evidence and use a new output directory")
    started = time.perf_counter()
    configs = [{"representation": rep, "objective": objective, "gamma_multiplier": gamma, "C": c}
               for rep in ("pca64", "pca64_text75") for objective in ("candidate", "pair_delta")
               for gamma in (.25, 1., 4.) for c in (.3, 3.)]
    for cfg in configs:
        cfg["key"] = f'{cfg["representation"]}_{cfg["objective"]}_g{cfg["gamma_multiplier"]:g}_C{cfg["C"]:g}'
    protocol = {"seed": SEED, "folds": 3,
        "fold_assignment": "p=np.random.default_rng(20261015).permutation(N); fold[p]=np.arange(N)%3",
        "finite_configurations": configs, "model_fit_calls": 3*len(configs)+1,
        "representation": "Fit raw-embedding standardization, randomized PCA64, PC standardization and optional 75-feature standardization on fitting-side questions only. Divide PC/text blocks by sqrt(64)/sqrt(75).",
        "PCA": {"components": 64, "solver": "randomized", "iterated_power": 4, "seed": "20261015+fold; 20261015 for full refit"},
        "kernel": "exact RBF; gamma=multiplier/median squared distance of 8192 sampled fitting-design pairs",
        "candidate_objective": "RBF SVC on valid candidates of disagreeing questions; each question total mass1, correct/incorrect halves equal, weights renormalized to mean1",
        "pair_delta_objective": "RBF SVC on correct-minus-incorrect transformed features and their negatives; equal total question weight. Candidate scores average antisymmetric margins versus other valid candidates.",
        "SVC": {"cache_size_MB": 512, "tol": 1e-4, "max_iter": 200000, "shrinking": True, "probability": False},
        "inference": "float64 NumPy RBF; pairwise support antipodes collapsed algebraically; chunks256",
        "normalization": "OOF raw score divided by valid within-question centered std on outer fitting questions only",
        "selection": "highest total training-OOF hard correct, then all correct, then summed fold exported bytes, then earlier config; max/mean aggregation chosen on same OOF",
        "resource_eligibility": "Every fold export <=16MiB; selected full export checked including config. Training-input latency probe plus validation timing, not official reference-machine eligibility.",
        "validation": "Only selected configuration and aggregation; one final full-refit validation evaluation",
        "data_access": ["train during selection", "validation once after frozen internal decision"],
        "forbidden": "test inputs/references/scores and frozen/release bundles", "threads": 4}
    dump(out/"protocol.json", protocol)
    source_names = [Path(__file__), Path(__file__).with_name("text_features.py"),
                    Path(__file__).with_name("selection.py"), Path(__file__).with_name("common.py")]
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_names}
    dump(out/"source_hashes.json", source_hashes)
    bank, embeddings, y, hard = development_data("train")
    labels, valid = y.astype(bool), valid_mask(bank)
    feature_tick = time.perf_counter()
    features = extract_features(bank)
    feature_seconds = time.perf_counter()-feature_tick
    permutation = np.random.default_rng(SEED).permutation(len(bank))
    fold = np.empty(len(bank), dtype=np.int16)
    fold[permutation] = np.arange(len(bank)) % 3
    np.savez(out/"fold_assignment.npz", fold=fold, labels=labels, valid=valid, hard=hard)
    raw_oof = np.full((len(bank), 4, len(configs)), np.nan)
    normalized_oof = raw_oof.copy()
    records = []
    with threadpool_limits(limits=4):
        for k in range(3):
            fit_ids, held_ids = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
            assert not np.intersect1d(fit_ids, held_ids).size
            transform_tick = time.perf_counter()
            transform = fit_transform(embeddings[fit_ids], features[fit_ids], SEED+k)
            transform_seconds = time.perf_counter()-transform_tick
            np.savez(out/f"fold_{k}_transform.npz", **transform)
            for rep in ("pca64", "pca64_text75"):
                tf = representation(transform, rep)
                xfit = transformed(embeddings[fit_ids], features[fit_ids], tf)
                xheld = transformed(embeddings[held_ids], features[held_ids], tf)
                for objective in ("candidate", "pair_delta"):
                    design, target, weight, eligible_count = design_matrix(xfit, labels[fit_ids], valid[fit_ids], objective)
                    gamma_base = base_gamma(design, SEED+k)
                    for index, cfg in enumerate(configs):
                        if cfg["representation"] != rep or cfg["objective"] != objective:
                            continue
                        model = SVC(C=cfg["C"], kernel="rbf", gamma=gamma_base*cfg["gamma_multiplier"],
                                    cache_size=512, tol=1e-4, max_iter=200000, shrinking=True,
                                    probability=False, random_state=SEED+k)
                        tick = time.perf_counter()
                        model.fit(design, target, sample_weight=weight)
                        fit_seconds = time.perf_counter()-tick
                        asset = export_model(model, tf, objective)
                        path = out/f"fold_{k}"/(cfg["key"]+".npz")
                        path.parent.mkdir(exist_ok=True)
                        np.savez(path, **asset)
                        asset_bytes = sum(v.nbytes for v in asset.values())
                        tick = time.perf_counter()
                        held_scores = score_transformed(xheld, valid[held_ids], asset)
                        inference_seconds = time.perf_counter()-tick
                        native = native_scores(xheld[:64], valid[held_ids[:64]], model, objective)
                        error = float(np.max(np.abs(native-held_scores[:64])))
                        assert error < 2e-6, (cfg, error)
                        tick = time.perf_counter()
                        train_scores = score_transformed(xfit, valid[fit_ids], asset)
                        scale = score_scale(train_scores, valid[fit_ids])
                        normalization_seconds = time.perf_counter()-tick
                        raw_oof[held_ids, :, index] = held_scores
                        normalized_oof[held_ids, :, index] = held_scores/scale
                        row = {**cfg, "fold": k, "fit_questions": len(fit_ids), "heldout_questions": len(held_ids),
                               "eligible_fit_questions": eligible_count, "fitting_rows": len(design),
                               "gamma": float(model._gamma), "support_vectors": len(model.support_),
                               "exported_centers": len(asset["centers"]), "asset_array_bytes": asset_bytes,
                               "asset_file_bytes": path.stat().st_size, "training_score_scale": scale,
                               "fit_seconds": fit_seconds, "heldout_numpy_score_seconds": inference_seconds,
                               "normalization_seconds": normalization_seconds, "transform_seconds_shared": transform_seconds,
                               "native_export_max_error_64_questions": error, "fit_status": int(model.fit_status_),
                               "iterations": int(model.n_iter_[0]), "resource_eligible": asset_bytes < 16*2**20,
                               "heldout_metrics": {aggregation: metrics(select([bank[q] for q in held_ids], held_scores, aggregation),
                                                                          labels[held_ids], hard[held_ids])
                                                   for aggregation in ("max", "mean")}}
                        records.append(row)
                        dump(out/"fold_results.json", records)
                        print(json.dumps({"fold": k, "key": cfg["key"], "fit_seconds": round(fit_seconds, 2),
                                          "support": len(model.support_), "centers": len(asset["centers"]),
                                          "hard": {a: m["hard_correct"] for a, m in row["heldout_metrics"].items()}}), flush=True)
            np.savez(out/"oof_logits.npz", raw=raw_oof, normalized=normalized_oof, labels=labels, valid=valid,
                     hard=hard, fold=fold, keys=np.array([c["key"] for c in configs]))
        assert np.isfinite(raw_oof).all() and np.isfinite(normalized_oof).all()
        evaluations = []
        for i, cfg in enumerate(configs):
            rr = [r for r in records if r["key"] == cfg["key"]]
            for aggregation in ("max", "mean"):
                evaluations.append({**cfg, "aggregation": aggregation,
                    "resource_eligible": all(r["resource_eligible"] and r["fit_status"] == 0 for r in rr),
                    "summed_fold_asset_bytes": sum(r["asset_array_bytes"] for r in rr),
                    "metrics": metrics(select(bank, normalized_oof[:, :, i], aggregation), labels, hard)})
        dump(out/"internal_results.json", evaluations)
        eligible = [r for r in evaluations if r["resource_eligible"]]
        best = max(eligible, key=lambda r: (r["metrics"]["hard_correct"], r["metrics"]["correct"], -r["summed_fold_asset_bytes"]))
        dump(out/"training_selection.json", {"selected": best, "selection_before_validation_access": True,
                                            "source_hashes": source_hashes, "elapsed_seconds": time.perf_counter()-started})
        print(json.dumps({"training_selected": best}), flush=True)
        transform_tick = time.perf_counter()
        full_transform = representation(fit_transform(embeddings, features, SEED), best["representation"])
        xfull = transformed(embeddings, features, full_transform)
        design, target, weight, _ = design_matrix(xfull, labels, valid, best["objective"])
        gamma = base_gamma(design, SEED)*best["gamma_multiplier"]
        transform_seconds = time.perf_counter()-transform_tick
        model = SVC(C=best["C"], kernel="rbf", gamma=gamma, cache_size=512, tol=1e-4,
                    max_iter=200000, shrinking=True, probability=False, random_state=SEED)
        tick = time.perf_counter()
        model.fit(design, target, sample_weight=weight)
        full_fit_seconds = time.perf_counter()-tick
        asset = export_model(model, full_transform, best["objective"])
        bundle = out/"selected_bundle"
        bundle.mkdir()
        np.savez(bundle/"model.npz", **asset)
        scale = score_scale(score_transformed(xfull, valid, asset), valid)
        config = {"method": "kernel_rank", "key": best["key"], "aggregation": best["aggregation"],
                  "training_score_scale": scale, "selection": "three-fold training OOF only", "seed": SEED}
        dump(bundle/"config.json", config)
        import zipfile
        with zipfile.ZipFile(bundle/"model.npz") as archive:
            asset_bytes = sum(i.file_size for i in archive.infolist())+(bundle/"config.json").stat().st_size
        if asset_bytes > 16*2**20 or model.fit_status_:
            raise SystemExit("Training-selected full model failed resource/convergence gate; no validation read")
        # This is the first validation access in the experiment.
        vb, ve, vy, vh = development_data("validation")
        tick = time.perf_counter()
        val_scores = kernel_rank_scores(vb, ve, asset)
        inference_seconds = time.perf_counter()-tick
        result = metrics(select(vb, val_scores, best["aggregation"]), vy, vh)
        np.savez(out/"validation_logits.npz", raw=val_scores, normalized=val_scores/scale, labels=vy, hard=vh)
        dump(out/"validation_result.json", {"selected": best, "metrics": result,
                                            "inference_seconds": inference_seconds, "validation_evaluations": 1})
        summary = {"protocol": protocol, "best": best, "validation_result": result,
                   "validation_inference_seconds_including_text": inference_seconds,
                   "selected_asset_uncompressed_npy_plus_config_bytes": asset_bytes,
                   "full_refit_seconds": full_fit_seconds, "full_transform_seconds": transform_seconds,
                   "full_support_vectors": len(model.support_), "full_exported_centers": len(asset["centers"]),
                   "training_score_scale": scale, "feature_extraction_train_seconds": feature_seconds,
                   "total_model_fit_seconds": sum(r["fit_seconds"] for r in records)+full_fit_seconds,
                   "seconds": time.perf_counter()-started, "model_fit_calls": len(records)+1,
                   "training_questions": len(bank), "training_hard_questions": int(hard.sum()),
                   "source_hashes": source_hashes,
                   "checkpoint_sha256": hashlib.sha256((bundle/"model.npz").read_bytes()).hexdigest()}
        dump(out/"summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
