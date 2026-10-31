"""Four predeclared conditional-comparison recipes; training folds only.

This driver deliberately has no validation stage. The selected asset and aligned
OOF scores are written for the root's independent decision about further study.
"""
from __future__ import annotations
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "4"
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np
from scipy.special import logsumexp
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
from answer_group_model import group_weights
from conditional_pair import (PAIRS, PARAMETERS, borda, conditional_pair_scores,
    pair_inputs, pair_values, score_features, stable_text_features, transform_inputs, valid_candidates)
from experiment_utils import development_data, dump, metrics
from selection import select

ROOT = Path(__file__).resolve().parent
OUT = ROOT/"runs/conditional_pair"
FOLD_SEED = 20261015
INIT_SEED = 7506


def array_hash(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def protocol():
    return {"configurations": [{"pca": k, "l2": l2, "hidden": 64, "key": f"pca{k}_l2{l2:g}"}
                               for k in (32, 64) for l2 in (.003, .03)],
            "question_folds": 3, "fold_seed": FOLD_SEED,
            "fold_mapping": "fold[default_rng(20261015).permutation(N)]=arange(N)%3",
            "inner_split": "80/20 of outer-fitting questions; seed=20261015+100+fold; fit-side eligible questions only enter gradients",
            "feature_fit": "raw embedding mean/std on valid fitting candidates; randomized PCA32/64 (iterated_power4); coordinate std of fitting projected candidates; text75 fitting-valid mean/std; all coordinates clipped[-12,12] after standardization",
            "text75": "Existing features; five relative mean columns recomputed in float64 then converted back to float32; no new features",
            "PCA_sharing": "Each dimension fitted once per fold inner/outer and reused across two L2 configs; selected full transform freshly fitted",
            "PCA_seeds": "20261015+100+fold inner, 20261015+fold outer, 20261015 full",
            "comparator": "d=xi-xj; m=(xi+xj)/2; h=.5*v.T[tanh(d*Wd+m*Wm+b)-tanh(-d*Wd+m*Wm+b)]+d*skip; six unordered candidate pairs",
            "Borda": "sum outgoing h divided by max(number_valid_candidates-1,1); invalid-pair contributions zero; selfpairs excluded; answer-group mean afterward",
            "objective": "logsumexp(valid answer-group means)-logsumexp(correct answer-group means), equal question weight; hard eligible training questions only",
            "optimizer": {"name": "Adam", "learning_rate": .001, "beta1": .9, "beta2": .999,
                          "epsilon": 1e-8, "batch_questions": 128, "dropout": 0., "max_epochs": 60,
                          "patience": 8, "minimum_improvement": 1e-6, "initialization_seed": INIT_SEED,
                          "dtype": "float64", "l2": "0.5*lambda*sum squared weights including skip; exclude hidden bias"},
            "stopping": "Lowest unregularized group CE on inner-stopping hard questions; fresh outer refit for selected epoch count",
            "selection": "pooled training OOF hard correct, then lower raw group CE, then fewer PCA dimensions, then earlier predefined config",
            "nested_selection_diagnostic": "Within each outer fold choose recipe only by inner-heldout hard correct, group CE, smaller PCA, predefined order; stitch its outer predictions. Save selected-epoch inner scores and report nested recipe OOF separately; do not change final pooled-OOF selection rule",
            "final_refit": "median selected inner epochs of winning config; fresh initialization and full-training input statistics",
            "model_fit_calls": 25, "PCA_fit_calls": 13, "aggregation": "mean, fixed", "threads": 4,
            "score_scale": "std of all-four-centered fitting-question scores on valid fitting candidates; no heldout rows",
            "cost_probe": "First planned config/fold retained; estimated 25-model study >3600s or 467-question inference >30s stops before more configs",
            "resource_limit": "uncompressed checkpoint NPY members+configuration<=16MiB; NumPy-only inference",
            "validation": "Not loaded or evaluated by this driver; root authorization required for later evaluation",
            "forbidden": "test inputs/labels/results, frozen or release modification, external weights/data, expansion of the four-config grid"}


def fit_transform(embeddings, text, valid, dimensions, seed):
    started = time.perf_counter()
    raw = np.asarray(embeddings, np.float64)[valid]
    mean = raw.mean(0); scale = np.maximum(raw.std(0), 1e-5)
    normalized = (raw-mean)/scale
    pca = PCA(n_components=dimensions, svd_solver="randomized", iterated_power=4, random_state=seed).fit(normalized)
    projected = pca.transform(normalized)
    pscale = np.maximum(projected.std(0), 1e-5)
    projection = pca.components_.T/scale[:, None]/pscale[None, :]
    bias = -mean@projection-(pca.mean_@pca.components_.T)/pscale
    values = np.asarray(text, np.float64)[valid]
    tmean = values.mean(0); tscale = np.maximum(values.std(0), 1e-5)
    asset = {"embedding_projection": projection, "embedding_bias": bias,
             "text_multiplier": 1/tscale, "text_bias": -tmean/tscale}
    exported = raw@projection+bias
    error = float(np.max(np.abs(exported-projected/pscale)))
    assert error < 1e-8, error
    return asset, {"dimensions": dimensions, "fit_questions": len(embeddings), "valid_fit_candidates": len(raw),
                   "seed": seed, "seconds": time.perf_counter()-started,
                   "native_PCA_export_max_error": error, "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum())}


def initialize(dim, hidden=64):
    rng = np.random.default_rng(INIT_SEED)
    return [rng.normal(0, 1/np.sqrt(2*dim), (dim, hidden)),
            rng.normal(0, 1/np.sqrt(2*dim), (dim, hidden)), np.zeros(hidden),
            rng.normal(0, 1/np.sqrt(hidden), hidden), np.zeros(dim)]


def group_loss_gradient(scores, weights, good, group_valid):
    grouped = np.einsum("ngc,nc->ng", weights, scores)
    all_scores = np.where(group_valid, grouped, -np.inf)
    good_scores = np.where(good, grouped, -np.inf)
    all_z = logsumexp(all_scores, axis=1, keepdims=True)
    good_z = logsumexp(good_scores, axis=1, keepdims=True)
    gradient = (np.exp(all_scores-all_z)-np.exp(good_scores-good_z))/len(scores)
    return float(np.mean(all_z-good_z)), np.einsum("ngc,ng->nc", weights, gradient)


def loss_gradient(x, valid, weights, good, group_valid, model, l2):
    wd, wm, bias, output, skip = model
    d, m, pair_valid = pair_inputs(x, valid)
    delta = d@wd; condition = m@wm+bias
    plus, minus = np.tanh(delta+condition), np.tanh(-delta+condition)
    h = .5*(plus-minus)@output+d@skip
    scores = borda(h, pair_valid, valid)
    loss, candidate_gradient = group_loss_gradient(scores, weights, good, group_valid)
    denominator = np.maximum(valid.sum(1)-1, 1)
    gh = np.stack([(candidate_gradient[:, i]-candidate_gradient[:, j])/denominator
                   for i, j in PAIRS], axis=1)*pair_valid
    gp = .5*gh[..., None]*output*(1-plus*plus)
    gm = -.5*gh[..., None]*output*(1-minus*minus)
    flat_d, flat_m = d.reshape(-1, d.shape[-1]), m.reshape(-1, m.shape[-1])
    gradients = [flat_d.T@(gp-gm).reshape(-1, len(output)),
                 flat_m.T@(gp+gm).reshape(-1, len(output)),
                 (gp+gm).sum(axis=(0, 1)),
                 .5*(plus-minus).reshape(-1, len(output)).T@gh.ravel(), flat_d.T@gh.ravel()]
    for k in (0, 1, 3, 4):
        loss += .5*l2*float(np.sum(model[k]**2)); gradients[k] += l2*model[k]
    return loss, gradients


def finite_difference_check():
    rng = np.random.default_rng(44); x = rng.normal(size=(3, 4, 5))
    valid = np.array([[1, 1, 1, 0], [1, 1, 1, 1], [1, 0, 0, 1]], bool)
    weights = np.array([[[.5,.5,0,0],[0,0,1,0],[0,0,0,0],[0,0,0,0]],
                        [[1,0,0,0],[0,.5,.5,0],[0,0,0,1],[0,0,0,0]],
                        [[1,0,0,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]]], np.float64)
    good = np.array([[1,0,0,0],[0,1,0,0],[0,1,0,0]], bool); gv = weights.sum(2)>0
    model = initialize(5, 4); loss, gradients = loss_gradient(x, valid, weights, good, gv, model, .03)
    errors = []
    for k, parameter in enumerate(model):
        for flat in np.linspace(0, parameter.size-1, min(5, parameter.size), dtype=int):
            index = np.unravel_index(flat, parameter.shape); original = parameter[index]; eps = 1e-5
            parameter[index] = original+eps; upper = loss_gradient(x, valid, weights, good, gv, model, .03)[0]
            parameter[index] = original-eps; lower = loss_gradient(x, valid, weights, good, gv, model, .03)[0]
            parameter[index] = original
            error = abs((upper-lower)/(2*eps)-gradients[k][index]); assert error < 1e-6
            errors.append({"parameter": PARAMETERS[k], "index": list(map(int, index)), "error": error})
    d, m, pv = pair_inputs(x, valid)
    h = pair_values(d, m, model)
    antisymmetry = float(np.max(np.abs(h+pair_values(-d, m, model))))
    self_error = float(np.max(np.abs(pair_values(np.zeros_like(d), m, model))))
    mean_effect = float(np.max(np.abs(h-pair_values(d, m+1., model))))
    changed = x.copy(); changed[~valid] += 500.
    invalid_error = float(np.max(np.abs(score_features(x, valid, model)-score_features(changed, valid, model))))
    permutation = np.array([2,0,3,1])
    permutation_error = float(np.max(np.abs(score_features(x[:,permutation], valid[:,permutation], model)-score_features(x,valid,model)[:,permutation])))
    # A nonzero triangle sum witnesses a comparator not constrained to s(xi)-s(xj).
    cycle = float(h[1, 0]+h[1, 3]-h[1, 1])
    assert antisymmetry == 0 and self_error == 0 and mean_effect > 1e-4
    assert invalid_error < 1e-10 and permutation_error < 1e-10 and abs(cycle)>1e-4
    return {"finite_differences": errors, "max_gradient_error": max(e["error"] for e in errors),
            "antisymmetry_error": antisymmetry, "self_pair_error": self_error, "conditional_mean_effect": mean_effect,
            "nonzero_triangle_cycle": cycle, "invalid_candidate_isolation_error": invalid_error,
            "candidate_permutation_error": permutation_error, "synthetic_loss": loss}


def fit(x, valid, weights, good, group_valid, config, epochs=60, heldout=None):
    model = initialize(x.shape[-1], config["hidden"])
    first = [np.zeros_like(p) for p in model]; second = [np.zeros_like(p) for p in model]
    rng = np.random.default_rng(INIT_SEED+1)
    best, selected, stale, step = np.inf, 0, 0, 0
    best_model = None
    curve = []; started = time.perf_counter()
    for epoch in range(1, epochs+1):
        order = rng.permutation(len(x)); total = 0.
        for start in range(0, len(x), 128):
            ii = order[start:start+128]
            loss, gradients = loss_gradient(x[ii], valid[ii], weights[ii], good[ii], group_valid[ii], model, config["l2"])
            total += loss*len(ii); step += 1
            for k, gradient in enumerate(gradients):
                first[k] = .9*first[k]+.1*gradient; second[k] = .999*second[k]+.001*gradient*gradient
                model[k] -= .001*(first[k]/(1-.9**step))/(np.sqrt(second[k]/(1-.999**step))+1e-8)
        record = {"epoch": epoch, "regularized_batch_train_loss": total/len(x), "seconds": time.perf_counter()-started}
        if heldout is not None:
            xx, vv, ww, gg, gvv = heldout
            held_loss = group_loss_gradient(score_features(xx, vv, model), ww, gg, gvv)[0]
            record["heldout_group_ce"] = held_loss
            if held_loss < best-1e-6:
                best, selected, stale = held_loss, epoch, 0
                best_model = [p.copy() for p in model]
            else:
                stale += 1
        curve.append(record)
        if heldout is not None and stale >= 8:
            break
    assert all(np.isfinite(p).all() for p in model)
    return best_model if heldout is not None else model, {"selected_epochs": selected if heldout is not None else epochs, "completed_epochs": len(curve),
                   "update_steps": step, "seconds": time.perf_counter()-started, "curve": curve}


def exported(model, transform):
    return {**transform, **dict(zip(PARAMETERS, model))}


def score_scale(scores, valid):
    return float(max(np.std((scores-scores.mean(1, keepdims=True))[valid]), 1e-6))


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args(); out = args.output
    if (out/"protocol.json").exists():
        raise SystemExit("Preserving existing experiment; choose a new output directory")
    started = time.perf_counter(); plan = protocol(); configs = plan["configurations"]
    dump(out/"protocol.json", plan)
    hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
              ("conditional_pair.py", "conditional_pair_experiment.py", "answer_group_model.py", "text_features.py", "selection.py", "common.py", "experiment_utils.py")}
    dump(out/"source_hashes.json", hashes)
    dump(out/"mathematical_checks.json", finite_difference_check())
    bank, embeddings, labels, hard = development_data("train")
    labels = labels.astype(bool); valid = valid_candidates(bank); weights = group_weights(bank).astype(np.float64)
    group_valid = weights.sum(2)>0; good = np.einsum("ngc,nc->ng", weights, labels)>0
    eligible = good.any(1)&(group_valid&~good).any(1)
    assert np.array_equal(eligible, hard)
    tick = time.perf_counter(); text = stable_text_features(bank); feature_seconds = time.perf_counter()-tick
    n = len(bank); fold = np.empty(n, np.int16)
    fold[np.random.default_rng(FOLD_SEED).permutation(n)] = np.arange(n)%3
    ids = np.array([q["id"] for q in bank])
    np.savez(out/"fold_assignment.npz", ids=ids, fold=fold, labels=labels, valid=valid, hard=hard)
    raw = np.full((n, 4, len(configs)), np.nan); normalized = raw.copy()
    records, transform_records = [], []; cached = {}
    with threadpool_limits(limits=4):
        for index, config in enumerate(configs):
            for f in range(3):
                fit_ids, eval_ids = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
                order = np.random.default_rng(FOLD_SEED+100+f).permutation(fit_ids); cutoff = int(.8*len(order))
                inner_ids, stop_ids = order[:cutoff], order[cutoff:]
                transforms = {}
                for stage, fit_side, seed in (("inner", inner_ids, FOLD_SEED+100+f), ("outer", fit_ids, FOLD_SEED+f)):
                    cache_key = (config["pca"], f, stage)
                    if cache_key not in cached:
                        tf, info = fit_transform(embeddings[fit_side], text[fit_side], valid[fit_side], config["pca"], seed)
                        cached[cache_key] = tf
                        transform_records.append({**info, "fold": f, "stage": stage, "fit_ids_sha256": array_hash(fit_side)})
                        dump(out/"transform_records.json", transform_records)
                    transforms[stage] = cached[cache_key]
                ii, si = inner_ids[hard[inner_ids]], stop_ids[hard[stop_ids]]
                inner_x = transform_inputs(embeddings[ii], text[ii], transforms["inner"])
                stop_x = transform_inputs(embeddings[si], text[si], transforms["inner"])
                inner_model, early = fit(inner_x, valid[ii], weights[ii], good[ii], group_valid[ii], config,
                               heldout=(stop_x, valid[si], weights[si], good[si], group_valid[si]))
                inner_fit_scores = score_features(transform_inputs(embeddings[inner_ids], text[inner_ids], transforms["inner"]), valid[inner_ids], inner_model)
                inner_scale = score_scale(inner_fit_scores, valid[inner_ids])
                inner_stop_scores = score_features(transform_inputs(embeddings[stop_ids], text[stop_ids], transforms["inner"]), valid[stop_ids], inner_model)
                inner_ce = group_loss_gradient(inner_stop_scores[hard[stop_ids]], weights[si], good[si], group_valid[si])[0]
                inner_metrics = metrics(select([bank[i] for i in stop_ids], inner_stop_scores, "mean"), labels[stop_ids], hard[stop_ids])
                inner_output = out/"inner_selected"/f"fold_{f}_{config['key']}.npz"
                inner_output.parent.mkdir(exist_ok=True)
                np.savez(inner_output, raw=inner_stop_scores, normalized=inner_stop_scores/inner_scale,
                         ids=ids[stop_ids], indices=stop_ids, labels=labels[stop_ids], valid=valid[stop_ids], hard=hard[stop_ids],
                         selected_epoch=early["selected_epochs"], training_score_scale=inner_scale)
                active = fit_ids[hard[fit_ids]]
                outer_x = transform_inputs(embeddings[active], text[active], transforms["outer"])
                model, refit = fit(outer_x, valid[active], weights[active], good[active], group_valid[active], config, epochs=early["selected_epochs"])
                asset = exported(model, transforms["outer"])
                path = out/f"fold_{f}"/(config["key"]+".npz"); path.parent.mkdir(exist_ok=True); np.savez(path, **asset)
                xheld = transform_inputs(embeddings[eval_ids], text[eval_ids], asset)
                scores = score_features(xheld, valid[eval_ids], model)
                train_scores = score_features(transform_inputs(embeddings[fit_ids], text[fit_ids], asset), valid[fit_ids], model)
                scale = score_scale(train_scores, valid[fit_ids])
                with np.load(path, allow_pickle=False) as loaded:
                    replay = conditional_pair_scores([bank[i] for i in eval_ids[:64]], embeddings[eval_ids[:64]], loaded)
                error = float(np.max(np.abs(replay-scores[:64]))); assert error < 1e-10, error
                assert select([bank[i] for i in eval_ids[:64]], replay, "mean") == select([bank[i] for i in eval_ids[:64]], scores[:64], "mean")
                raw[eval_ids, :, index] = scores; normalized[eval_ids, :, index] = scores/scale
                record = {"key": config["key"], "configuration": config, "fold": f,
                          "outer_fit_questions": len(fit_ids), "heldout_questions": len(eval_ids), "outer_fit_hard": len(active),
                          "inner_fit_questions": len(inner_ids), "inner_fit_hard": len(ii), "stopping_questions": len(stop_ids), "stopping_hard": len(si),
                          "outer_fit_ids_sha256": array_hash(fit_ids), "heldout_ids_sha256": array_hash(eval_ids),
                          "inner_fit_ids_sha256": array_hash(inner_ids), "stopping_ids_sha256": array_hash(stop_ids),
                          "inner_selected": {"epoch": early["selected_epochs"], "metrics": inner_metrics, "group_ce": inner_ce,
                                             "inner_training_score_scale": inner_scale, "scores": str(inner_output.relative_to(out))},
                          "early": early, "refit": refit, "training_score_scale": scale,
                          "asset_array_bytes": sum(v.nbytes for v in asset.values()), "export_max_error": error,
                          "metrics": metrics(select([bank[i] for i in eval_ids], scores, "mean"), labels[eval_ids], hard[eval_ids])}
                records.append(record); dump(out/"fold_results.json", records)
                if index == 0 and f == 0:
                    probe_ids = np.flatnonzero(hard)[:467]; tick = time.perf_counter()
                    conditional_pair_scores([bank[i] for i in probe_ids], embeddings[probe_ids], asset)
                    inference_seconds = time.perf_counter()-tick
                    estimate = (early["seconds"]+refit["seconds"])*12.5
                    probe = {"first_inner_seconds": early["seconds"], "first_refit_seconds": refit["seconds"],
                             "estimated_25_model_seconds": estimate, "training_proxy467_inference_seconds": inference_seconds,
                             "retained_in_grid": True, "cost_limit_passed": estimate <= 3600 and inference_seconds <= 30}
                    dump(out/"cost_probe.json", probe)
                    if not probe["cost_limit_passed"]:
                        raise SystemExit("Predeclared cost gate exceeded; no further fits")
                print(json.dumps({"key": config["key"], "fold": f, "epoch": early["selected_epochs"],
                                  "fit_seconds": early["seconds"]+refit["seconds"], "hard_correct": record["metrics"]["hard_correct"]}), flush=True)
            np.savez(out/"oof_logits.npz", raw=raw, normalized=normalized, keys=np.array([c["key"] for c in configs]),
                     ids=ids, fold=fold, labels=labels, valid=valid, hard=hard)
        assert np.isfinite(raw).all() and np.isfinite(normalized).all()
        results = []
        for i, config in enumerate(configs):
            ce = group_loss_gradient(raw[hard, :, i], weights[hard], good[hard], group_valid[hard])[0]
            results.append({"key": config["key"], "configuration": config, "oof_group_ce": ce,
                            "metrics": metrics(select(bank, raw[:, :, i], "mean"), labels, hard)})
        dump(out/"internal_results.json", results)
        nested_raw = np.empty((n, 4)); nested_normalized = np.empty_like(nested_raw); nested_selections = []
        for f in range(3):
            candidates = [r for r in records if r["fold"] == f]
            nested_best = max(candidates, key=lambda r:(r["inner_selected"]["metrics"]["hard_correct"],
                              -r["inner_selected"]["group_ce"], -r["configuration"]["pca"]))
            i = next(i for i, c in enumerate(configs) if c["key"] == nested_best["key"])
            mask = fold == f; nested_raw[mask] = raw[mask, :, i]; nested_normalized[mask] = normalized[mask, :, i]
            nested_selections.append({"fold": f, "key": nested_best["key"], "inner_selected": nested_best["inner_selected"],
                                      "outer_metrics": nested_best["metrics"]})
        nested_metrics = metrics(select(bank, nested_raw, "mean"), labels, hard)
        dump(out/"nested_recipe_selection.json", {"fold_selections": nested_selections, "metrics": nested_metrics,
             "raw_group_ce": group_loss_gradient(nested_raw[hard], weights[hard], good[hard], group_valid[hard])[0],
             "scope": "Recipe and epoch selection use outer-fit-internal heldout questions only. Outer predictions are evaluated only after those choices; this diagnostic does not alter final pooled OOF recipe selection."})
        np.savez(out/"nested_recipe_oof_logits.npz", raw=nested_raw, normalized=nested_normalized,
                 ids=ids, fold=fold, labels=labels, valid=valid, hard=hard)
        best = max(results, key=lambda r: (r["metrics"]["hard_correct"], -r["oof_group_ce"], -r["configuration"]["pca"]))
        chosen = [r for r in records if r["key"] == best["key"]]
        epochs = int(np.median([r["early"]["selected_epochs"] for r in chosen]))
        dump(out/"training_selection.json", {"selected": best, "full_epochs": epochs, "validation_accessed": False, "source_hashes": hashes})
        tf, full_transform = fit_transform(embeddings, text, valid, best["configuration"]["pca"], FOLD_SEED)
        x = transform_inputs(embeddings, text, tf)
        model, full_fit = fit(x[hard], valid[hard], weights[hard], good[hard], group_valid[hard], best["configuration"], epochs=epochs)
        asset = exported(model, tf); bundle = out/"selected_bundle"; bundle.mkdir(); np.savez(bundle/"model.npz", **asset)
        full_scores = score_features(x, valid, model); scale = score_scale(full_scores, valid)
        dump(bundle/"config.json", {"method": "conditional_pair", "key": best["key"], "configuration": best["configuration"],
                                    "epochs": epochs, "aggregation": "mean", "training_score_scale": scale,
                                    "validation_accessed": False, "model_dtype": "float64"})
        with zipfile.ZipFile(bundle/"model.npz") as archive:
            asset_bytes = sum(i.file_size for i in archive.infolist())+(bundle/"config.json").stat().st_size
        assert asset_bytes <= 16*2**20
        dump(out/"full_refit.json", {"transform": full_transform, "model": full_fit})
        summary = {"protocol": plan, "best": best, "validation_pending": True, "validation_accessed": False,
                   "nested_recipe_selection_metrics": nested_metrics,
                   "model_fit_calls": 2*len(records)+1, "PCA_fit_calls": len(transform_records)+1,
                   "total_model_fit_seconds": sum(r[k]["seconds"] for r in records for k in ("early", "refit"))+full_fit["seconds"],
                   "total_PCA_transform_fit_seconds": sum(r["seconds"] for r in transform_records)+full_transform["seconds"],
                   "full_epochs": epochs, "training_score_scale": scale, "feature_seconds": feature_seconds,
                   "selected_asset_uncompressed_npy_plus_config_bytes": asset_bytes, "total_seconds": time.perf_counter()-started,
                   "source_hashes": hashes, "checkpoint_sha256": hashlib.sha256((bundle/"model.npz").read_bytes()).hexdigest(),
                   "caveat": "OOF winner was selected on these folds; this is not an independent estimate. No validation or test data accessed."}
        dump(out/"summary.json", summary); print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
