"""Nested routing diagnostic conditional on five historically fixed recipes.

All 60 learners are freshly fitted on question-level training partitions.
No validation or test stage exists in this program. See the registered protocol
in docs/selective_routing_protocol.md for the two independent acceptance gates.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "4"

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import zipfile

import numpy as np
import sklearn
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

import answer_group_experiment as group
from answer_group_model import group_model_scores
import conditional_pair_experiment as pair
from conditional_pair import score_features, transform_inputs
import condition_experiment as condition
from condition_features import extract_features as condition_features
from condition_model import feature_matrix as condition_matrix
import kernel_rank as kernel
from quantitative_model import answer_valid, stable_base_features, feature_matrix as quantity_matrix
from common import DATA, parse_answer
from experiment_utils import development_data, dump, metrics
from selection import select
from text_features import extract_features
from tree_model import export_trees, tree_scores

ROOT = Path(__file__).resolve().parent
OUT = ROOT/"runs/selective_routing"
PROTOCOL = ROOT.parent/"docs/selective_routing_protocol.md"
OUTER_SEED = 20261015
INNER_SEED = 20261016
NAMES = ["A", "K", "Q", "P", "C"]
GROUP_CONFIG = dict(representation="group", objective="pairwise", dropout=.2, hidden=128, l2=.01)
PAIR_CONFIG = dict(pca=64, hidden=64, l2=.03)
RULES = (("F_only", 0.), ("low20_disagree", .2), ("low40_disagree", .4))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def folds_for(n, seed):
    fold = np.empty(n, np.int8)
    fold[np.random.default_rng(seed).permutation(n)] = np.arange(n)%3
    return fold


def mixtures(normalized):
    f = .5*normalized[:, :, 0]+.25*normalized[:, :, 1]+.25*normalized[:, :, 2]
    g = .25*f+.5*normalized[:, :, 3]+.25*normalized[:, :, 4]
    return f, g


def routing_inputs(bank, f, g):
    fc, margin, _ = select(bank, f, "mean", return_details=True)
    gc = select(bank, g, "mean")
    disagreement = np.array([parse_answer(row["candidates"][i]["text"]) !=
                             parse_answer(row["candidates"][j]["text"])
                             for row, i, j in zip(bank, fc, gc)])
    return np.asarray(fc), np.asarray(gc), margin, disagreement


def apply_rule(bank, f, g, hard, rule):
    _, _, margin, disagreement = routing_inputs(bank, f, g)
    routed = hard & disagreement & (margin <= rule["threshold"]) if rule["q"] else np.zeros(len(bank), bool)
    return np.where(routed[:, None], g, f), routed


def choose_rule(bank, f, g, labels, hard):
    _, _, margin, disagreement = routing_inputs(bank, f, g)
    rows = []
    for index, (name, q) in enumerate(RULES):
        threshold = float(np.quantile(margin[hard], q, method="linear")) if q and hard.any() else None
        rule = dict(name=name, q=q, threshold=threshold, rule_index=index)
        z, routed = apply_rule(bank, f, g, hard, rule) if hard.any() else (f, np.zeros(len(bank), bool))
        rows.append({**rule, "metrics": metrics(select(bank, z, "mean"), labels, hard),
                     "hard_questions_below_threshold": int((hard & (margin <= threshold)).sum()) if threshold is not None else 0,
                     "hard_disagreements": int((hard & disagreement).sum()), "routed_hard": int(routed.sum())})
    selected = min(rows, key=lambda r: (-r["metrics"]["hard_correct"], r["routed_hard"], r["q"], r["rule_index"]))
    return {"selected": selected, "rules": rows, "selection_data": "fitting-side OOF only"}


def paired(candidate, reference, labels, hard, fold):
    c = labels[np.arange(len(labels)), candidate].astype(bool)
    r = labels[np.arange(len(labels)), reference].astype(bool)
    result = []
    for index in [None, 0, 1, 2]:
        mask = hard if index is None else hard & (fold == index)
        repairs = int((mask & c & ~r).sum()); regressions = int((mask & ~c & r).sum())
        result.append(dict(fold=index, n=int(mask.sum()), candidate_correct=int((c & mask).sum()),
                           reference_correct=int((r & mask).sum()), repairs=repairs, regressions=regressions,
                           gain=repairs-regressions))
    return result


def historical_reference(bank, labels, valid, hard, fold):
    directory = ROOT/"runs/refinement_fusion"
    config = read(directory/"selected_config.json")
    evidence = read(directory/"source_evidence.json")
    assert file_hash(directory/"source_evidence.json") == config["source_evidence_sha256"]
    ids = np.array([r["id"] for r in bank])
    # The historical manifest prefixes dtype and shape, unlike local raw-byte
    # index hashes. Match that schema rather than comparing incompatible hashes.
    def historical_hash(value):
        value = np.ascontiguousarray(value)
        return hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes()).hexdigest()
    assert historical_hash(ids) == evidence["training_ids_sha256"]
    assert historical_hash(fold) == evidence["training_fold_sha256"]
    with np.load(directory/"oof_logits.npz", allow_pickle=False) as data:
        for key, expected in (("labels", labels), ("valid", valid), ("hard", hard), ("fold", fold)):
            assert np.array_equal(data[key], expected), key
        z = data["scores"].copy()
        assert np.array_equal(data["normalized_heads"]@np.asarray(config["weights"]), z)
    result = metrics(select(bank, z, "mean"), labels, hard)
    assert result["hard_correct"] == 1778
    manifest = {p.name: file_hash(p) for p in [directory/"selected_config.json", directory/"source_evidence.json", directory/"oof_logits.npz"]}
    return z, {"metrics": result, "files_sha256": manifest,
               "relationship": "Historical 1778 reference uses fold-specific A epochs; not the matched fixed-14-epoch reference"}


def prepare(bank, embeddings, labels, hard):
    valid = answer_valid(bank)
    weights = group.groups(bank)
    group_valid = weights.sum(2) > 0
    good = np.einsum("ngc,nc->ng", weights, labels) > 0
    eligible = good.any(1) & (group_valid & ~good).any(1)
    assert np.array_equal(eligible, hard)
    assert np.all(good.sum(1) == 1)
    tick = time.perf_counter(); original = extract_features(bank); original_seconds = time.perf_counter()-tick
    tick = time.perf_counter(); stable = stable_base_features(bank); stable_seconds = time.perf_counter()-tick
    tick = time.perf_counter(); conditions = condition_features(bank); condition_seconds = time.perf_counter()-tick
    tick = time.perf_counter(); grouped = group.make_features(embeddings, original, weights, "group"); group_seconds = time.perf_counter()-tick
    return dict(bank=bank, embeddings=embeddings, labels=labels, hard=hard, eligible=eligible,
                valid=valid, weights=weights, good=good, group_valid=group_valid, original=original,
                stable=stable, conditions=conditions, grouped=grouped), {
        "original75_seconds": original_seconds, "stable75_seconds": stable_seconds,
        "condition54_seconds": condition_seconds, "group_pooling_seconds": group_seconds,
        "learned_transform_fits": 0, "cache_scope": "Deterministic input-only features on train only"}


def fit_partition(data, fit_ids, held_ids, seed, directory, records, transforms):
    """Fresh five-base fit, all learned statistics restricted to fit_ids."""
    directory.mkdir(parents=True, exist_ok=True)
    assert not np.intersect1d(fit_ids, held_ids).size
    ix = np.concatenate((fit_ids, held_ids)); nfit = len(fit_ids); local_fit = np.arange(nfit)
    emb = data["embeddings"][ix]; labels = data["labels"][ix]; valid = data["valid"][ix]
    hard = data["hard"][ix]; eligible = data["eligible"][ix]
    original = data["original"][ix]; stable = data["stable"][ix]
    weights = data["weights"][ix]; good = data["good"][ix]; gv = data["group_valid"][ix]
    raw = np.empty((len(ix), 4, 5), np.float64); normalized = raw.copy()
    fit_info = dict(partition=directory.name, fit_questions=nfit, heldout_questions=len(held_ids),
                    fit_hard_questions=int(hard[:nfit].sum()), fit_indices_sha256=array_hash(fit_ids.astype(np.int64)),
                    held_indices_sha256=array_hash(held_ids.astype(np.int64)), transform_seed=seed)
    np.savez(directory/"partition.npz", fit_indices=fit_ids, held_indices=held_ids,
             fit_ids=np.array([data["bank"][i]["id"] for i in fit_ids]),
             held_ids=np.array([data["bank"][i]["id"] for i in held_ids]))
    for column, name in enumerate(NAMES):
        started = time.perf_counter(); transform_seconds = 0.; native_error = 0.; detail = {}
        if name == "A":
            features = data["grouped"][ix]
            mean, std = group.fit_statistics(features[:nfit], gv[:nfit])
            x = group.normalized(features, mean, std)
            active = local_fit[hard[:nfit]]
            tick = time.perf_counter()
            model, curve, epochs = group.fit(x[active], weights[active], good[active], gv[active], GROUP_CONFIG, epochs=14)
            fit_seconds = time.perf_counter()-tick
            z = group.candidate_scores(group.scores(x, model), weights, "group")
            scale = group.scale_scores(z[:nfit], valid[:nfit])
            asset = dict(w1=model[0], b1=model[1], w2=model[2], skip=model[3], mean=mean, scale=std)
            replay = group_model_scores([data["bank"][i] for i in held_ids[:32]], emb[nfit:nfit+32], asset, "group", original[nfit:nfit+32])
            native_error = float(np.max(np.abs(replay-z[nfit:nfit+32])))
            assert native_error < 2e-5
            detail = dict(configuration=GROUP_CONFIG, epochs=epochs, curve=curve, normalizer_fit_questions=nfit)
        elif name == "K":
            tick = time.perf_counter(); tf = kernel.fit_transform(emb[:nfit], original[:nfit], seed)
            transform_seconds = time.perf_counter()-tick
            transforms.append({**fit_info, "head": name, "kind": "PCA", "components": 64, "seconds": transform_seconds})
            x = kernel.transformed(emb, original, tf)
            design, target, weight, count = kernel.design_matrix(x[:nfit], labels[:nfit].astype(bool), valid[:nfit], "pair_delta")
            gamma = kernel.base_gamma(design, seed)
            model = SVC(C=3., kernel="rbf", gamma=gamma, cache_size=512, tol=1e-4,
                        max_iter=200000, shrinking=True, probability=False, random_state=seed)
            tick = time.perf_counter(); model.fit(design, target, sample_weight=weight); fit_seconds = time.perf_counter()-tick
            assert model.fit_status_ == 0
            asset = kernel.export_model(model, tf, "pair_delta")
            z = kernel.score_transformed(x, valid, asset)
            native = kernel.native_scores(x[nfit:nfit+32], valid[nfit:nfit+32], model, "pair_delta")
            native_error = float(np.max(np.abs(native-z[nfit:nfit+32]))); assert native_error < 2e-6
            scale = kernel.score_scale(z[:nfit], valid[:nfit])
            detail = dict(C=3., gamma=gamma, gamma_multiplier=1., objective="pair_delta", eligible_fit_questions=count,
                          support_vectors=int(len(model.support_)), merged_centers=int(len(asset["centers"])),
                          iterations=model.n_iter_.tolist(), fit_status=int(model.fit_status_))
        elif name in ("Q", "C"):
            # This helper is the repaired Q projection verbatim: valid-centered
            # float32 inputs with float64 accumulation, then float32 export.
            tf, info = condition.fit_pls(emb, labels, valid, eligible, local_fit)
            transform_seconds = info["seconds"]
            transforms.append({**fit_info, "head": name, "kind": "PLS", "projection": info,
                               "components": 8, "seconds": info["seconds"],
                               "local_fit_indices_sha256": info["fit_indices_sha256"],
                               "global_eligible_fit_indices_sha256": array_hash(fit_ids[hard[:nfit]].astype(np.int64))})
            leaves = 15 if name == "Q" else 31
            x = quantity_matrix(stable, emb, valid, tf) if name == "Q" else condition_matrix(
                stable, data["conditions"][ix], emb, valid, "condition_pls8", tf)
            tick = time.perf_counter(); model = condition.fit_tree(x, labels, valid, eligible, local_fit, leaves, 300)
            fit_seconds = time.perf_counter()-tick
            z = model.decision_function(x.reshape(-1, x.shape[-1])).reshape(-1, 4)
            if name == "Q":
                asset = {**export_trees(model), **tf, "precision_fix_version": np.array(1, np.int32), "design_code": np.array(3, np.int32)}
                scale = kernel.score_scale(z[:nfit], valid[:nfit])
            else:
                asset = condition.asset_for(model, tf, {"design": "condition_pls8"})
                scale = condition.scale_scores(z, valid, local_fit)
            replay = tree_scores(x[nfit:].reshape(-1, x.shape[-1]), asset).reshape(-1, 4)
            native_error = float(np.max(np.abs(replay-z[nfit:]))); assert native_error < 1e-8
            detail = dict(parameters=condition.parameters(leaves, 300), feature_count=x.shape[-1],
                          helper_equivalence="Q/C use identical eligible-question PLS and HGBT implementation; only fixed feature matrix and leaf count differ")
        else:
            tf, info = pair.fit_transform(emb[:nfit], stable[:nfit], valid[:nfit], 64, seed)
            transform_seconds = info["seconds"]
            transforms.append({**fit_info, "head": name, "kind": "PCA", **info})
            x = transform_inputs(emb, stable, tf); active = local_fit[hard[:nfit]]
            tick = time.perf_counter()
            model, detail = pair.fit(x[active], valid[active], weights[active].astype(np.float64), good[active], gv[active], PAIR_CONFIG, epochs=27)
            fit_seconds = time.perf_counter()-tick
            asset = pair.exported(model, tf); z = score_features(x, valid, model)
            scale = pair.score_scale(z[:nfit], valid[:nfit])
            detail.update(configuration=PAIR_CONFIG)
        assert np.isfinite(z).all() and np.isfinite(scale) and scale >= 1e-6
        asset["score_scale"] = np.array(scale)
        path = directory/(name+".npz"); np.savez(path, **asset)
        with zipfile.ZipFile(path) as archive:
            expanded_bytes = sum(i.file_size for i in archive.infolist())
        assert expanded_bytes < 16*1024*1024
        raw[:, :, column] = z
        # Preserve each source's dtype at normalization, notably A float32.
        normalized[:, :, column] = z/scale
        row = {**fit_info, "head": name, "fit_seconds": fit_seconds, "transform_seconds": transform_seconds,
               "total_seconds": time.perf_counter()-started, "score_scale": scale,
               "native_export_max_error": native_error, "asset_sha256": file_hash(path),
               "asset_uncompressed_bytes": expanded_bytes, "details": detail}
        records.append(row); dump(OUT/"fit_records.json", records); dump(OUT/"transform_records.json", transforms)
        print(json.dumps({"partition": directory.name, "head": name, "completed_learner_fits": len(records),
                          "fit_seconds": round(fit_seconds, 3), "total_seconds": round(row["total_seconds"], 3)}), flush=True)
    np.savez(directory/"scores.npz", held_indices=held_ids, fit_indices=fit_ids, names=np.array(NAMES),
             raw=raw[nfit:], normalized=normalized[nfit:], fitting_raw=raw[:nfit], fitting_normalized=normalized[:nfit],
             score_scales=np.array([r["score_scale"] for r in records[-5:]]))
    dump(directory/"metadata.json", {**fit_info, "learners": records[-5:], "model_fit_calls": 5, "PCA_fit_calls": 2, "PLS_fit_calls": 2})
    return raw[nfit:], normalized[nfit:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-protocol", action="store_true", help="Check registration without reading any data or fitting")
    args = parser.parse_args()
    if not PROTOCOL.exists():
        raise FileNotFoundError("The independently registered protocol must exist before fitting")
    if args.check_protocol:
        print(json.dumps({"registered_protocol": str(PROTOCOL), "sha256": file_hash(PROTOCOL), "planned_learner_fits": 60})); return
    if (OUT/"protocol.json").exists():
        raise FileExistsError("Preserve this finite study; reruns require an independently named output")
    started = time.perf_counter(); OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"registered_protocol.md").write_bytes(PROTOCOL.read_bytes())
    protocol = dict(protocol_sha256=file_hash(PROTOCOL), data_scope=["train"], validation_access=False, test_access=False,
                    fixed_heads=NAMES, planned_model_fit_calls=60, planned_PCA_fit_calls=24, planned_PLS_fit_calls=24,
                    outer_seed=OUTER_SEED, inner_seed=INNER_SEED, group_epochs=14, conditional_pair_epochs=27,
                    final_full_model_fits=0, routing_rule_candidates=3, thresholds=[.2, .4],
                    interpretation="Nested gating conditional on historically selected base recipes, weights and hypothesis; not full-search unbiased")
    dump(OUT/"protocol.json", protocol)
    source_names = ["selective_routing.py", "answer_group_experiment.py", "answer_group_model.py", "kernel_rank.py",
                    "quantitative_model.py", "quantitative_experiment.py", "condition_experiment.py", "condition_model.py",
                    "condition_features.py", "conditional_pair_experiment.py", "conditional_pair.py", "text_features.py",
                    "selection.py", "tree_model.py", "common.py", "experiment_utils.py",
                    "runs/quantitative_experiments/repair_precision.py"]
    source_hashes = {name: file_hash(ROOT/name) for name in source_names}
    dump(OUT/"source_hashes.json", source_hashes)
    dump(OUT/"data_hashes.json", {p.relative_to(DATA).as_posix(): file_hash(p) for p in
        [DATA/"gsm8k_train.jsonl", DATA/"candidates/gsm8k_train.jsonl", DATA/"candidates/gsm8k_train.npz"]})
    dump(OUT/"environment.json", dict(python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__,
                                      threads=4, command="python selective_routing.py", dependencies="requirements-research.txt"))
    bank, emb, labels, hard = development_data("train"); n = len(bank)
    valid = answer_valid(bank); fold = folds_for(n, OUTER_SEED); ids = np.array([q["id"] for q in bank])
    old_f, old_info = historical_reference(bank, labels, valid, hard, fold); dump(OUT/"historical_reference.json", old_info)
    np.savez(OUT/"fold_assignment.npz", ids=ids, fold=fold, labels=labels, valid=valid, hard=hard)
    raw = np.full((n, 4, 5), np.nan); normalized = raw.copy(); nested = np.full((n, 4), np.nan)
    nested_routed = np.zeros(n, bool); records = []; transforms = []; outer_records = []
    with threadpool_limits(limits=4):
        data, feature_times = prepare(bank, emb, labels, hard); dump(OUT/"feature_cost.json", feature_times)
        for f in range(3):
            fit_ids = np.flatnonzero(fold != f); held_ids = np.flatnonzero(fold == f)
            inner_fold = folds_for(len(fit_ids), INNER_SEED+100*f)
            inner_raw = np.full((len(fit_ids), 4, 5), np.nan); inner_normalized = inner_raw.copy()
            for j in range(3):
                inner_fit = fit_ids[inner_fold != j]; inner_held = fit_ids[inner_fold == j]
                r, z = fit_partition(data, inner_fit, inner_held, INNER_SEED+1000*f+j,
                                     OUT/f"outer{f}_inner{j}", records, transforms)
                inner_raw[inner_fold == j] = r; inner_normalized[inner_fold == j] = z
                if f == 0 and j == 0:
                    dump(OUT/"cost_probe.json", dict(reused_as_first_planned_partition=True, extra_model_fits=0,
                        model_fit_calls=5, seconds=sum(r["total_seconds"] for r in records),
                        projected_60_fit_seconds=12*sum(r["total_seconds"] for r in records)))
            assert np.isfinite(inner_normalized).all()
            inner_f, inner_g = mixtures(inner_normalized)
            choice = choose_rule([bank[i] for i in fit_ids], inner_f, inner_g, labels[fit_ids], hard[fit_ids])
            choice.update(outer_fold=f, fit_indices_sha256=array_hash(fit_ids.astype(np.int64)),
                          selected_before_outer_fit_and_correctness=True)
            # This write precedes even fitting the models used for outer scoring.
            dump(OUT/f"outer{f}_inner_choice.json", choice)
            choice_hash = file_hash(OUT/f"outer{f}_inner_choice.json")
            np.savez(OUT/f"outer{f}_inner_oof.npz", ids=ids[fit_ids], indices=fit_ids, inner_fold=inner_fold,
                     raw=inner_raw, normalized=inner_normalized, names=np.array(NAMES), F=inner_f, G=inner_g,
                     labels=labels[fit_ids], valid=valid[fit_ids], hard=hard[fit_ids])
            r, z = fit_partition(data, fit_ids, held_ids, INNER_SEED+1000*f+100,
                                 OUT/f"outer{f}_refit", records, transforms)
            raw[held_ids] = r; normalized[held_ids] = z
            outer_f, outer_g = mixtures(z)
            routed_scores, mask = apply_rule([bank[i] for i in held_ids], outer_f, outer_g, hard[held_ids], choice["selected"])
            nested[held_ids] = routed_scores; nested_routed[held_ids] = mask
            row = dict(fold=f, inner_choice=choice["selected"], inner_choice_sha256=choice_hash,
                matched_F=metrics(select([bank[i] for i in held_ids], outer_f, "mean"), labels[held_ids], hard[held_ids]),
                nested=metrics(select([bank[i] for i in held_ids], routed_scores, "mean"), labels[held_ids], hard[held_ids]),
                routed_hard=int(mask.sum()))
            outer_records.append(row); dump(OUT/"outer_records.json", outer_records); print(json.dumps(row), flush=True)
    assert len(records) == 60 and len(transforms) == 48
    assert np.isfinite(normalized).all() and np.isfinite(nested).all()
    assert all(file_hash(ROOT/name) == digest for name, digest in source_hashes.items())
    f, g = mixtures(normalized)
    final_choice = choose_rule(bank, f, g, labels, hard)
    final_z, final_routed = apply_rule(bank, f, g, hard, final_choice["selected"])
    reference_choice = select(bank, f, "mean"); historical_choice = select(bank, old_f, "mean")
    nested_choice = select(bank, nested, "mean"); final_indices = select(bank, final_z, "mean")
    nested_pairs = paired(nested_choice, reference_choice, labels, hard, fold)
    historical_pairs = paired(final_indices, historical_choice, labels, hard, fold)
    final_matched_pairs = paired(final_indices, reference_choice, labels, hard, fold)
    training_gate = dict(passed=bool(historical_pairs[0]["gain"] >= 8 and all(r["gain"] > 0 for r in historical_pairs[1:])),
                         required_pooled_gain=8, pooled_gain=historical_pairs[0]["gain"],
                         fold_gains=[r["gain"] for r in historical_pairs[1:]], reference="historical F=1778")
    nested_gate = dict(passed=bool(nested_pairs[0]["gain"] > 0 and all(r["gain"] >= 0 for r in nested_pairs[1:])),
                       pooled_gain=nested_pairs[0]["gain"], fold_gains=[r["gain"] for r in nested_pairs[1:]],
                       reference="fresh matched fixed-epoch F")
    eligible_validation = training_gate["passed"] and nested_gate["passed"]
    final_choice.update(training_gate=training_gate, nested_gate=nested_gate, selected_before_validation=True,
                        interpretation="Final rule/threshold selected on newly generated outer OOF; this is tuning evidence")
    dump(OUT/"selected_config.json", final_choice)
    summary = dict(selected=final_choice["selected"], model_fit_calls=60, training_configurations=1,
        transform_fit_calls={"PCA": 24, "PLS": 24, "total": 48},
        fit_accounting={"inner_base_learners": 45, "outer_base_learners": 15, "full_training_base_learners": 0,
                        "probe_additional_fits": 0, "learned_projections": 48, "meta_learner_fits": 0,
                        "rule_comparisons": 12, "routing_rules_per_selection": 3, "selection_collections": 4},
        historical_F=old_info["metrics"], matched_F=metrics(reference_choice, labels, hard),
        fixed_G=metrics(select(bank, g, "mean"), labels, hard),
        final_oof_selected_rule=metrics(final_indices, labels, hard), nested_oof_metrics=metrics(nested_choice, labels, hard),
        nested_paired_vs_matched_F=nested_pairs, final_paired_vs_historical_F=historical_pairs,
        final_paired_vs_matched_F=final_matched_pairs, training_gate=training_gate, nested_gate=nested_gate,
        validation_status="eligible_pending_root_authorization" if eligible_validation else "not_evaluated_training_gate_failed",
        validation_evaluations=0, test_access=False, total_seconds=time.perf_counter()-started,
        learner_fit_seconds=sum(r["fit_seconds"] for r in records),
        transform_seconds=sum(r["seconds"] for r in transforms),
        protocol_sha256=file_hash(PROTOCOL), source_hashes_sha256=file_hash(OUT/"source_hashes.json"),
        conclusion="Both predeclared gates passed; no validation run here" if eligible_validation else "At least one predeclared gate failed; stop this route without validation",
        interpretation=protocol["interpretation"])
    np.savez(OUT/"oof_logits.npz", ids=ids, labels=labels, valid=valid, hard=hard, fold=fold, names=np.array(NAMES),
             raw=raw, normalized=normalized, F=f, G=g, historical_F=old_f,
             nested=nested, selected=final_z, nested_routed=nested_routed, selected_routed=final_routed)
    dump(OUT/"summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
