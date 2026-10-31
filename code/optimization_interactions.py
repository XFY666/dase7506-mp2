"""Registered D1+D2 transfer to fixed group/PLS models; explicit --run only.

Top-level inference helpers require NumPy only. Training dependencies are
imported inside the execution path, and references never enter feature maps.
"""
from __future__ import annotations
import os
import sys
sys.dont_write_bytecode = True
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "4"
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import zipfile
import numpy as np

from answer_group_model import group_weights, features_for_groups, group_model_scores
from common import DATA
from condition_features import extract_features as condition_features
import cross_candidate_features as cross
from quantitative_model import answer_valid, stable_base_features, stable_center
from selection import select
from text_features import extract_features
from tree_model import tree_scores

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "optimization_interactions"
SEEDS = (7506, 17001, 17002)
FOLD_SEEDS = (20261027, 20261028)
CROSS_INDICES = cross.FEATURE_GROUPS["D1"] + cross.FEATURE_GROUPS["D2"]
CONFIGS = (
    dict(name="A_base", family="A", with_cross=False, member_seeds=SEEDS),
    dict(name="A_D1D2", family="A", with_cross=True, member_seeds=SEEDS),
    dict(name="Q_base", family="Q", with_cross=False, leaves=15),
    dict(name="Q_D1D2", family="Q", with_cross=True, leaves=15),
    dict(name="C_base", family="C", with_cross=False, leaves=31),
    dict(name="C_D1D2", family="C", with_cross=True, leaves=31),
)
SOURCES = ("optimization_interactions.py", "cross_candidate_features.py", "answer_group_model.py",
    "answer_group_experiment.py", "condition_model.py", "condition_experiment.py", "condition_features.py",
    "quantitative_model.py", "quantitative_features.py", "text_features.py", "tree_model.py", "common.py",
    "selection.py", "advanced_neural.py", "experiment_utils.py")
STARTED = None


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(x):
    x = np.ascontiguousarray(x)
    return hashlib.sha256(str(x.dtype).encode() + str(x.shape).encode() + x.tobytes()).hexdigest()


def progress(stage, count, detail):
    path = OUT / "progress.json"
    temp = path.with_suffix(".json.tmp")
    dump(temp, {"stage": stage, "completed_fits": count, "planned_fits": 60, "detail": detail,
        "elapsed_seconds": time.perf_counter() - STARTED if STARTED is not None else 0.,
        "updated_utc": datetime.now(timezone.utc).isoformat()})
    os.replace(temp, path)


def feature_blocks(bank):
    original = extract_features(bank)
    stable = stable_base_features(bank)
    conditions = condition_features(bank)
    directed = cross.extract_features(bank)[:, :, CROSS_INDICES]
    assert directed.shape == (len(bank), 4, 24)
    return original, stable, conditions, directed


def group_features(embeddings, original, directed, weights, with_cross):
    text = np.concatenate((original, directed), axis=-1) if with_cross else original
    return features_for_groups(embeddings, text, weights, "group")


def tree_features(embeddings, stable, conditions, directed, valid, projection, family, with_cross):
    e = np.asarray(embeddings, np.float32)
    projected = (e - projection["pls_mean"]) @ projection["pls_rotations"]
    parts = [stable]
    if family == "C":
        parts.append(conditions)
    parts.extend((projected, stable_center(projected, valid)))
    if with_cross:
        parts.append(directed)
    return np.concatenate(parts, axis=-1).astype(np.float32)


def score_asset(bank, embeddings, asset, blocks=None):
    if not bank:
        return np.empty((0, 4), np.float64)
    original, stable, conditions, directed = blocks if blocks is not None else feature_blocks(bank)
    with_cross = bool(int(asset["with_cross"]))
    family = ("A", "Q", "C")[int(asset["family_code"])]
    if family == "A":
        text = np.concatenate((original, directed), axis=-1) if with_cross else original
        score = group_model_scores(bank, embeddings, asset, "group", text)
    else:
        x = tree_features(embeddings, stable, conditions, directed, answer_valid(bank), asset, family, with_cross)
        assert x.shape[-1] == int(asset["feature_width"])
        score = tree_scores(x.reshape(-1, x.shape[-1]), asset).reshape(-1, 4)
    return score


def predict(bank, arrays, checkpoint):
    directory = Path(checkpoint)
    paths = sorted(directory.glob("seed_*.npz"))
    if not paths:
        paths = [directory / "model.npz"]
    blocks = feature_blocks(bank)
    score = np.zeros((len(bank), 4), np.float64)
    for path in paths:
        with np.load(path, allow_pickle=False) as a:
            asset = {key: a[key] for key in a.files}
        score += score_asset(bank, arrays["embeddings"], asset, blocks) / float(asset["score_scale"])
    return select(bank, score / len(paths), "mean")


def register():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "protocol.json"
    hashes = {name: digest(ROOT / name) for name in SOURCES}
    if path.exists():
        protocol = json.loads(path.read_text(encoding="utf-8"))
        assert hashes == protocol["source_sha256"], "Registered implementation changed; preserve registration and resolve before run"
        return protocol
    protocol = {
        "purpose": "Transfer the previously useful D1+D2 candidate relations to stronger fixed learners, with matched base controls",
        "configs": CONFIGS, "fold_seeds": FOLD_SEEDS, "fold_mapping": "fold[rng.permutation(allN)] = arange(N)%3",
        "planned_model_fits": 60, "group_fits": 36, "tree_fits": 24, "PLS_fits": 6,
        "group_recipe": {"hidden": 128, "objective": "pairwise", "dropout": .2, "l2": .01,
            "epochs": 14, "batch_questions": 128, "learning_rate": .001, "seeds": SEEDS,
            "shuffle_dropout_seed": "initialization_seed+1", "training_rows": "hard outer-fitting questions only",
            "normalizer": "all valid answer groups from outer-fitting questions only",
            "original_feature_input": "original extract_features75, identical to historical A; expanded input appends24 D1/D2 before groupmean/max/std"},
        "tree_recipe": {"Q_leaves": 15, "C_leaves": 31, "rounds": 300, "learning_rate": .05,
            "l2": 10., "min_samples_leaf": 30, "seed": 20261015, "early_stopping": False,
            "training_rows": "valid candidates on hard outer-fitting questions; each question total weight4",
            "PLS": "one PLS8 per outer fit, shared across Q/Q+/C/C+; float32 labels and repaired float64-centered embeddings/labels; fit eligible outer-training rows only",
            "feature_order": "stablebase75,[condition54 forC],projected8,valid-centered projected8,[D1D2 appended24]"},
        "cross_feature_indices": CROSS_INDICES, "cross_feature_names": [cross.FEATURE_NAMES[i] for i in CROSS_INDICES],
        "expected_feature_widths": {"A_base": 1252, "A_D1D2": 1324, "Q_base": 91, "Q_D1D2": 115, "C_base": 145, "C_D1D2": 169},
        "normalization": "Each scorer scaled by outer-fitting valid candidate scores only; groupmembers normalizedindividually thenfixed3seedaverage; all4candidatecentering, valid-positionstd",
        "aggregation": "answer-group mean, zero consensus bonus",
        "selection": "none; report all6configs, 2repeats and matchedwith/withoutD1D2 differences",
        "statistical_scope": "Historically informed D1D2 hypothesis; repeated questions are not independent doubled sample size",
        "validation_access": False, "test_access": False, "threads": 4, "source_sha256": hashes,
    }
    dump(path, protocol)
    progress("queued", 0, "D1+D2强模型迁移：60次拟合已登记，等待队列启动")
    return protocol


def fit_group(x, weights, good, valid, seed):
    import answer_group_experiment as group
    config = dict(representation="group", objective="pairwise", dropout=.2, hidden=128, l2=.01)
    model = group.initialize(x.shape[-1], 128, seed)
    momentum = [np.zeros_like(value) for value in model]
    variance = [np.zeros_like(value) for value in model]
    rng = np.random.default_rng(seed + 1)
    step = 0
    for _epoch in range(14):
        order = rng.permutation(len(x))
        for start in range(0, len(x), 128):
            ids = order[start:start + 128]
            _, gradients = group.loss_gradient(x[ids], weights[ids], good[ids], valid[ids], model, config, rng)
            step += 1
            for i, gradient in enumerate(gradients):
                momentum[i] = .9 * momentum[i] + .1 * gradient
                variance[i] = .999 * variance[i] + .001 * gradient * gradient
                model[i] -= .001 * (momentum[i] / (1 - .9**step)) / (np.sqrt(variance[i] / (1 - .999**step)) + 1e-8)
    assert all(np.isfinite(value).all() for value in model)
    return model, step


def io_audit():
    accessed = set()
    def audit(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path.name.startswith(("gsm8k_validation", "gsm8k_test")):
            raise RuntimeError("Validation/test access forbidden")
        if DATA.resolve() in path.parents:
            accessed.add(path.relative_to(DATA.resolve()).as_posix())
        mode, flags = args[1], args[2] or 0
        write = isinstance(mode, str) and any(c in mode for c in "wax+") or bool(flags & (os.O_WRONLY | os.O_RDWR))
        if write and path != OUT.resolve() and OUT.resolve() not in path.parents:
            raise RuntimeError(f"Write outside owned output: {path}")
    sys.addaudithook(audit)
    return accessed


def run():
    global STARTED
    from scipy.special import logsumexp
    from threadpoolctl import threadpool_limits
    from advanced_neural import development
    from experiment_utils import metrics
    import answer_group_experiment as group
    import condition_experiment as condition
    from tree_model import export_trees
    protocol = register()
    if (OUT / "run_started.json").exists():
        raise FileExistsError("Recorded execution already started; do not overwrite")
    STARTED = time.perf_counter()
    accessed = io_audit()
    dump(OUT / "run_started.json", {"utc": datetime.now(timezone.utc).isoformat(), "source_sha256": protocol["source_sha256"]})
    progress("preparing", 0, "核对仅训练数据，提取原始/条件/D1+D2特征")
    dump(OUT / "training_input_hashes.json", {str(p.relative_to(ROOT)): digest(p) for p in
        (DATA / "gsm8k_train.jsonl", DATA / "candidates/gsm8k_train.jsonl", DATA / "candidates/gsm8k_train.npz")})
    dump(OUT / "feature_checks.json", cross.self_checks())
    bank, embeddings, labels, valid, hard = development("train")
    labels_float = labels.astype(np.float32)
    centered = stable_center(labels_float[hard], valid[hard])[valid[hard]]
    assert centered.dtype == np.float32 and centered.min() < 0 < centered.max()
    weights = group_weights(bank)
    group_valid = weights.sum(axis=2) > 0
    good = np.einsum("ngc,nc->ng", weights, labels) > 0
    blocks = feature_blocks(bank)
    original, stable, conditions, directed = blocks
    group_designs = {flag: group_features(embeddings, original, directed, weights, flag) for flag in (False, True)}
    assert np.array_equal(group_designs[False], group.make_features(embeddings, original, weights, "group"))
    assert np.array_equal(group_designs[True], group.make_features(embeddings, np.concatenate((original, directed), -1), weights, "group"))
    folds = np.empty((2, len(bank)), np.int8)
    for r, seed in enumerate(FOLD_SEEDS):
        folds[r, np.random.default_rng(seed).permutation(len(bank))] = np.arange(len(bank)) % 3
    raw = np.full((2, len(bank), 4, 6), np.nan, np.float64)
    normalized = np.full_like(raw, np.nan)
    group_raw = np.full((2, len(bank), 4, 2, 3), np.nan, np.float32)
    group_normalized = np.full_like(group_raw, np.nan)
    records, transformations = [], []
    with threadpool_limits(limits=4):
        for repeat in range(2):
            for fold in range(3):
                fitting = np.flatnonzero(folds[repeat] != fold)
                held = np.flatnonzero(folds[repeat] == fold)
                partition = OUT / f"repeat{repeat}_fold{fold}"
                partition.mkdir()
                np.savez(partition / "partition.npz", fitting=fitting, held=held)
                projection, pinfo = condition.fit_pls(embeddings, labels_float, valid, hard, fitting)
                assert np.any(projection["pls_rotations"] != 0)
                np.savez(partition / "shared_pls.npz", **projection)
                transformations.append({"repeat": repeat, "fold": fold, "kind": "PLS8", "info": pinfo,
                    "fit_indices_sha256": array_hash(fitting), "shared_by": [c["name"] for c in CONFIGS if c["family"] != "A"],
                    "asset_sha256": digest(partition / "shared_pls.npz")})
                dump(OUT / "transform_records.json", transformations)
                for column, config in enumerate(CONFIGS):
                    folder = partition / config["name"]
                    folder.mkdir()
                    seeds = SEEDS if config["family"] == "A" else (20261015,)
                    if config["family"] == "A":
                        features = group_designs[config["with_cross"]]
                        mean, std = group.fit_statistics(features[fitting], group_valid[fitting])
                        x = group.normalized(features, mean, std)
                    else:
                        x = tree_features(embeddings, stable, conditions, directed, valid, projection, config["family"], config["with_cross"])
                    assert x.shape[-1] == protocol["expected_feature_widths"][config["name"]]
                    outputs, normalized_outputs = [], []
                    for member, seed in enumerate(seeds):
                        tick = time.perf_counter()
                        if config["family"] == "A":
                            active = fitting[hard[fitting]]
                            model, steps = fit_group(x[active], weights[active], good[active], group_valid[active], seed)
                            score = group.candidate_scores(group.scores(x, model), weights, "group")
                            scale = group.scale_scores(score[fitting], valid[fitting])
                            asset = dict(w1=model[0], b1=model[1], w2=model[2], skip=model[3], mean=mean, scale=std)
                            fit_detail = {"epochs": 14, "steps": steps, "seed": seed}
                        else:
                            estimator = condition.fit_tree(x, labels_float, valid, hard, fitting, config["leaves"], 300)
                            score = estimator.decision_function(x.reshape(-1, x.shape[-1])).reshape(-1, 4)
                            # Preserve each original family's scale definition.
                            scale = group.scale_scores(score[fitting], valid[fitting]) if config["family"] == "Q" else condition.scale_scores(score, valid, fitting)
                            asset = {**export_trees(estimator), **projection}
                            fit_detail = {"parameters": condition.parameters(config["leaves"], 300)}
                        asset.update(family_code=np.array(("A", "Q", "C").index(config["family"]), np.int32),
                            with_cross=np.array(config["with_cross"], np.int8), feature_width=np.array(x.shape[-1], np.int32),
                            score_scale=np.array(scale), interaction_version=np.array(1, np.int32))
                        sample = held[:32]
                        replay = score_asset([bank[int(i)] for i in sample], embeddings[sample], asset, tuple(b[sample] for b in blocks))
                        error = float(np.max(np.abs(replay - score[sample])))
                        assert error < 2e-5
                        path = folder / (f"seed_{seed}.npz" if config["family"] == "A" else "model.npz")
                        np.savez(path, **asset)
                        with zipfile.ZipFile(path) as archive:
                            expanded = sum(item.file_size for item in archive.infolist())
                        z = score / scale
                        outputs.append(score[held]); normalized_outputs.append(z[held])
                        if column < 2:
                            group_raw[repeat, held, :, column, member] = score[held]
                            group_normalized[repeat, held, :, column, member] = z[held]
                        records.append({"repeat": repeat, "fold": fold, "config": config["name"], "seed": seed,
                            "fit_questions": len(fitting), "fit_hard_questions": int(hard[fitting].sum()),
                            "fit_indices_sha256": array_hash(fitting), "held_indices_sha256": array_hash(held),
                            "feature_width": x.shape[-1], "score_scale": scale, "fit_detail": fit_detail,
                            "seconds": time.perf_counter() - tick, "asset_sha256": digest(path), "asset_uncompressed_bytes": expanded,
                            "export_replay_max_error": error})
                        dump(OUT / "fit_records.json", records)
                        progress("fitting", len(records), f"划分{repeat+1}/折{fold+1}：{config['name']}，种子{seed}完成")
                        print(json.dumps({"completed_fits": len(records), "config": config["name"], "repeat": repeat, "fold": fold, "seed": seed}), flush=True)
                    raw[repeat, held, :, column] = np.stack(outputs).mean(axis=0, dtype=np.float64)
                    normalized[repeat, held, :, column] = np.stack(normalized_outputs).mean(axis=0, dtype=np.float64)
    assert len(records) == 60 and len(transformations) == 6
    assert np.isfinite(raw).all() and np.isfinite(normalized).all()
    choices = np.empty((2, len(bank), 6), np.int8)
    losses = np.empty((2, len(bank), 6), np.float64)
    results = []
    for repeat in range(2):
        rows = {}
        for column, config in enumerate(CONFIGS):
            score = normalized[repeat, :, :, column]
            chosen = np.asarray(select(bank, score, "mean"), np.int8)
            choices[repeat, :, column] = chosen
            grouped = np.einsum("ngc,nc->ng", weights.astype(np.float64), score)
            loss = logsumexp(np.where(group_valid, grouped, -np.inf), axis=1) - (grouped * good).sum(axis=1)
            losses[repeat, :, column] = loss
            result = {**metrics(chosen, labels, hard), "group_ce": float(loss[hard].mean())}
            correct = labels[np.arange(len(bank)), chosen]
            result["folds"] = [{"fold": fold, "hard_n": int((mask := hard & (folds[repeat] == fold)).sum()),
                "hard_correct": int(correct[mask].sum()), "group_ce": float(loss[mask].mean())} for fold in range(3)]
            if column % 2:
                reference = labels[np.arange(len(bank)), choices[repeat, :, column-1]]
                result["vs_matched_base"] = {"repairs": int((correct & ~reference & hard).sum()),
                    "regressions": int((~correct & reference & hard).sum()), "gain": int(correct[hard].sum()-reference[hard].sum()),
                    "group_ce_difference": result["group_ce"] - rows[CONFIGS[column-1]["name"]]["group_ce"]}
            rows[config["name"]] = result
        results.append({"repeat": repeat, "fold_seed": FOLD_SEEDS[repeat], "results": rows})
    np.savez(OUT / "oof_logits.npz", ids=np.array([row["id"] for row in bank]), raw=raw, normalized=normalized,
        group_raw_members=group_raw, group_normalized_members=group_normalized, config_names=np.array([c["name"] for c in CONFIGS]),
        labels=labels, valid=valid, hard=hard, folds=folds, fold_seeds=np.array(FOLD_SEEDS), choices=choices, group_losses=losses)
    dump(OUT / "oof_schema.json", {"raw_normalized": "[2,N,4,6] float64;group configs arefixed3normalizedseedmean",
        "config_order": [c["name"] for c in CONFIGS], "group_raw_normalized_members": "[2,N,4,2,3] float32;3seeds inprotocolorder",
        "choices_group_losses": "[2,N,6]", "labels_valid": "[N,4]", "folds": "[2,N]", "scale_location": "individual assets andfit_records"})
    assert all(digest(ROOT / path) == expected for path, expected in protocol["source_sha256"].items())
    summary = {"status": "complete", "results": results, "model_fit_calls": len(records), "PLS_fit_calls": len(transformations),
        "seconds": time.perf_counter()-STARTED, "validation_accessed": False, "test_accessed": False,
        "data_files_read": sorted(accessed), "source_hashes_unchanged": True,
        "max_asset_bytes_by_config": {c["name"]: max(r["asset_uncompressed_bytes"] for r in records if r["config"]==c["name"])*(3 if c["family"]=="A" else 1) for c in CONFIGS},
        "asset_scope": "uncompressedmemberarraypayload includingNPYheaders; confignotincluded;outerfitsnotfinalfulltraining deployment",
        "caveat": "Matched fixedfeature comparisons, historically motivated hypothesis and repeatedquestions;noindependentdoubledsamplesize ormodelpromotion"}
    dump(OUT / "summary.json", summary)
    progress("complete", 60, "D1+D2强模型迁移60次拟合完成；结果已保存，未访问验证或测试")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--register-only", action="store_true")
    action.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.register_only:
        registered = register()
        print(json.dumps({"status": "registered_not_started", "planned_fits": registered["planned_model_fits"], "command": "python optimization_interactions.py --run"}))
    else:
        try:
            run()
        except Exception as error:
            if STARTED is not None:
                file = OUT / "fit_records.json"
                count = len(json.loads(file.read_text())) if file.exists() else 0
                progress("failed", count, f"{type(error).__name__}: {error}")
            raise
