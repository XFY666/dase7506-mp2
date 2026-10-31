"""Training-only fixed weak process supervision controls: 5 x 3 x 6 fits."""
from __future__ import annotations
import os
import sys
sys.dont_write_bytecode = True
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "4"
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import platform
import re
import time
import zipfile
import numpy as np
from scipy.special import expit, logsumexp
from threadpoolctl import threadpool_limits
import answer_group_experiment as group
from advanced_neural import development
from common import DATA, parse_answer, read_jsonl
from experiment_utils import dump, metrics
from selection import select
from text_features import extract_features
from optimization_process_model import member_scores, ensemble_scores

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "optimization_process"
FOLD_SEEDS = (20261027, 20261028)
SEEDS = (7506, 17001, 17002)
CONFIGS = (
    dict(name="control_allq", auxiliary=0., hard_aux_only=False, coverage_weight=False),
    dict(name="coverage01_allq", auxiliary=.1, hard_aux_only=False, coverage_weight=False),
    dict(name="coverage03_allq", auxiliary=.3, hard_aux_only=False, coverage_weight=False),
    dict(name="coverage01_hardq", auxiliary=.1, hard_aux_only=True, coverage_weight=False),
    dict(name="coverage_weight", auxiliary=0., hard_aux_only=False, coverage_weight=True),
)
HIDDEN = 128
EPOCHS = 14
L2 = .01
DROPOUT = .2
BATCH = 128
STARTED = None
NUMBER = r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:\s*/\s*[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+))?"
NUM_RE = re.compile(r"(?<![\w.])(" + NUMBER + r")(?![\w.])")
EXACT_NUM = re.compile(NUMBER)
ANNOTATION = re.compile(r"<<([^<>]*)>>")
FRACTION = re.compile(r"\\(?:dfrac|tfrac|frac)\s*\{\s*(" + NUMBER + r")\s*\}\s*\{\s*(" + NUMBER + r")\s*\}")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode() + value.tobytes()).hexdigest()


def progress(stage, completed, detail):
    path = OUT / "progress.json"
    temporary = path.with_suffix(".json.tmp")
    dump(temporary, {"stage": stage, "completed_fits": completed, "planned_fits": 90, "detail": detail,
        "elapsed_seconds": time.perf_counter() - STARTED if STARTED is not None else 0.,
        "updated_utc": datetime.now(timezone.utc).isoformat()})
    os.replace(temporary, path)


def audit_io():
    accesses = set()
    allowed = OUT.resolve()
    def audit(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path.name.startswith(("gsm8k_validation", "gsm8k_test")):
            raise RuntimeError(f"Forbidden split: {path}")
        if DATA.resolve() in path.parents:
            accesses.add(path.relative_to(DATA.resolve()).as_posix())
        mode, flags = args[1], args[2] or 0
        write = isinstance(mode, str) and any(c in mode for c in "wax+") or bool(flags & (os.O_WRONLY | os.O_RDWR))
        if write and path != allowed and allowed not in path.parents:
            raise RuntimeError(f"Persistent write outside experiment: {path}")
    sys.addaudithook(audit)
    return accesses


def normalized_fractions(text):
    return FRACTION.sub(lambda m: m.group(1) + "/" + m.group(2), str(text))


def literal_value(text):
    token = normalized_fractions(text).strip().replace(" ", "")
    if not EXACT_NUM.fullmatch(token):
        return None
    return parse_answer("#### " + token)


def number_values(text):
    return {value for match in NUM_RE.finditer(normalized_fractions(text))
            if (value := literal_value(match.group(1))) is not None}


def intermediate_values(reference):
    annotations = ANNOTATION.findall(reference)
    values = set()
    # Exclude the final annotation by position, even if its RHS is unsupported.
    for expression in annotations[:-1]:
        if "=" in expression:
            value = literal_value(expression.rsplit("=", 1)[1])
            if value is not None:
                values.add(value)
    return values, len(annotations)


def make_targets(bank, references, weights):
    coverage = np.zeros((len(bank), 4), np.float32)
    observed = np.zeros(len(bank), bool)
    unique_counts = np.zeros(len(bank), np.int16)
    annotation_counts = np.zeros(len(bank), np.int16)
    for i, (row, reference) in enumerate(zip(bank, references)):
        assert row["id"] == reference["id"] and row["question"] == reference["question"]
        values, count = intermediate_values(reference["answer"])
        unique_counts[i], annotation_counts[i] = len(values), count
        if values:
            observed[i] = True
            for j, candidate in enumerate(row["candidates"]):
                coverage[i, j] = len(values & number_values(candidate["text"])) / len(values)
    return coverage, np.einsum("ngc,nc->ng", weights, coverage), observed, unique_counts, annotation_counts


def initialize(dim, seed):
    model = group.initialize(dim, HIDDEN, seed)
    rng = np.random.default_rng(seed + 137)
    return model + [rng.normal(0, 1 / np.sqrt(HIDDEN), HIDDEN).astype(np.float32), np.array(0., np.float32)]


def loss_gradient(x, labels, valid, target, observed, model, config, rng=None):
    w1, b1, w2, skip, aux_w, aux_b = model
    hidden = np.tanh(x @ w1 + b1)
    mask = 1.
    if rng is not None:
        mask = (rng.random(hidden.shape) >= DROPOUT).astype(hidden.dtype) / (1 - DROPOUT)
    used = hidden * mask
    score = used @ w2 + x @ skip
    pairs = labels[:, :, None] * (valid & ~labels)[:, None, :]
    pair_count = pairs.sum(axis=(1, 2))
    hard = pair_count > 0
    qweight = hard.astype(x.dtype)
    if config["coverage_weight"]:
        correct_coverage = (target * labels).sum(axis=1)
        qweight *= np.where(observed, .5 + .5 * correct_coverage, 1.)
    weight = pairs / np.maximum(pair_count[:, None, None], 1)
    weight = weight * qweight[:, None, None] / max(float(qweight.sum()), 1e-12)
    difference = score[:, None, :] - score[:, :, None]
    main_loss = float((np.logaddexp(0, difference) * weight).sum())
    probability = expit(difference) * weight
    dscore = (probability.sum(axis=1) - probability.sum(axis=2)).astype(x.dtype)
    dh = dscore[:, :, None] * w2
    aux_loss = 0.
    aux_w_gradient = np.zeros_like(aux_w)
    aux_b_gradient = np.zeros_like(aux_b)
    strength = config["auxiliary"]
    if strength:
        active = observed & (hard if config["hard_aux_only"] else True)
        aux_mask = valid & active[:, None]
        counts = aux_mask.sum(axis=1)
        aw = aux_mask / np.maximum(counts[:, None], 1) / max(int((counts > 0).sum()), 1)
        pred = expit(used @ aux_w + aux_b)
        residual = pred - target
        aux_loss = float(.5 * strength * (aw * residual * residual).sum())
        da = (strength * aw * residual * pred * (1 - pred)).astype(x.dtype)
        dh += da[:, :, None] * aux_w
        aux_w_gradient = used.reshape(-1, len(aux_w)).T @ da.ravel()
        aux_b_gradient = da.sum()
    dh *= mask * (1 - hidden * hidden)
    grads = [x.reshape(-1, x.shape[-1]).T @ dh.reshape(-1, len(w2)), dh.sum(axis=(0, 1)),
             used.reshape(-1, len(w2)).T @ dscore.ravel(), x.reshape(-1, x.shape[-1]).T @ dscore.ravel(),
             aux_w_gradient, np.asarray(aux_b_gradient, dtype=x.dtype)]
    loss = main_loss + aux_loss
    for index in (0, 2, 3):
        loss += .5 * L2 * float(np.sum(model[index] ** 2))
        grads[index] += L2 * model[index]
    if strength:
        loss += .5 * strength * L2 * float(np.sum(aux_w ** 2))
        grads[4] += strength * L2 * aux_w
    return float(loss), grads, {"main": main_loss, "auxiliary_weighted": aux_loss}


def fit(x, labels, valid, target, observed, config, seed):
    model = initialize(x.shape[-1], seed)
    moment = [np.zeros_like(p) for p in model]
    variance = [np.zeros_like(p) for p in model]
    rng = np.random.default_rng(seed + 1)
    step = 0
    curve = []
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(x))
        accumulated = np.zeros(3, np.float64)
        batches = 0
        for start in range(0, len(x), BATCH):
            ix = order[start:start + BATCH]
            loss, gradients, parts = loss_gradient(x[ix], labels[ix], valid[ix], target[ix], observed[ix], model, config, rng)
            step += 1
            for index, gradient in enumerate(gradients):
                moment[index] = .9 * moment[index] + .1 * gradient
                variance[index] = .999 * variance[index] + .001 * gradient * gradient
                model[index] -= .001 * (moment[index] / (1 - .9 ** step)) / (np.sqrt(variance[index] / (1 - .999 ** step)) + 1e-8)
            accumulated += (loss, parts["main"], parts["auxiliary_weighted"])
            batches += 1
        curve.append({"epoch": epoch, "mean_batch_objective": float(accumulated[0] / batches),
                      "mean_batch_main_loss": float(accumulated[1] / batches), "mean_batch_weighted_auxiliary": float(accumulated[2] / batches),
                      "steps": step, "batches": batches})
    assert all(np.isfinite(value).all() for value in model)
    return model, curve


def gradient_check():
    rng = np.random.default_rng(441)
    x = rng.normal(size=(3, 4, 5))
    labels = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]], bool)
    valid = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [1, 0, 0, 0]], bool)
    target = rng.random((3, 4))
    observed = np.array([True, False, True])
    model = [value.astype(np.float64) for value in initialize(5, 7506)]
    results = []
    for config in CONFIGS:
        _, grads, _ = loss_gradient(x, labels, valid, target, observed, model, config)
        errors = []
        for parameter, index in ((0, (1, 2)), (1, (2,)), (2, (3,)), (3, (1,)), (4, (3,)), (5, ())):
            delta = 1e-5
            model[parameter][index] += delta
            hi = loss_gradient(x, labels, valid, target, observed, model, config)[0]
            model[parameter][index] -= 2 * delta
            lo = loss_gradient(x, labels, valid, target, observed, model, config)[0]
            model[parameter][index] += delta
            errors.append(abs((hi - lo) / (2 * delta) - float(grads[parameter][index])))
        assert max(errors) < 1e-7, (config, errors)
        results.append({"configuration": config["name"], "max_absolute_error": max(errors)})
    # With all-hard batches and auxiliary disabled, exactly the old objective.
    config = dict(representation="group", objective="pairwise", dropout=0, l2=L2)
    weights = np.broadcast_to(np.eye(4), (2, 4, 4))
    old_loss, old_grad = group.loss_gradient(x[:2], weights, labels[:2], valid[:2], model[:4], config)
    new_loss, new_grad, _ = loss_gradient(x[:2], labels[:2], valid[:2], target[:2], observed[:2], model, CONFIGS[0])
    assert abs(old_loss - new_loss) < 1e-12
    assert max(float(np.max(np.abs(a - b))) for a, b in zip(old_grad, new_grad[:4])) < 1e-12
    assert intermediate_values("a <<2+3=5>> then <<5*2=10>> #### 10")[0] == {"5"}
    assert intermediate_values("one <<2+3=5>> #### 5")[0] == set()
    assert "1/2" in number_values(r"The intermediate value is \frac{1}{2} and 0.5.")
    return {"finite_difference": results, "hard_only_control_matches_prior_objective": True, "target_examples_passed": True}


def evaluate(bank, score, labels, hard, weights, group_valid, good):
    choice = np.asarray(select(bank, score, "mean"), dtype=np.int8)
    grouped = np.einsum("ngc,nc->ng", weights.astype(np.float64), score)
    loss = logsumexp(np.where(group_valid, grouped, -np.inf), axis=1) - (grouped * good).sum(axis=1)
    return choice, loss, {**metrics(choice, labels, hard), "group_ce": float(loss[hard].mean())}


def main():
    global STARTED
    if OUT.exists():
        raise FileExistsError("Preserve recorded process-supervision study")
    OUT.mkdir(parents=True)
    STARTED = time.perf_counter()
    accesses = audit_io()
    progress("preparing", 0, "登记过程弱监督定义；准备梯度与目标解析检查")
    sources = {Path(__file__).resolve(), ROOT / "optimization_process_model.py", ROOT.parent / "docs/optimization_protocol.md"}
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path and Path(path).resolve().parent == ROOT and Path(path).suffix == ".py":
            sources.add(Path(path).resolve())
    source_hashes = {os.path.relpath(path, ROOT).replace("\\", "/"): digest(path) for path in sorted(sources)}
    protocol = {"configs": CONFIGS, "fold_seeds": FOLD_SEEDS, "member_seeds": SEEDS, "planned_fits": 90,
        "fold_mapping": "fold[rng.permutation(all4755)] = arange(N)%3", "feature_width": 1252,
        "architecture": "original pooled answer-group features; tanh128 + linear skip; auxiliary sigmoid linear head on same dropout hidden activations",
        "epochs": EPOCHS, "batch_questions": BATCH, "dropout": DROPOUT, "l2": L2, "optimizer": "Adam .001 beta .9/.999 epsilon1e-8",
        "target": "Distinct canonical numeric literal RHS from all training-reference <<expr=RHS>> annotations except the last annotation; candidate lexical numeric presence coverage, then mean within answer group. Empty/unsupported reference intermediate set is unobserved.",
        "numeric_extraction": "Signed decimal/integer/fraction literal tokens; comma removal through official parser; simple LaTex fractions normalized; percent literals retain their written numeric value. No expression execution. Full candidate including final answer/headings is used, so overlap is only a weak proxy.",
        "main_loss": "Mean question-normalized valid correct/incorrect group pair logistic over hard questions in each minibatch; official final-answer labels unchanged",
        "auxiliary_loss": "lambda*.5 sigmoid squared-error, average within valid groups then equally across observed questions. hard_aux_only masks unanimous questions. lambda*L2 applies only to auxiliary output weights; ordinary L2 on mainW1/W2/skip.",
        "coverage_weight": "C3 only: hard-question main-loss weight .5+.5*correctgroup meancoverage if observed else1; normalize weights within batch; no auxiliary loss",
        "matched_control": "All configurations use identical all-question minibatch orders and dropout for a seed; easy questions contribute no pair term. More optimizer steps than historical hard-only A, so use the new lambda0 control for causal ablation.",
        "input_boundary": "No reference-derived quantity, target or mask enters features or inference assets other than learned weights. Only outer-fitting target slices enter losses. Outer-held labels used solely for final OOF reporting.",
        "normalization": "All outer-fitting valid group features; member score scale from outer-fitting valid candidate scores only; average all3 normalized members without seed selection",
        "selection": "none in this script; report every fixed config/repeat. Same questions repeated, no doubled independent sample size.",
        "validation_access": False, "test_access": False, "threads": 4, "source_sha256": source_hashes,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()}}
    dump(OUT / "protocol.json", protocol)
    dump(OUT / "gradient_checks.json", gradient_check())
    bank, emb, labels, valid, hard = development("train")
    references = read_jsonl(DATA / "gsm8k_train.jsonl")
    weights = group.groups(bank)
    group_valid = weights.sum(axis=2) > 0
    good = np.einsum("ngc,nc->ng", weights, labels) > 0
    assert np.all(good.sum(axis=1) == 1)
    coverage, target, observed, counts, annotations = make_targets(bank, references, weights)
    del references
    np.savez(OUT / "training_targets.npz", ids=np.array([row["id"] for row in bank]), candidate_coverage=coverage,
        group_coverage=target, observed=observed, intermediate_count=counts, annotation_count=annotations)
    dump(OUT / "target_summary.json", {"questions": len(bank), "observed": int(observed.sum()),
        "observed_hard": int((observed & hard).sum()), "observed_easy": int((observed & ~hard).sum()),
        "reference_targets_are_training_only": True, "target_interpretation": "Numeric presence proxy, not verified reasoning or semantic equivalence"})
    text = extract_features(bank)
    features = group.make_features(emb, text, weights, "group")
    assert features.shape == (4755, 4, 1252)
    folds = np.empty((2, len(bank)), np.int8)
    for repeat, seed in enumerate(FOLD_SEEDS):
        folds[repeat, np.random.default_rng(seed).permutation(len(bank))] = np.arange(len(bank)) % 3
    raw = np.full((2, len(bank), 4, len(CONFIGS), len(SEEDS)), np.nan, np.float32)
    normalized = np.full_like(raw, np.nan)
    scales = np.full((2, 3, len(CONFIGS), len(SEEDS)), np.nan, np.float64)
    records = []
    with threadpool_limits(limits=4):
        for repeat in range(2):
            for fold in range(3):
                fitting = np.flatnonzero(folds[repeat] != fold)
                held = np.flatnonzero(folds[repeat] == fold)
                mean, std = group.fit_statistics(features[fitting], group_valid[fitting])
                x = group.normalized(features, mean, std)
                directory = OUT / f"repeat{repeat}_fold{fold}"
                directory.mkdir()
                np.savez(directory / "partition.npz", fit_indices=fitting, held_indices=held, mean=mean, scale=std)
                for c, config in enumerate(CONFIGS):
                    config_dir = directory / config["name"]
                    config_dir.mkdir()
                    for s, seed in enumerate(SEEDS):
                        tick = time.perf_counter()
                        model, curve = fit(x[fitting], good[fitting], group_valid[fitting], target[fitting], observed[fitting], config, seed)
                        score = group.candidate_scores(group.scores(x, model[:4]), weights, "group")
                        scale = group.scale_scores(score[fitting], valid[fitting])
                        raw[repeat, held, :, c, s] = score[held]
                        normalized[repeat, held, :, c, s] = score[held] / scale
                        scales[repeat, fold, c, s] = scale
                        asset = {"w1": model[0], "b1": model[1], "w2": model[2], "skip": model[3],
                                 "aux_w": model[4], "aux_b": model[5], "mean": mean, "scale": std,
                                 "score_scale": np.array(scale), "seed": np.array(seed)}
                        replay = member_scores([bank[int(i)] for i in held[:16]], emb[held[:16]], asset, text[held[:16]])
                        replay_error = float(np.max(np.abs(replay - score[held[:16]])))
                        assert replay_error < 2e-5
                        path = config_dir / f"seed_{seed}.npz"
                        np.savez(path, **asset)
                        with zipfile.ZipFile(path) as archive:
                            expanded_bytes = sum(item.file_size for item in archive.infolist())
                        record = {"repeat": repeat, "fold": fold, "config": config["name"], "seed": seed,
                            "fit_questions": len(fitting), "fit_hard_questions": int(hard[fitting].sum()), "heldout_questions": len(held),
                            "fit_indices_sha256": array_hash(fitting), "held_indices_sha256": array_hash(held),
                            "auxiliary_fit_targets_sha256": array_hash(target[fitting]), "auxiliary_fit_observed_sha256": array_hash(observed[fitting]),
                            "score_scale": scale, "seconds": time.perf_counter() - tick, "curve": curve,
                            "asset_sha256": digest(path), "asset_uncompressed_bytes": expanded_bytes, "inference_replay_max_error": replay_error}
                        records.append(record)
                        dump(OUT / "fit_records.json", records)
                        progress("fitting", len(records), f"第{repeat + 1}次划分/折{fold + 1}：{config['name']}，种子{seed}完成")
                        print(json.dumps({"completed_fits": len(records), "repeat": repeat, "fold": fold,
                            "config": config["name"], "seed": seed, "seconds": record["seconds"]}), flush=True)
    assert len(records) == 90 and np.isfinite(raw).all() and np.isfinite(normalized).all()
    averaged = normalized.mean(axis=-1, dtype=np.float64)
    choices = np.empty((2, len(bank), len(CONFIGS)), np.int8)
    group_losses = np.empty((2, len(bank), len(CONFIGS)), np.float64)
    results = []
    for repeat in range(2):
        rows = {}
        for c, config in enumerate(CONFIGS):
            choice, loss, result = evaluate(bank, averaged[repeat, :, :, c], labels, hard, weights, group_valid, good)
            choices[repeat, :, c], group_losses[repeat, :, c] = choice, loss
            correct = labels[np.arange(len(bank)), choice]
            result["folds"] = [{"fold": fold, "hard_n": int((mask := hard & (folds[repeat] == fold)).sum()),
                "hard_correct": int(correct[mask].sum()), "group_ce": float(loss[mask].mean())} for fold in range(3)]
            rows[config["name"]] = result
        baseline = labels[np.arange(len(bank)), choices[repeat, :, 0]]
        for c, config in enumerate(CONFIGS):
            correct = labels[np.arange(len(bank)), choices[repeat, :, c]]
            rows[config["name"]]["vs_control"] = {"repairs": int((correct & ~baseline & hard).sum()),
                "regressions": int((~correct & baseline & hard).sum()), "gain": int(correct[hard].sum() - baseline[hard].sum()),
                "group_ce_difference": rows[config["name"]]["group_ce"] - rows[CONFIGS[0]["name"]]["group_ce"]}
        results.append({"repeat": repeat, "fold_seed": FOLD_SEEDS[repeat], "results": rows})
    np.savez(OUT / "oof_logits.npz", ids=np.array([row["id"] for row in bank]), raw_members=raw,
        normalized_members=normalized, raw=raw.mean(axis=-1, dtype=np.float64), normalized=averaged,
        config_names=np.array([c["name"] for c in CONFIGS]), member_seeds=np.array(SEEDS), choices=choices, group_losses=group_losses,
        labels=labels, valid=valid, hard=hard, folds=folds, fold_seeds=np.array(FOLD_SEEDS), score_scales=scales)
    dump(OUT / "oof_schema.json", {"normalized": "[2,4755,4,5] three fixed seed-normalized member mean",
        "raw": "[2,4755,4,5] descriptive mean of unnormalized member scores", "raw_members_normalized_members": "[2,4755,4,5,3]",
        "config_names": [c["name"] for c in CONFIGS], "choices_group_losses": "[2,4755,5]", "score_scales": "[2,3,5,3]",
        "folds": "[2,4755]", "labels_valid": "[4755,4] bool", "hard_ids": "[4755]"})
    assert all(digest(ROOT / path) == expected for path, expected in source_hashes.items())
    summary = {"status": "complete", "results": results, "model_fit_calls": len(records), "transform_fit_calls": 0,
        "normalizer_fits": 6, "seconds": time.perf_counter() - STARTED, "data_files_read": sorted(accesses),
        "validation_accessed": False, "test_accessed": False, "source_hashes_unchanged": True,
        "max_member_asset_uncompressed_bytes": max(r["asset_uncompressed_bytes"] for r in records),
        "max_three_member_asset_bytes": 3 * max(r["asset_uncompressed_bytes"] for r in records),
        "asset_scope": "Training audit assets include unused auxiliary head; final deployment may omit it. Config bytes not included; not a final resource certificate.",
        "caveat": "Weak lexical process proxies; alternate correct derivations remain correct. Allq control has more steps than historical hard-only A. Repeated questions are not independent samples. No configuration promotion or validation."}
    dump(OUT / "summary.json", summary)
    progress("complete", 90, "过程监督90次拟合完成；全部配置结果已保存，未访问验证或测试")
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
