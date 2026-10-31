"""True inner cross-fitting for B3 scales and separate B4 support fusion.

Registration does not fit. --run creates 144 inner learners and reuses the
previously fixed outer A2 fits. B3 and B4 are deliberately not combined.
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
import itertools
import json
from pathlib import Path
import time
import zipfile
import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from advanced_neural import development
from common import DATA, parse_answer
from ensemble import head_scores
from experiment_utils import dump, metrics
from finalize_validation import score_scale
from selection import select
from train_selected import fit_head
import optimization_baselines as baseline
import selective_routing as routing

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs/optimization_calibration"
BASE = ROOT / "runs/optimization_baselines"
FOLD_SEEDS = (20261027, 20261028)
MODELS = ("D", "T", "F", "G")
PATTERNS = ("3:1", "2:2", "2:1:1", "all_single")
METHODS = ("D", "T", "F", "G", "D_crossfit_scale", "T_crossfit_scale", "F_crossfit_scale", "G_crossfit_scale", "global_fusion", "pattern_shrink_fusion")
STARTED = None


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(x):
    x = np.ascontiguousarray(x)
    return hashlib.sha256(str(x.dtype).encode() + str(x.shape).encode() + x.tobytes()).hexdigest()


def progress(stage, count, detail):
    p = OUT / "progress.json"
    temp = p.with_suffix(".json.tmp")
    dump(temp, {"stage": stage, "completed_fits": count, "planned_fits": 144, "detail": detail,
        "elapsed_seconds": time.perf_counter() - STARTED if STARTED is not None else 0.,
        "updated_utc": datetime.now(timezone.utc).isoformat()})
    os.replace(temp, p)


def grid():
    values = np.array([v for v in itertools.product(range(5), repeat=4) if sum(v) == 4], np.float64) / 4
    assert values.shape == (35, 4)
    return values


def support_patterns(bank):
    result = []
    for row in bank:
        counts = {}
        for candidate in row["candidates"]:
            answer = parse_answer(candidate["text"])
            if answer is not None:
                counts[answer] = counts.get(answer, 0) + 1
        pattern = tuple(sorted(counts.values(), reverse=True))
        if pattern == (3, 1): label = "3:1"
        elif pattern == (2, 2): label = "2:2"
        elif pattern == (2, 1, 1): label = "2:1:1"
        elif len(pattern) >= 2 and all(n == 1 for n in pattern): label = "all_single"
        else: label = "other"
        result.append(label)
    return np.asarray(result)


def losses(data, indices, scores):
    grouped = np.einsum("ngc,nc->ng", data["weights"][indices].astype(np.float64), scores)
    return logsumexp(np.where(data["group_valid"][indices], grouped, -np.inf), axis=1) - (grouped * data["good"][indices]).sum(axis=1)


def choose_fusion(data, indices, sources):
    """Only inner-OOF rows belonging to one outer-fitting partition enter here."""
    bank = [data["bank"][int(i)] for i in indices]
    y = data["labels"][indices] > 0
    hard = data["hard"][indices]
    patterns = support_patterns(bank)
    weights = grid()
    correct = []
    ce = []
    for w in weights:
        score = sources @ w
        choices = select(bank, score, "mean")
        correct.append(y[np.arange(len(y)), choices])
        ce.append(losses(data, indices, score))
    correct, ce = np.asarray(correct), np.asarray(ce)
    def best(mask):
        assert mask.any()
        rows = [{"index": i, "weights": w.tolist(), "correct": int(correct[i, mask].sum()),
                 "n": int(mask.sum()), "group_ce": float(ce[i, mask].mean()), "active_heads": int(np.count_nonzero(w))}
                for i, w in enumerate(weights)]
        chosen = min(rows, key=lambda r: (-r["correct"], r["group_ce"], r["active_heads"], r["index"]))
        return chosen, rows
    global_choice, global_rows = best(hard)
    global_weights = np.asarray(global_choice["weights"])
    by_pattern = {}
    for pattern in PATTERNS:
        mask = hard & (patterns == pattern)
        if mask.sum() >= 80:
            local, rows = best(mask)
            effective = .5 * global_weights + .5 * np.asarray(local["weights"])
            by_pattern[pattern] = {"hard_questions": int(mask.sum()), "fallback": False, "local_selected": local,
                "effective_weights": effective.tolist(), "grid_rows": rows}
        else:
            by_pattern[pattern] = {"hard_questions": int(mask.sum()), "fallback": True, "effective_weights": global_weights.tolist()}
    return {"global_selected": global_choice, "global_grid": global_rows, "patterns": by_pattern,
            "other_fallback": global_weights.tolist(), "selection_indices_sha256": array_hash(indices),
            "selection_inputs": "Inner OOF predictions and labels within outer fitting partition only"}


def apply_fusion(bank, sources, policy):
    global_weights = np.asarray(policy["global_selected"]["weights"])
    global_score = sources @ global_weights
    row_weights = np.stack([policy["patterns"].get(p, {"effective_weights": global_weights})["effective_weights"] for p in support_patterns(bank)])
    return global_score, np.einsum("ncm,nm->nc", sources, row_weights)


def source_closure():
    paths = {Path(__file__).resolve()}
    for module in list(sys.modules.values()):
        p = getattr(module, "__file__", None)
        if p and Path(p).resolve().parent == ROOT and Path(p).suffix == ".py":
            paths.add(Path(p).resolve())
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths)}


def register():
    OUT.mkdir(parents=True, exist_ok=True)
    source_hashes = source_closure()
    existing = OUT / "protocol.json"
    if existing.exists():
        p = json.loads(existing.read_text(encoding="utf-8"))
        assert source_hashes == p["source_sha256"], "Registered source changed; preserve protocol and resolve before fitting"
        return p
    base_protocol = json.loads((BASE / "protocol.json").read_text(encoding="utf-8"))
    assert base_protocol["fold_seeds"] == list(FOLD_SEEDS)
    protocol = {"purpose": "B3 cross-fitted score-scale estimation and separate B4 genuinely nested support-pattern fusion",
        "outer_fold_seeds": FOLD_SEEDS, "outer_folds": 3,
        "inner_fold_seed": "20261029 + 100*repeat + outer_fold",
        "inner_fold_mapping": "inner[default_rng(seed).permutation(len(outer_fit))]=arange(len(outer_fit))%3; relative to sorted outerfit indices",
        "transform_seed": "20261101 + 1000*repeat + 100*outer_fold + inner_fold",
        "recipes": {"heads_D": base_protocol["heads_D"], "five_base_recipes": base_protocol["five_base_recipes"]},
        "head_order": baseline.HEADS, "source_models": MODELS, "method_order": METHODS,
        "planned_model_fits": 144, "planned_outer_new_fits": 0, "reused_outer_model_fits": 48,
        "planned_PCA_fits": 36, "planned_PLS_fits": 36,
        "B3": "For each head independently: std of all4-candidate-centered innerOOF raw scores over valid candidate positions in outerfit, float64, floor1e-6. Apply these scales to saved outer-model raw scores, keeping fixedD/T/F/G weights. No fittedlabels in scale calculation; fitting references only train inner models.",
        "B4": {"source_scores": "D/T/F/G mixtures of innerOOF heads normalized by each corresponding inner-fitting in-sample scale. Outer evaluation uses original A2 outer-fitting normalized scores.",
            "weights": grid().tolist(), "global_order": ["more hard correct", "lower hard group CE", "fewer active sources", "earlier grid row"],
            "patterns": PATTERNS, "all_single_definition": "At least2 valid parsed answer groups, each with one candidate; may have2/3/4 valid candidates",
            "other_patterns": "fallbackglobal; e.g2:1 withoneinvalid, unanimous, orno valid answer",
            "minimum_pattern_hard_questions": 80, "shrinkage": "effective=.5global+.5patternselected; below80usesglobal",
            "policy_freeze": "Save all inner choices and B3 scales before producing outer-method scores"},
        "separation": "B3 scales are NEVER used in B4 source weight selection or B4 outer prediction; a combined scheme would require a separately nested construction",
        "selection": "No final recipe promotion; compare all fixed B3scales vsownbaseline and B4global vsfixedF/patternvsglobal",
        "boundaries": "All8fresh inner learners/transforms see only innerfit. Outerheldlabels never choose scales/weights. Same questions repeated, no doubled independent samplesize.",
        "threads": 4, "validation_access": False, "test_access": False, "source_sha256": source_hashes,
        "outer_evidence_sha256": {name: digest(BASE / name) for name in ("protocol.json", "oof_logits.npz", "fit_records.json", "verification.json")}}
    dump(existing, protocol)
    progress("queued", 0, "B3/B4真正内层交叉拟合144次已登记；等待队列启动")
    return protocol


def audit_io():
    accesses = set()
    def audit(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path.name.startswith(("gsm8k_validation", "gsm8k_test")):
            raise RuntimeError("Forbidden split access")
        if DATA.resolve() in path.parents:
            accesses.add(path.relative_to(DATA.resolve()).as_posix())
        mode, flags = args[1], args[2] or 0
        write = isinstance(mode, str) and any(c in mode for c in "wax+") or bool(flags & (os.O_WRONLY | os.O_RDWR))
        if write and path != OUT.resolve() and OUT.resolve() not in path.parents:
            raise RuntimeError(f"Write outside experiment: {path}")
    sys.addaudithook(audit)
    return accesses


def fit_eight(data, fitting, held, seed, folder, recipes, records, transforms):
    folder.mkdir()
    fit_bank = [data["bank"][int(i)] for i in fitting]
    held_bank = [data["bank"][int(i)] for i in held]
    labels = data["labels"][fitting] > 0
    valid = data["valid"][fitting]
    pairs = labels[:, :, None] * ((~labels) & valid)[:, None, :]
    counts = pairs.sum(axis=(1, 2))
    pair_weights = (pairs / np.maximum(counts[:, None, None], 1)).astype(np.float32)
    raw = np.empty((len(held), 4, 8), np.float64)
    normalized = np.empty_like(raw)
    for col, head in enumerate(recipes):
        tick = time.perf_counter()
        asset = fit_head(head["kind"], head["training"], fit_bank, data["embeddings"][fitting], data["original"][fitting], labels, valid, pair_weights, counts > 0)
        fit_score = head_scores(head["kind"], fit_bank, data["embeddings"][fitting], data["original"][fitting], asset)
        scale = score_scale(fit_score, fit_bank)
        z = head_scores(head["kind"], held_bank, data["embeddings"][held], data["original"][held], asset)
        raw[:, :, col] = z
        normalized[:, :, col] = (z / scale).astype(np.float32)
        path = folder / f"{baseline.HEADS[col]}.npz"
        np.savez(path, **asset)
        records.append({"partition": folder.name, "head": baseline.HEADS[col], "recipe": head["training"],
            "fit_questions": len(fitting), "heldout_questions": len(held), "fit_indices_sha256": routing.array_hash(fitting.astype(np.int64)),
            "held_indices_sha256": routing.array_hash(held.astype(np.int64)), "score_scale": scale,
            "asset_sha256": digest(path), "asset_uncompressed_bytes": baseline.asset_bytes(path), "seconds": time.perf_counter()-tick})
        dump(OUT / "fit_records.json", records)
        progress("fitting", len(records), f"{folder.name}：{baseline.HEADS[col]}拟合完成")
        print(json.dumps({"completed_fits": len(records), "partition": folder.name, "head": baseline.HEADS[col]}), flush=True)
    five_raw, five_norm = routing.fit_partition(data, fitting, held, seed, folder, records, transforms)
    raw[:, :, 3:], normalized[:, :, 3:] = five_raw, five_norm
    for name in ("Q", "C"):
        with np.load(folder / f"{name}.npz", allow_pickle=False) as asset:
            assert np.any(asset["pls_rotations"] != 0)
    np.savez(folder / "all_scores.npz", held_indices=held, raw=raw, normalized=normalized, head_names=np.array(baseline.HEADS))
    dump(folder / "all_heads.json", {"model_fit_calls": 8, "learners": records[-8:]})
    return raw, normalized


def run():
    global STARTED
    protocol = register()
    if (OUT / "run_started.json").exists():
        raise FileExistsError("Execution already recorded; preserve existing work")
    STARTED = time.perf_counter()
    accessed = audit_io()
    routing.OUT = OUT
    def routing_dump(path, value):
        dump(path, value)
        if Path(path) == OUT / "fit_records.json":
            progress("fitting", len(value), f"{value[-1]['partition']}：{value[-1]['head']}拟合完成")
    routing.dump = routing_dump
    dump(OUT / "run_started.json", {"utc": datetime.now(timezone.utc).isoformat()})
    for name, expected in protocol["outer_evidence_sha256"].items():
        assert digest(BASE / name) == expected
    progress("preparing", 0, "准备训练输入与固定外折证据；B3/B4保持独立")
    bank, embeddings, labels, valid, hard = development("train")
    data, preparation = routing.prepare(bank, embeddings, labels.astype(np.float32), hard)
    centered = routing.condition.stable_center(data["labels"][hard], valid[hard])[valid[hard]]
    assert data["labels"].dtype == np.float32 and centered.min() < 0 < centered.max()
    dump(OUT / "preparation.json", preparation)
    with np.load(BASE / "oof_logits.npz", allow_pickle=False) as archive:
        outer = {key: archive[key] for key in ("raw", "normalized", "folds", "ids", "labels", "valid", "hard", "mixture_scores")}
    for key, expected in (("labels", labels), ("valid", valid), ("hard", hard), ("ids", np.array([r["id"] for r in bank]))):
        assert np.array_equal(outer[key], expected)
    output = np.full((2, len(bank), 4, len(METHODS)), np.nan, np.float64)
    output[:, :, :, :4] = outer["mixture_scores"]
    crossfit_scales = np.empty((2, 3, 8), np.float64)
    records, transforms, policies = [], [], []
    with threadpool_limits(limits=4):
        for repeat in range(2):
            for fold in range(3):
                outer_fit = np.flatnonzero(outer["folds"][repeat] != fold)
                outer_held = np.flatnonzero(outer["folds"][repeat] == fold)
                folder = OUT / f"repeat{repeat}_outer{fold}"
                folder.mkdir()
                inner_seed = 20261029 + 100*repeat + fold
                inner = routing.folds_for(len(outer_fit), inner_seed)
                inner_raw = np.full((len(outer_fit), 4, 8), np.nan, np.float64)
                inner_norm = np.full_like(inner_raw, np.nan)
                np.savez(folder / "outer_partition.npz", fitting=outer_fit, held=outer_held, inner_fold=inner)
                for inner_fold in range(3):
                    fitting = outer_fit[inner != inner_fold]
                    held = outer_fit[inner == inner_fold]
                    assert not np.intersect1d(fitting, outer_held).size and not np.intersect1d(held, outer_held).size
                    seed = 20261101 + 1000*repeat + 100*fold + inner_fold
                    raw, normalized = fit_eight(data, fitting, held, seed, folder / f"repeat{repeat}_outer{fold}_inner{inner_fold}", protocol["recipes"]["heads_D"], records, transforms)
                    inner_raw[inner == inner_fold] = raw
                    inner_norm[inner == inner_fold] = normalized
                assert np.isfinite(inner_raw).all() and np.isfinite(inner_norm).all()
                centered = inner_raw - inner_raw.mean(axis=1, keepdims=True)
                scales = np.std(centered[valid[outer_fit]], axis=0, dtype=np.float64).clip(1e-6)
                assert scales.shape == (8,) and np.isfinite(scales).all()
                crossfit_scales[repeat, fold] = scales
                inner_sources = baseline.mixture_scores(inner_norm)
                policy = choose_fusion(data, outer_fit, inner_sources)
                policy.update(repeat=repeat, outer_fold=fold, inner_seed=inner_seed,
                    B3_scales=scales.tolist(), B3_scale_indices_sha256=array_hash(outer_fit),
                    B4_normalization="Original inner-fitting score scales; not B3 scales",
                    outer_held_indices_sha256=array_hash(outer_held))
                dump(folder / "frozen_policy.json", policy)
                policies.append(policy)
                np.savez(folder / "inner_oof.npz", ids=np.array([bank[int(i)]["id"] for i in outer_fit]), indices=outer_fit,
                    raw=inner_raw, normalized=inner_norm, source_scores=inner_sources, inner_fold=inner,
                    labels=labels[outer_fit], valid=valid[outer_fit], hard=hard[outer_fit], B3_scales=scales)
                # Only after freezing inner choices, apply them to outer held rows.
                recalibrated = outer["raw"][repeat, outer_held] / scales
                recalibrated[:, :, :3] = recalibrated[:, :, :3].astype(np.float32)
                output[repeat, outer_held, :, 4:8] = baseline.mixture_scores(recalibrated)
                global_score, pattern_score = apply_fusion([bank[int(i)] for i in outer_held], outer["mixture_scores"][repeat, outer_held], policy)
                output[repeat, outer_held, :, 8] = global_score
                output[repeat, outer_held, :, 9] = pattern_score
    assert len(records) == 144 and len(transforms) == 72 and np.isfinite(output).all()
    choices = np.empty((2, len(bank), len(METHODS)), np.int8)
    group_loss = np.empty((2, len(bank), len(METHODS)), np.float64)
    results = []
    comparisons = ((4, 0), (5, 1), (6, 2), (7, 3), (8, 2), (9, 8), (9, 2))
    for repeat in range(2):
        rows = {}
        for column, name in enumerate(METHODS):
            score = output[repeat, :, :, column]
            choice = np.asarray(select(bank, score, "mean"), np.int8)
            ce = losses(data, np.arange(len(bank)), score)
            choices[repeat, :, column], group_loss[repeat, :, column] = choice, ce
            correct = labels[np.arange(len(bank)), choice]
            rows[name] = {**metrics(choice, labels, hard), "group_ce": float(ce[hard].mean()),
                "folds": [{"fold": fold, "hard_n": int((mask := hard & (outer["folds"][repeat] == fold)).sum()),
                    "hard_correct": int(correct[mask].sum()), "group_ce": float(ce[mask].mean())} for fold in range(3)]}
        paired = []
        for c, r in comparisons:
            paired.append({"candidate": METHODS[c], "reference": METHODS[r],
                "paired": routing.paired(choices[repeat, :, c], choices[repeat, :, r], labels, hard, outer["folds"][repeat]),
                "group_ce_difference": rows[METHODS[c]]["group_ce"] - rows[METHODS[r]]["group_ce"]})
        results.append({"repeat": repeat, "fold_seed": FOLD_SEEDS[repeat], "results": rows, "paired_comparisons": paired})
    np.savez(OUT / "oof_logits.npz", ids=outer["ids"], scores=output, normalized=output, method_names=np.array(METHODS),
        labels=labels, valid=valid, hard=hard, folds=outer["folds"], choices=choices, group_losses=group_loss, crossfit_scales=crossfit_scales)
    dump(OUT / "policies.json", policies)
    dump(OUT / "oof_schema.json", {"scores_normalized": "[2,N,4,10] method_order inprotocol; scores are finalmethod scales, not additionally calibrated",
        "crossfit_scales": "[2,3,8] B3 only", "choices_group_losses": "[2,N,10]", "folds": "[2,N]",
        "inner_evidence": "Eachrepeat/outer folder containsinnerOOF scores,3partitionswith8models,and frozenpolicy",
        "B4": "Global/pattern weights use originalbase normalization; never B3scales"})
    assert all(digest(ROOT / path) == expected for path, expected in protocol["source_sha256"].items())
    assert all(digest(BASE / path) == expected for path, expected in protocol["outer_evidence_sha256"].items())
    summary = {"status": "complete", "results": results, "model_fit_calls": 144, "new_outer_model_fits": 0,
        "reused_outer_model_fits": 48, "PCA_fit_calls": sum(r["kind"] == "PCA" for r in transforms),
        "PLS_fit_calls": sum(r["kind"] == "PLS" for r in transforms), "seconds": time.perf_counter()-STARTED,
        "validation_accessed": False, "test_accessed": False, "data_files_read": sorted(accessed),
        "source_hashes_unchanged": True, "outer_evidence_hashes_unchanged": True,
        "caveat": "Conditional onhistorically selectedfixedbase recipes. TrueinnerOOF isolatesnewscale/weightfitting fromouterlabels,notentirehistoricalsearch. NoB3+B4combinedscheme, nofinalmodelpromotion."}
    dump(OUT / "summary.json", summary)
    progress("complete", 144, "嵌套校准与支持模式融合完成；未访问验证或测试")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--register-only", action="store_true")
    action.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.register_only:
        p = register()
        print(json.dumps({"status": "registered_not_started", "planned_model_fits": p["planned_model_fits"], "command": "python optimization_calibration.py --run"}))
    else:
        try:
            run()
        except Exception as error:
            if STARTED is not None:
                p = OUT / "fit_records.json"
                count = len(json.loads(p.read_text())) if p.exists() else 0
                progress("failed", count, f"{type(error).__name__}: {error}")
            raise
