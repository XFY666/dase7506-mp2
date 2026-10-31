"""C2/E2 token-process auxiliary control; prepare first, run only explicitly."""
from __future__ import annotations
import os
import sys
sys.dont_write_bytecode = True
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "4"
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
_TEMP = Path(__file__).resolve().parent/"runs/optimization_sequence_aux/tmp"
_TEMP.mkdir(parents=True, exist_ok=True)
os.environ["TMP"] = os.environ["TEMP"] = str(_TEMP)
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from common import DATA, parse_answer, read_jsonl
from experiment_utils import development_data, dump, metrics
from selection import select
import sequence_experiment as base
import sequence_model as original
import optimization_sequence as screen
import optimization_sequence_model as variants

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs/optimization_sequence_aux"
FOLD_SEEDS = (20261027, 20261028)
CONFIGS = (("control_allq", 0.), ("token_aux01", .1))
PLANNED_FITS = 24
SOURCES = ("optimization_sequence_aux.py", "optimization_sequence.py", "optimization_sequence_model.py",
           "sequence_experiment.py", "sequence_model.py", "common.py", "experiment_utils.py",
           "selection.py", "text_features.py")
NUMBER = r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:\s*/\s*[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+))?"
EXACT_NUM = re.compile(NUMBER)
ANNOTATION = re.compile(r"<<([^<>]*)>>")
FRACTION = re.compile(r"\\(?:dfrac|tfrac|frac)\s*\{\s*(" + NUMBER + r")\s*\}\s*\{\s*(" + NUMBER + r")\s*\}")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    return {name: sha(ROOT/name) for name in SOURCES}


def literal_value(text):
    token = FRACTION.sub(lambda m: m.group(1)+"/"+m.group(2), str(text)).strip().replace(" ", "")
    return parse_answer("#### "+token) if EXACT_NUM.fullmatch(token) else None


def intermediate_values(reference):
    # Same annotation-position rule and literal grammar as optimization_process.py.
    annotations = ANNOTATION.findall(reference)
    values = set()
    for expression in annotations[:-1]:
        if "=" in expression:
            value = literal_value(expression.rsplit("=", 1)[1])
            if value is not None:
                values.add(value)
    return values, len(annotations)


def token_targets(bank, references):
    target = np.zeros((len(bank), 4, 322), np.float32)
    mask = np.zeros(target.shape, bool)
    observed = np.zeros(len(bank), bool)
    gold_counts = np.zeros(len(bank), np.int16)
    for i, (row, ref) in enumerate(zip(bank, references)):
        assert row["id"] == ref["id"] and row["question"] == ref["question"]
        qparts = variants.role_parts(row["question"])
        qvalues = {value for part in qparts if part and part[0].isdigit()
                   and (value := variants.number(part)) is not None}
        gold, _ = intermediate_values(ref["answer"])
        parsed_final = parse_answer(ref["answer"])
        final = variants.number(parsed_final) if parsed_final is not None else None
        # The existing role tokenizer gives unsigned literals and retains signs as separate tokens.
        # Limit this weak target to positive/zero intermediate literals; never infer unary sign meaning.
        usable = {value for text in gold if (value := variants.number(text)) is not None
                  and value >= 0 and value not in qvalues and value != final}
        gold_counts[i] = len(usable)
        if not usable:
            continue
        q = qparts if len(qparts) <= 64 else qparts[:32]+qparts[-32:]
        offset = len(q)+1
        for j, candidate in enumerate(row["candidates"]):
            if parse_answer(candidate["text"]) is None:
                continue
            parts = variants.role_parts(candidate["text"])
            values = [variants.number(part) if part and part[0].isdigit() else None for part in parts]
            # Conservatively omit a number immediately following a +/- token: the lexer does not
            # disambiguate signed literals versus subtraction/addition operands.
            valid_tokens = [value is not None and value not in qvalues and value != final
                            and not (k and parts[k-1] in ("-", "+", "−"))
                            for k, value in enumerate(values)]
            if len(parts) > 256:
                values = values[:128]+[None]+values[-128:]
                valid_tokens = valid_tokens[:128]+[False]+valid_tokens[-128:]
            for k, (value, active) in enumerate(zip(values, valid_tokens)):
                if active:
                    mask[i, j, offset+k] = True
                    target[i, j, offset+k] = float(value in usable)
        observed[i] = bool(mask[i].any())
    return target, mask, observed, gold_counts


class AuxCNN(base.WordCNN):
    def __init__(self, vocabulary_size):
        super().__init__(vocabulary_size, True)
        self.auxiliary = nn.Linear(64, 1)

    def forward_aux(self, codes, numeric):
        shape = codes.shape
        flat = codes.reshape(-1, shape[-1])
        mask = flat.ne(0)
        x = self.embedding(flat).transpose(1, 2)
        pooled, token_features = [], []
        for conv in (self.conv3, self.conv5):
            z = torch.relu(conv(x)).transpose(1, 2)
            token_features.append(z)
            maximum = z.masked_fill(~mask[:, :, None], -1e9).amax(dim=1)
            mean = (z*mask[:, :, None]).sum(dim=1)/mask.sum(dim=1, keepdim=True).clamp_min(1)
            pooled.extend((maximum, mean))
        pooled.append(numeric.reshape(-1, 1099))
        score = self.output(torch.tanh(self.hidden(torch.cat(pooled, dim=1)))).reshape(shape[0], 4)
        aux = self.auxiliary(torch.cat(token_features, dim=2)).reshape(shape)
        return score, aux


def objective(score, aux, weights, gv, good, eligible, target, mask, strength):
    main = base.objective(score[eligible], weights[eligible], gv[eligible], good[eligible]) if eligible.any() else score.sum()*0.
    per_token = F.binary_cross_entropy_with_logits(aux, target, reduction="none")
    counts = mask.sum(dim=(1, 2))
    observed = counts > 0
    per_question = (per_token*mask).sum(dim=(1, 2))/counts.clamp_min(1)
    auxiliary = per_question[observed].mean() if observed.any() else aux.sum()*0.
    return main + strength*auxiliary, main, auxiliary


def train(codes, numeric, structure, eligible, target, mask, fit_ids, strength, epochs, held_ids=None):
    weights, gv, good = structure
    torch.manual_seed(7506)
    rng = np.random.default_rng(7507)
    model = AuxCNN(int(codes.max())+1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    best, chosen, stale = float("inf"), 0, 0
    curves = []
    tick = time.perf_counter()
    for epoch in range(1, epochs+1):
        model.train()
        order = rng.permutation(fit_ids)
        sums = np.zeros(3)
        for start in range(0, len(order), 64):
            ids = order[start:start+64]
            optimizer.zero_grad(set_to_none=True)
            z, aux = model.forward_aux(codes[ids], numeric[ids])
            loss, main, weak = objective(z, aux, weights[ids], gv[ids], good[ids], eligible[ids],
                                         target[ids], mask[ids], strength)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            sums += np.array([float(loss.detach()), float(main.detach()), float(weak.detach())])*len(ids)
        row = {"epoch": epoch, "objective": sums[0]/len(fit_ids), "main": sums[1]/len(fit_ids),
               "weak_auxiliary": sums[2]/len(fit_ids), "elapsed_seconds": time.perf_counter()-tick}
        if held_ids is not None:
            model.eval()
            ids_all = held_ids[eligible[held_ids].numpy()]
            total = 0.
            with torch.no_grad():
                for start in range(0, len(ids_all), 64):
                    ids = ids_all[start:start+64]
                    z = model(codes[ids], numeric[ids])
                    total += float(base.objective(z, weights[ids], gv[ids], good[ids]))*len(ids)
            current = total/len(ids_all)
            row["heldout_main_group_ce"] = current
            if current < best-1e-4:
                best, chosen, stale = current, epoch, 0
            else:
                stale += 1
        curves.append(row)
        if held_ids is not None and stale >= 4:
            break
    return model, curves, chosen if held_ids is not None else epochs


def checks():
    bank = [{"id": "tiny", "question": "There are 3 boxes, each with 4 apples. How many remain after using 2?",
             "candidates": [{"text": text, "index": j} for j, text in enumerate([
                 "3 * 4 = 12. 12 - 2 = 10. #### 10", "3 + 4 = 7. #### 7",
                 "There are 12 before use, 2 used, and 10 remain. #### 10", "12 apples. #### 12"])]}]
    refs = [{"id": "tiny", "question": bank[0]["question"], "answer": "3*4=<<3*4=12>>12; 12-2=<<12-2=10>>10. #### 10"}]
    targets, masks, observed, counts = token_targets(bank, refs)
    assert observed.tolist() == [True] and counts.tolist() == [1]
    seqs = variants.sequences(bank, "headtail_roles")
    for j, seq in enumerate(seqs[0]):
        assert not masks[0, j, :seq.index("<sep>")+1].any()
        positives = [tok for k, tok in enumerate(seq) if masks[0, j, k] and targets[0, j, k]]
        if j in (0, 2, 3):
            assert positives
    # Missing intermediate evidence gives no negatives and no auxiliary gradient.
    absent = [dict(refs[0], answer="Only <<12-2=10>>. #### 10")]
    assert not token_targets(bank, absent)[1].any()
    torch.manual_seed(44)
    score = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    aux = torch.randn(2, 4, 5, dtype=torch.float64, requires_grad=True)
    weights = torch.eye(4, dtype=torch.float64).expand(2, 4, 4)
    gv = torch.ones(2, 4, dtype=torch.bool)
    good = torch.tensor([[1, 0, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
    eligible = torch.tensor([True, False])
    target = torch.zeros_like(aux)
    target[0, 1, 2] = 1
    mask = torch.zeros_like(aux, dtype=torch.bool)
    mask[:, :2, 1:4] = True
    assert torch.autograd.gradcheck(lambda a, b: objective(a, b, weights, gv, good, eligible, target, mask, .1)[0], (score, aux), eps=1e-6, atol=1e-5)
    zero = objective(score, aux, weights, gv, good, eligible, target, mask, 0.)[0]
    assert torch.allclose(zero, base.objective(score[:1], weights[:1], gv[:1], good[:1]))
    torch.manual_seed(7506)
    model = AuxCNN(20).eval()
    codes = torch.randint(0, 20, (2, 4, 322))
    numeric = torch.randn(2, 4, 1099)
    with torch.no_grad():
        assert torch.equal(model(codes, numeric), model.forward_aux(codes, numeric)[0])
    return {"target_alignment": True, "question_and_true_final_mask": True,
            "empty_gold_is_unobserved": True, "combined_objective_gradcheck": True,
            "lambda_zero_equals_main": True, "main_forward_unchanged": True}


def protocol():
    return {"task": "E2/C2 after repeated-OOF numeric-role CNN improvement; all training-only",
        "configs": [{"name": name, "auxiliary_weight": weight} for name, weight in CONFIGS],
        "fold_seeds": list(FOLD_SEEDS), "folds": 3,
        "fold_assignment": "fold[default_rng(seed).permutation(N)] = arange(N)%3",
        "inner_split": "85/15 outer-fitting question IDs; seed outer_seed+100+fold",
        "input_recipe": "unchanged headtail_roles from completed optimization_sequence study",
        "vocabulary_statistics": "independently fitted inside inner-fit and outer-fit; no gold-derived features, masks or labels enter vocabulary, encoding, numeric inputs or transforms",
        "gold_parser": "same as optimization_process.intermediate_values: <<...>> annotations, remove final annotation by position, canonicalize literal RHS after rightmost '='",
        "weak_target": "binary membership of a visible candidate numeric token in usable gold intermediate values; a literal proxy, not proof of step correctness",
        "usable_gold": "nonnegative parsed intermediate values excluding any question literal and true final reference value; empty set unobserved, never all-negative supervision",
        "auxiliary_mask": "candidate positions only, valid parsed-answer candidates only; exclude question literals, true final value, and numeric token immediately preceded by +, -, or unicode minus; no question/padding/separator/gap labels",
        "token_alignment": "exact headtail_roles token sequence and truncation; numeric roles derived only from inputs; original unsigned literal lexer retained",
        "auxiliary_head": "linear64→1 on concatenated ReLU Conv3/Conv5 contextual token features; dropped at inference/export",
        "loss": "main group-mean CE averaged across eligible questions in each batch + lambda * BCE averaged over masked tokens within each observed question then over observed questions",
        "all_question_batches": "both lambda0 andlambda.1 use all outer-fitting questions; isolates auxiliary signal from extra steps/easy-question exposure; lambda0 is a fresh matched control, not cached E1 scores",
        "optimizer": "AdamW lr.001 weight_decay.001, gradientclip5, batch64, fixedseed7506/shuffle7507",
        "stopping": "max24epochs/patience4/minimprovement1e-4; main groupCE only on inner-held eligible questions; fresh outer refit",
        "fitting_scope": "weak targets precomputed from supplied training references; only fitting IDs participate in auxiliary gradients; held-out weak targets never used for stopping or scoring",
        "screen_order": "highest mean hard correct across two repeats, then lower mean normalized groupCE, thenlambda0",
        "estimation_scope": "same2114 hard questions repeated; exploratory paired comparison, not4228 independent samples",
        "planned_fits": PLANNED_FITS, "max_elapsed_seconds": 3600,
        "no_validation_test_full_refit": True, "threads": 4}


def progress(stage, completed, detail, started):
    screen.atomic_dump(OUT/"progress.json", {"stage": stage, "completed_fits": completed,
        "planned_fits": PLANNED_FITS, "detail": detail, "elapsed_seconds": time.perf_counter()-started,
        "updated_utc": datetime.now(timezone.utc).isoformat()})


def prepare_registration():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/"protocol.json").exists():
        raise FileExistsError("Keep existing registration immutable")
    result = checks()
    dump(OUT/"input_gradient_checks.json", result)
    dump(OUT/"protocol.json", protocol())
    dump(OUT/"source_evidence.json", source_hashes())
    progress("ready", 0, "prepared and checked; waiting for explicit --run", time.perf_counter())
    print(json.dumps({"ready": True, "planned_fits": PLANNED_FITS, "checks": result}, indent=2))


def audit_io():
    accesses = set()
    def hook(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path.name.startswith(("gsm8k_validation", "gsm8k_test")):
            raise RuntimeError(f"Forbidden split {path}")
        if DATA.resolve() in path.parents:
            accesses.add(path.relative_to(DATA.resolve()).as_posix())
        mode, flags = args[1], args[2] or 0
        writing = isinstance(mode, str) and any(c in mode for c in "wax+") or bool(flags & (os.O_WRONLY | os.O_RDWR))
        if writing and OUT.resolve() not in path.parents:
            raise RuntimeError(f"Persistent output outside assigned directory {path}")
    sys.addaudithook(hook)
    return accesses


def run():
    started = time.perf_counter()
    expected = json.loads((OUT/"source_evidence.json").read_text(encoding="utf-8"))
    assert expected == source_hashes(), "Registered source changed"
    assert json.loads((OUT/"protocol.json").read_text(encoding="utf-8")) == protocol()
    if (OUT/"started.json").exists():
        raise FileExistsError("This stage has already started; preserve partial evidence")
    (OUT/"tmp").mkdir(exist_ok=True)
    os.environ["TMP"] = os.environ["TEMP"] = str(OUT/"tmp")
    accesses = audit_io()
    dump(OUT/"started.json", {"utc": datetime.now(timezone.utc).isoformat(), "sources": expected})
    progress("prepare_train_only", 0, "input tokenization and weak labels", started)
    bank, embeddings, labels, hard = development_data("train")
    refs = read_jsonl(DATA/"gsm8k_train.jsonl")
    weights, gv, good, valid, eligible = base.group_structure(bank, labels)
    assert np.array_equal(eligible, hard)
    structure = weights, gv, good
    target, mask, observed, gold_counts = token_targets(bank, refs)
    dump(OUT/"target_diagnostics.json", {"questions": len(bank), "observed_questions": int(observed.sum()),
        "observed_easy_questions": int((observed & ~hard).sum()), "tokens": int(mask.sum()),
        "positive_tokens": int((target*mask).sum()), "usable_gold_values_mean": float(gold_counts.mean()),
        "all_training_only": True, "weak_labels_not_features": True})
    raw_inputs = original.numeric_inputs(bank, embeddings)
    seqs = variants.sequences(bank, "headtail_roles")
    folds = np.empty((2, len(bank)), np.int8)
    for r, seed in enumerate(FOLD_SEEDS):
        folds[r, np.random.default_rng(seed).permutation(len(bank))] = np.arange(len(bank))%3
    ids = np.array([row["id"] for row in bank])
    raw = np.full((2, len(bank), 4, 2), np.nan, np.float32)
    normalized = raw.copy()
    records = []
    completed = 0
    target_t, mask_t, eligible_t = torch.from_numpy(target), torch.from_numpy(mask), torch.from_numpy(eligible)
    for r, seed in enumerate(FOLD_SEEDS):
        for f in range(3):
            fit_ids, held_ids = np.flatnonzero(folds[r] != f), np.flatnonzero(folds[r] == f)
            order = np.random.default_rng(seed+100+f).permutation(fit_ids)
            cut = int(.85*len(order))
            inner, early_held = order[:cut], order[cut:]
            np.savez_compressed(OUT/f"repeat{r}_fold{f}_indices.npz", fit=fit_ids, held=held_ids, inner=inner, early_held=early_held)
            for c, (name, strength) in enumerate(CONFIGS):
                if time.perf_counter()-started > 3600:
                    raise RuntimeError("Registered time bound reached before a new fit")
                stem = f"repeat{r}_fold{f}_{name}"
                progress("early_fit", completed, stem, started)
                codes, numeric, transform, info = screen.prepare(seqs, "headtail_roles", raw_inputs, valid, inner)
                ct, nt, st = base.tensor_inputs(codes, numeric, structure)
                model, curve, epochs = train(ct, nt, st, eligible_t, target_t, mask_t, inner, strength, 24, early_held)
                completed += 1
                dump(OUT/f"{stem}_early_curve.json", curve)
                early_seconds = curve[-1]["elapsed_seconds"]
                progress("outer_refit", completed, f"{stem}; epochs={epochs}", started)
                codes, numeric, transform, outer = screen.prepare(seqs, "headtail_roles", raw_inputs, valid, fit_ids)
                ct, nt, st = base.tensor_inputs(codes, numeric, structure)
                model, curve, _ = train(ct, nt, st, eligible_t, target_t, mask_t, fit_ids, strength, epochs)
                completed += 1
                dump(OUT/f"{stem}_refit_curve.json", curve)
                z = base.native_scores(model, ct, nt)
                asset = base.export(model, transform)
                scale = float(max(np.std((z[fit_ids]-z[fit_ids].mean(1, keepdims=True))[valid[fit_ids]]), 1e-6))
                asset.update(score_scale=np.array(scale), token_recipe=np.array("headtail_roles"))
                np.savez(OUT/f"{stem}.npz", **asset)
                replay = original.numpy_forward(codes[held_ids[:32]], numeric[held_ids[:32]], asset)
                error = float(np.max(abs(replay-z[held_ids[:32]])))
                assert error < 1e-4
                assert select([bank[i] for i in held_ids[:32]], replay, "mean") == select([bank[i] for i in held_ids[:32]], z[held_ids[:32]], "mean")
                raw[r, held_ids, :, c] = z[held_ids]
                normalized[r, held_ids, :, c] = z[held_ids]/scale
                record = {"repeat": r, "fold": f, "config": name, "auxiliary_weight": strength,
                    "selected_epochs": epochs, "early_fit_seconds": early_seconds,
                    "refit_seconds": curve[-1]["elapsed_seconds"], "inner_transform": info,
                    "outer_transform": outer, "score_scale": scale,
                    "fit_indices_sha256": screen.ahash(fit_ids), "held_indices_sha256": screen.ahash(held_ids),
                    "asset_bytes": (OUT/f"{stem}.npz").stat().st_size, "asset_sha256": sha(OUT/f"{stem}.npz"),
                    "export_max_abs_error": error,
                    "metrics": metrics(select([bank[i] for i in held_ids], z[held_ids], "mean"), labels[held_ids], hard[held_ids])}
                records.append(record)
                screen.atomic_dump(OUT/"fit_records.json", records)
                np.savez_compressed(OUT/"partial_oof.npz", raw=raw, normalized=normalized, ids=ids,
                    folds=folds, labels=labels, valid=valid, hard=hard, configs=np.array([n for n,_ in CONFIGS]))
                progress("outer_complete", completed, stem, started)
    assert np.isfinite(raw).all()
    rows = []
    for c, (name, strength) in enumerate(CONFIGS):
        repeats = []
        for r in range(2):
            choices = np.asarray(select(bank, raw[r,:,:,c], "mean"))
            controls = np.asarray(select(bank, raw[r,:,:,0], "mean"))
            right = labels[np.arange(len(bank)), choices].astype(bool)
            control = labels[np.arange(len(bank)), controls].astype(bool)
            repeats.append({"seed": FOLD_SEEDS[r], "metrics": metrics(choices, labels, hard),
                "group_ce": base.numpy_group_loss(normalized[r,:,:,c], weights, gv, good, eligible),
                "repairs_vs_control": int((hard & right & ~control).sum()),
                "regressions_vs_control": int((hard & control & ~right).sum())})
        rows.append({"config": name, "auxiliary_weight": strength, "repeats": repeats,
                     "mean_hard_correct": float(np.mean([v["metrics"]["hard_correct"] for v in repeats])),
                     "mean_group_ce": float(np.mean([v["group_ce"] for v in repeats]))})
    selected = min(rows, key=lambda x: (-x["mean_hard_correct"], x["mean_group_ce"], x["auxiliary_weight"]))
    np.savez_compressed(OUT/"oof_logits.npz", raw=raw, normalized=normalized, ids=ids,
        folds=folds, labels=labels, valid=valid, hard=hard, configs=np.array([n for n,_ in CONFIGS]))
    assert expected == source_hashes()
    summary = {"configs": rows, "selected": selected, "model_fit_calls": completed,
        "vocabulary_fits": completed, "numeric_normalizer_fits": completed,
        "max_asset_bytes": max(v["asset_bytes"] for v in records),
        "max_export_abs_error": max(v["export_max_abs_error"] for v in records),
        "seconds": time.perf_counter()-started, "data_files_read": sorted(accesses),
        "validation_evaluations": 0, "test_access": False, "full_model_fitted": False,
        "interpretation": "Weak token-membership supervision; all-question matched control differs from historical hard-only CNN. Same questions repeated, exploratory comparison only."}
    dump(OUT/"summary.json", summary)
    progress("complete", completed, "selected "+selected["config"], started)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.prepare_only == args.run:
        parser.error("Choose exactly one of --prepare-only or --run")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    assert torch.version.cuda is None
    if args.prepare_only:
        prepare_registration()
    else:
        run()
