"""Two bounded from-scratch attention variants; explicit prepare and run stages."""
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
import time
ROOT = Path(__file__).resolve().parent
OUT = ROOT/"runs/optimization_transformer"
(OUT/"tmp").mkdir(parents=True, exist_ok=True)
os.environ["TMP"] = os.environ["TEMP"] = str(OUT/"tmp")
import numpy as np
import torch
from torch import nn

from common import DATA
from experiment_utils import development_data, dump, metrics
from selection import select
import sequence_experiment as base
import sequence_model as original
import optimization_sequence as screen
import optimization_sequence_model as variants
import optimization_transformer_model as inference

FOLD_SEEDS = (20261027, 20261028)
CONFIGS = (("token_attention", False), ("token_set_attention", True))
SOURCES = ("optimization_transformer.py", "optimization_transformer_model.py", "optimization_sequence.py",
           "optimization_sequence_model.py", "sequence_model.py", "sequence_experiment.py",
           "common.py", "experiment_utils.py", "selection.py", "text_features.py")
PLANNED_FITS = 24
TIME_CAP = 7200.
INFERENCE_CAP = 25.
MICRO_BATCH = 8


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    return {name: sha(ROOT/name) for name in SOURCES}


class AttentionBlock(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.width = width
        self.qkv = nn.Linear(width, 3*width)
        self.projection = nn.Linear(width, width)
        self.norm1 = nn.LayerNorm(width, eps=1e-5)
        self.ff1 = nn.Linear(width, hidden)
        self.ff2 = nn.Linear(hidden, width)
        self.norm2 = nn.LayerNorm(width, eps=1e-5)

    def forward(self, x, valid):
        b, length, width = x.shape
        safe = valid | (~valid.any(dim=1, keepdim=True))
        q, k, value = self.qkv(x).chunk(3, dim=-1)
        def reshape(a):
            return a.reshape(b, length, 4, width//4).transpose(1, 2)
        q, k, value = map(reshape, (q, k, value))
        logits = (q@k.transpose(-1, -2))*((width//4)**-.5)
        logits = logits.masked_fill(~safe[:, None, None, :], -1e9)
        probability = torch.softmax(logits, dim=-1)
        attended = (probability@value).transpose(1, 2).reshape(b, length, width)
        x = self.norm1(x+self.projection(attended))
        return self.norm2(x+self.ff2(torch.relu(self.ff1(x))))


class TinyTransformer(nn.Module):
    def __init__(self, vocabulary_size, use_set):
        super().__init__()
        self.use_set = use_set
        self.embedding = nn.Embedding(vocabulary_size, 32, padding_idx=0)
        self.token = AttentionBlock(32, 64)
        self.hidden = nn.Linear(64+1099, 64)
        if use_set:
            self.set_block = AttentionBlock(64, 128)
        self.output = nn.Linear(64, 1)
        self.register_buffer("position", torch.from_numpy(inference.positions()), persistent=False)
        with torch.no_grad():
            self.embedding.weight.normal_(0, .1)
            self.embedding.weight[0].zero_()

    def forward(self, codes, numeric, valid):
        n, _, length = codes.shape
        flat = codes.reshape(-1, length)
        mask = flat != 0
        x = self.embedding(flat)*(32.**.5)+self.position[:length]
        x = self.token(x, mask)
        maximum = x.masked_fill(~mask[:, :, None], -1e9).amax(dim=1)
        mean = (x*mask[:, :, None]).sum(dim=1)/mask.sum(dim=1, keepdim=True).clamp_min(1)
        joined = torch.cat((maximum, mean, numeric.reshape(-1, 1099)), dim=1)
        hidden = torch.tanh(self.hidden(joined)).reshape(n, 4, 64)
        if self.use_set:
            hidden = self.set_block(hidden, valid)
        return self.output(hidden).reshape(n, 4)


def export(model, transform):
    asset = {"token_embedding": model.embedding.weight.detach().numpy(),
             "set_attention": np.array(model.use_set), **transform}
    for prefix, block in (("token", model.token), ("set", getattr(model, "set_block", None))):
        if block is None:
            continue
        for name in ("qkv", "projection", "norm1", "ff1", "ff2", "norm2"):
            layer = getattr(block, name)
            asset[prefix+"_"+name+"_weight"] = layer.weight.detach().numpy()
            asset[prefix+"_"+name+"_bias"] = layer.bias.detach().numpy()
    for name in ("hidden", "output"):
        layer = getattr(model, name)
        asset[name+"_weight"] = layer.weight.detach().numpy()
        asset[name+"_bias"] = layer.bias.detach().numpy()
    return asset


def native_scores(model, codes, numeric, valid):
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(codes), MICRO_BATCH):
            out.append(model(codes[start:start+MICRO_BATCH], numeric[start:start+MICRO_BATCH], valid[start:start+MICRO_BATCH]).numpy())
    return np.concatenate(out)


class TimeBudget(Exception):
    pass


def train(codes, numeric, structure, valid, fit_ids, use_set, epochs, deadline, held_ids=None):
    weights, gv, good = structure
    torch.manual_seed(7506)
    rng = np.random.default_rng(7507)
    model = TinyTransformer(int(codes.max())+1, use_set)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    best, chosen, stale = float("inf"), 0, 0
    curves = []
    tick = time.perf_counter()
    for epoch in range(1, epochs+1):
        model.train()
        order = rng.permutation(fit_ids)
        total = 0.
        for start in range(0, len(order), 64):
            if time.perf_counter() > deadline:
                raise TimeBudget("Two-hour family runtime cap reached during fitting")
            batch = order[start:start+64]
            optimizer.zero_grad(set_to_none=True)
            for offset in range(0, len(batch), MICRO_BATCH):
                ids = batch[offset:offset+MICRO_BATCH]
                z = model(codes[ids], numeric[ids], valid[ids])
                loss = base.objective(z, weights[ids], gv[ids], good[ids])
                (loss*(len(ids)/len(batch))).backward()
                total += float(loss.detach())*len(ids)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
        row = {"epoch": epoch, "fit_group_ce": total/len(fit_ids), "elapsed_seconds": time.perf_counter()-tick}
        if held_ids is not None:
            model.eval()
            value = 0.
            with torch.no_grad():
                for start in range(0, len(held_ids), MICRO_BATCH):
                    ids = held_ids[start:start+MICRO_BATCH]
                    z = model(codes[ids], numeric[ids], valid[ids])
                    value += float(base.objective(z, weights[ids], gv[ids], good[ids]))*len(ids)
            value /= len(held_ids)
            row["heldout_group_ce"] = value
            if value < best-1e-4:
                best, chosen, stale = value, epoch, 0
            else:
                stale += 1
        curves.append(row)
        if held_ids is not None and stale >= 4:
            break
    return model, curves, chosen if held_ids is not None else epochs


def protocol():
    return {"task": "E3 two bounded randomly initialized token/set-attention models",
        "configs": [{"name": name, "candidate_set_attention": flag} for name, flag in CONFIGS],
        "fold_seeds": list(FOLD_SEEDS), "folds": 3,
        "fold_assignment": "fold[default_rng(seed).permutation(N)] = arange(N)%3",
        "inner_split": "85/15 outer-fitting question IDs with seed outer_seed+100+fold",
        "inputs": "unchanged E1 headtail_roles322 tokens and1099 numerical embedding+text inputs",
        "token_block": {"width": 32, "heads": 4, "layers": 1, "ff_width": 64,
            "activation": "ReLU", "normalization": "post-residual LayerNorm eps1e-5", "dropout": 0.},
        "position": "fixed standard sinusoidal token positions; embedding multiplied by sqrt32; no candidate positions",
        "pooling": "valid-token maximum and mean, concat1099numeric, tanh64",
        "candidate_block": "second variant only: width64/heads4/FF128/layer1 with valid-candidate keys, no candidate positional encoding; all-invalid keys fall back to all four",
        "training": "all models random initialization, seed7506; AdamW lr.001/weight_decay.001; gradientclip5; hard fitting questions only",
        "batching": "effective batch64, gradient accumulation over8-question microbatches; weight each micro loss by its question fraction",
        "objective": "mean answer-group-mean cross entropy",
        "stopping": "inner groupCE, max24epochs/patience4/minimprovement1e-4; fresh refit for selected epochs",
        "transforms": "fit vocabulary/normalization only on applicable inner/outer-fitting inputs, never heldout; roles contain no references",
        "score_scale": "valid outer-fit centered score standard deviation",
        "planned_fits": PLANNED_FITS, "family_runtime_cap_seconds": TIME_CAP,
        "first_fit_probe": "After first planned early-fit, export and time NumPy preprocessing+forward on first467 outer-held TRAIN inputs; no label access or accuracy during probe. If >25seconds, record inference-budget risk and stop family before remaining fits; no architecture change.",
        "probe_scope": "local CPU timing under current load, not official reference-machine qualification; prediction preprocessing+forward only, not fresh-process loading/scoring",
        "runtime_stop_interpretation": "computational/resource limit only, no conclusion that model accuracy cannot improve",
        "selection": "mean hard correct over two repeats, then mean normalized groupCE, then token-only",
        "repeated_questions": "2114 unique hard questions reused, not4228 independent samples; post-selection exploratory evidence",
        "threads": 4, "full_refit_validation_test": False,
        "numpy_export": "mandatory mathematical attention/LayerNorm replay and permutation checks; no torch dependency at inference"}


def progress(stage, completed, detail, started):
    screen.atomic_dump(OUT/"progress.json", {"stage": stage, "completed_fits": completed,
        "planned_fits": PLANNED_FITS, "detail": detail, "elapsed_seconds": time.perf_counter()-started,
        "updated_utc": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"stage": stage, "completed_fits": completed, "detail": detail}), flush=True)


def prepare_registration():
    if (OUT/"protocol.json").exists():
        raise FileExistsError("Registration exists")
    rng = np.random.default_rng(20261026)
    codes = rng.integers(1, 40, (3, 4, 322), dtype=np.int64)
    codes[:, :, 80:] = 0
    numeric = rng.normal(size=(3, 4, 1099)).astype(np.float32)
    valid = np.array([[1,1,1,1], [1,0,1,0], [0,0,0,0]], bool)
    payload = {"codes": codes, "numeric": numeric, "valid": valid}
    ct, nt, vt = map(torch.from_numpy, (codes, numeric, valid))
    records = []
    for name, use_set in CONFIGS:
        torch.manual_seed(7506)
        model = TinyTransformer(40, use_set).eval()
        with torch.no_grad():
            z = model(ct, nt, vt).numpy()
        asset = export(model, {})
        replay = inference.numpy_forward(codes, numeric, valid, asset)
        error = float(np.max(abs(z-replay)))
        assert error < 2e-5, (name, error)
        perm = np.array([2,0,3,1])
        with torch.no_grad():
            zp = model(ct[:, perm], nt[:, perm], vt[:, perm]).numpy()
        perr = float(np.max(abs(zp-z[:, perm])))
        assert perr < 2e-5
        np.savez(OUT/f"{name}_readiness_asset.npz", **asset)
        payload[name] = z
        records.append({"name": name, "torch_numpy_max_abs_error": error,
                        "torch_permutation_max_abs_error": perr, "model_fits": 0})
    np.savez(OUT/"readiness_inputs.npz", **payload)
    dump(OUT/"readiness_checks.json", records)
    dump(OUT/"protocol.json", protocol())
    dump(OUT/"source_evidence.json", sources())
    progress("ready", 0, "synthetic checks done; waiting for NumPy-only check and explicit --run", time.perf_counter())


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
            raise RuntimeError(f"Persistent write outside experiment {path}")
    sys.addaudithook(hook)
    return accesses


def run():
    started = time.perf_counter()
    expected = json.loads((OUT/"source_evidence.json").read_text(encoding="utf-8"))
    assert expected == sources()
    assert json.loads((OUT/"protocol.json").read_text(encoding="utf-8")) == protocol()
    readiness = json.loads((OUT/"numpy_readiness.json").read_text(encoding="utf-8"))
    assert not readiness["torch_imported"] and len(readiness["checks"]) == 2
    if (OUT/"started.json").exists():
        raise FileExistsError("Preserve previous run evidence")
    accesses = audit_io()
    dump(OUT/"started.json", {"utc": datetime.now(timezone.utc).isoformat(), "sources": expected})
    progress("prepare_train_only", 0, "load training bank", started)
    bank, embeddings, labels, hard = development_data("train")
    weights, gv, good, valid, eligible = base.group_structure(bank, labels)
    assert np.array_equal(hard, eligible)
    structure = weights, gv, good
    raw_inputs = original.numeric_inputs(bank, embeddings)
    seqs = variants.sequences(bank, "headtail_roles")
    folds = np.empty((2, len(bank)), np.int8)
    for r, seed in enumerate(FOLD_SEEDS):
        folds[r, np.random.default_rng(seed).permutation(len(bank))] = np.arange(len(bank))%3
    ids = np.array([row["id"] for row in bank])
    raw = np.full((2, len(bank), 4, 2), np.nan, np.float32)
    normalized = raw.copy()
    completed = 0
    records = []
    valid_t = torch.from_numpy(valid)
    probe = None
    try:
        for r, seed in enumerate(FOLD_SEEDS):
            for f in range(3):
                fit_ids, held_ids = np.flatnonzero(folds[r] != f), np.flatnonzero(folds[r] == f)
                order = np.random.default_rng(seed+100+f).permutation(fit_ids)
                cut = int(.85*len(order))
                inner, early_held = order[:cut], order[cut:]
                np.savez_compressed(OUT/f"repeat{r}_fold{f}_indices.npz", fit=fit_ids, held=held_ids, inner=inner, early_held=early_held)
                for c, (name, use_set) in enumerate(CONFIGS):
                    stem = f"repeat{r}_fold{f}_{name}"
                    progress("early_fit", completed, stem, started)
                    codes, numeric, transform, info = screen.prepare(seqs, "headtail_roles", raw_inputs, valid, inner)
                    ct, nt, st = base.tensor_inputs(codes, numeric, structure)
                    model, curve, epochs = train(ct, nt, st, valid_t, inner[eligible[inner]], use_set, 24,
                                                 started+TIME_CAP, early_held[eligible[early_held]])
                    completed += 1
                    dump(OUT/f"{stem}_early_curve.json", curve)
                    early_seconds = curve[-1]["elapsed_seconds"]
                    if completed == 1:
                        probe_asset = export(model, transform)
                        probe_ids = held_ids[:467]
                        assert len(probe_ids) == 467 and not np.intersect1d(probe_ids, inner).size
                        tick = time.perf_counter()
                        scores = inference.score_candidates([bank[i] for i in probe_ids], embeddings[probe_ids], probe_asset)
                        seconds = time.perf_counter()-tick
                        assert scores.shape == (467,4) and np.isfinite(scores).all()
                        probe = {"first_planned_early_fit_seconds": early_seconds,
                            "epochs_executed": len(curve), "selected_epoch": epochs,
                            "numpy_preprocessing_forward_467_training_heldout_seconds": seconds,
                            "probe_indices_sha256": screen.ahash(probe_ids), "accuracy_computed": False,
                            "inference_threshold_seconds": INFERENCE_CAP,
                            "inference_budget_risk": seconds > INFERENCE_CAP,
                            "family_runtime_cap_seconds": TIME_CAP,
                            "interpretation": "Local running-process train-input resource probe only; not official fresh prediction-plus-scoring timing"}
                        dump(OUT/"cost_probe.json", probe)
                        np.savez(OUT/"first_fit_probe_asset.npz", **probe_asset)
                        if seconds > INFERENCE_CAP:
                            dump(OUT/"resource_stop.json", {"reason": "NumPy inference resource risk", "model_fit_calls": completed,
                                "seconds": time.perf_counter()-started, "probe": probe,
                                "does_not_establish_accuracy_ceiling": True})
                            progress("resource_stopped", completed, "467-input NumPy probe exceeded25s; no architecture change", started)
                            return
                    progress("outer_refit", completed, f"{stem}; epochs={epochs}", started)
                    codes, numeric, transform, outer = screen.prepare(seqs, "headtail_roles", raw_inputs, valid, fit_ids)
                    ct, nt, st = base.tensor_inputs(codes, numeric, structure)
                    model, curve, _ = train(ct, nt, st, valid_t, fit_ids[eligible[fit_ids]], use_set, epochs, started+TIME_CAP)
                    completed += 1
                    dump(OUT/f"{stem}_refit_curve.json", curve)
                    z = native_scores(model, ct, nt, valid_t)
                    asset = export(model, transform)
                    scale = float(max(np.std((z[fit_ids]-z[fit_ids].mean(1, keepdims=True))[valid[fit_ids]]),1e-6))
                    asset["score_scale"] = np.array(scale)
                    np.savez(OUT/f"{stem}.npz", **asset)
                    replay = inference.numpy_forward(codes[held_ids[:32]], numeric[held_ids[:32]], valid[held_ids[:32]], asset)
                    error = float(np.max(abs(replay-z[held_ids[:32]])))
                    assert error < 1e-4, error
                    assert select([bank[i] for i in held_ids[:32]], replay, "mean") == select([bank[i] for i in held_ids[:32]], z[held_ids[:32]], "mean")
                    raw[r, held_ids, :, c] = z[held_ids]
                    normalized[r, held_ids, :, c] = z[held_ids]/scale
                    rec = {"repeat": r, "fold": f, "config": name, "selected_epochs": epochs,
                        "early_fit_seconds": early_seconds, "refit_seconds": curve[-1]["elapsed_seconds"],
                        "inner_transform": info, "outer_transform": outer, "score_scale": scale,
                        "asset_bytes": (OUT/f"{stem}.npz").stat().st_size, "asset_sha256": sha(OUT/f"{stem}.npz"),
                        "export_error": error, "metrics": metrics(select([bank[i] for i in held_ids], z[held_ids], "mean"), labels[held_ids], hard[held_ids])}
                    records.append(rec)
                    screen.atomic_dump(OUT/"fit_records.json", records)
                    np.savez_compressed(OUT/"partial_oof.npz", raw=raw, normalized=normalized, ids=ids, folds=folds,
                        labels=labels, valid=valid, hard=hard, configs=np.array([n for n,_ in CONFIGS]))
                    progress("outer_complete", completed, stem, started)
    except TimeBudget as error:
        dump(OUT/"resource_stop.json", {"reason": str(error), "model_fit_calls_completed": completed,
            "partial_fit_started": True, "seconds": time.perf_counter()-started,
            "family_runtime_cap_seconds": TIME_CAP, "does_not_establish_accuracy_ceiling": True,
            "validation_test_access": False})
        progress("resource_stopped", completed, str(error), started)
        return
    assert np.isfinite(raw).all()
    rows = []
    for c, (name, _) in enumerate(CONFIGS):
        repeats = []
        for r in range(2):
            choices = np.asarray(select(bank, raw[r,:,:,c], "mean"))
            baseline = np.asarray(select(bank, raw[r,:,:,0], "mean"))
            right = labels[np.arange(len(bank)), choices].astype(bool)
            reference = labels[np.arange(len(bank)), baseline].astype(bool)
            repeats.append({"seed": FOLD_SEEDS[r], "metrics": metrics(choices, labels, hard),
                "group_ce": base.numpy_group_loss(normalized[r,:,:,c], weights, gv, good, eligible),
                "repairs_vs_token_only": int((hard & right & ~reference).sum()),
                "regressions_vs_token_only": int((hard & reference & ~right).sum())})
        rows.append({"config": name, "repeats": repeats,
            "mean_hard_correct": float(np.mean([v["metrics"]["hard_correct"] for v in repeats])),
            "mean_group_ce": float(np.mean([v["group_ce"] for v in repeats]))})
    selected = min(rows, key=lambda v: (-v["mean_hard_correct"], v["mean_group_ce"], [n for n,_ in CONFIGS].index(v["config"])))
    np.savez_compressed(OUT/"oof_logits.npz", raw=raw, normalized=normalized, ids=ids, folds=folds,
        labels=labels, valid=valid, hard=hard, configs=np.array([n for n,_ in CONFIGS]))
    assert expected == sources()
    result = {"configs": rows, "selected": selected, "model_fit_calls": completed,
        "vocabulary_fits": completed, "numeric_normalizer_fits": completed,
        "max_asset_bytes": max(v["asset_bytes"] for v in records),
        "max_export_error": max(v["export_error"] for v in records),
        "seconds": time.perf_counter()-started, "cost_probe": probe, "data_files_read": sorted(accesses),
        "validation_evaluations": 0, "test_access": False, "full_model_fitted": False,
        "interpretation": "Matched exploratory training-only repeated-fold comparison; no independent generalization claim"}
    dump(OUT/"summary.json", result)
    progress("complete", completed, "selected "+selected["config"], started)
    print(json.dumps(result, indent=2), flush=True)


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
    prepare_registration() if args.prepare_only else run()
