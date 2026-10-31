"""Refit the selected inference heads without rerunning model selection.

Hyperparameters and training-selected epoch counts come from the checkpoint
configuration. Only supplied training references are opened. The output is a
new directory; the submitted checkpoint is never overwritten.
"""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "4"

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from threadpoolctl import threadpool_limits

from common import load_inputs
from ensemble import head_scores, predict
from finalize_validation import score_scale
from text_features import extract_features
import nonlinear_experiment as nonlinear


def fit_head(kind, recipe, bank, raw, features, labels, valid, pair_weights, eligible):
    seed = recipe["seed"]
    if kind == "nonlinear":
        nonlinear.SEED = seed
        mean = raw.reshape(-1, 1024).mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = np.maximum(raw.reshape(-1, 1024).std(axis=0, dtype=np.float64), 1e-8).astype(np.float32)
        model, _, _ = nonlinear.fit((raw[eligible]-mean)/scale, pair_weights[eligible],
                                    recipe["hidden"], recipe["l2"], recipe["epochs"])
        w1 = model[0] / scale[:, None]
        return {"w1": w1, "b1": model[1]-mean@w1, "w2": model[2], "train_mean": mean, "train_scale": scale}
    if kind == "residual":
        import advanced_neural as advanced
        stats = advanced.fit_transform(raw, valid)
        x = advanced.transform(raw[eligible], valid[eligible], stats, False)
        model, _, _ = advanced.fit(x, labels[eligible], valid[eligible], objective=recipe["objective"],
                                  l2=recipe["l2"], hidden=recipe["hidden"], seed=seed, epochs=recipe["epochs"])
        return advanced.export_model(model, stats, False)
    if kind == "joint":
        from advanced_neural import fit
        from joint_experiment import export, statistics
        x = np.concatenate((raw, features), axis=2)
        mean, scale = statistics(x)
        model, _, _ = fit((x[eligible]-mean)/scale, labels[eligible], valid[eligible],
                         objective=recipe["objective"], l2=recipe["l2"], hidden=recipe["hidden"],
                         seed=seed, epochs=recipe["epochs"])
        return export(model, mean, scale)
    if kind == "trees" and not recipe.get("pca"):
        from sklearn.ensemble import HistGradientBoostingClassifier
        from tree_model import export_trees
        estimator = HistGradientBoostingClassifier(max_iter=recipe["rounds"], learning_rate=.05,
            max_leaf_nodes=recipe["leaves"], l2_regularization=recipe["l2"], min_samples_leaf=30,
            early_stopping=False, random_state=seed)
        estimator.fit(features.reshape(-1, features.shape[-1]), labels.ravel())
        asset = export_trees(estimator)
        asset["feature_indices"] = np.arange(features.shape[-1], dtype=np.int32)
        return asset
    raise ValueError(f"No selected training recipe is implemented for {kind}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/model_bundle"))
    parser.add_argument("--output", type=Path, default=Path("runs/reproduced_bundle"))
    parser.add_argument("--record", type=Path, default=Path("runs/selected_training_reproduction.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Output already exists; use a new --output directory")
    config_path = args.checkpoint / "config.json"
    config = json.loads(config_path.read_text())
    started = time.perf_counter()
    bank, arrays = load_inputs("train")
    answers, y, weights, eligible, _ = nonlinear.supervision(bank, "train")
    labels = y.astype(bool)
    valid = np.array([[a is not None for a in row] for row in answers])
    raw = arrays["embeddings"]
    features = extract_features(bank)
    args.output.mkdir(parents=True)
    records = []
    with threadpool_limits(limits=4):
        for head in config["heads"]:
            tick = time.perf_counter()
            asset = fit_head(head["kind"], head["training"], bank, raw, features, labels, valid, weights, eligible)
            np.savez(args.output/head["file"], **asset)
            scale = score_scale(head_scores(head["kind"], bank, raw, features, asset), bank)
            with np.load(args.checkpoint/head["file"], allow_pickle=False) as expected:
                errors = {}
                for key in expected.files:
                    # Structured tree nodes contain several numeric fields.
                    fields = expected[key].dtype.names
                    pairs = [(key+"."+field, asset[key][field], expected[key][field]) for field in fields] if fields else [(key, asset[key], expected[key])]
                    for label, actual_array, expected_array in pairs:
                        errors[label] = (0.0 if np.array_equal(actual_array, expected_array, equal_nan=True)
                                         else float(np.max(np.abs(actual_array.astype(float)-expected_array.astype(float)))))
            records.append({"file": head["file"], "kind": head["kind"], "seconds": time.perf_counter()-tick,
                            "scale": scale, "recorded_scale": head["scale"], "parameter_max_errors": errors,
                            "retrained_sha256": hashlib.sha256((args.output/head["file"]).read_bytes()).hexdigest()})
        shutil.copyfile(config_path, args.output/"config.json")
        # Validation inputs check reproduction; their reference labels are not loaded.
        vb, va = load_inputs("validation")
        expected = predict(vb, va, args.checkpoint)
        actual = predict(vb, va, args.output)
    difference = sum(a != b for a, b in zip(actual, expected))
    result = {"heads": records, "seconds": time.perf_counter()-started, "validation_choice_differences": difference,
              "validation_questions": len(vb), "training_reference_access_only": True,
              "submitted_checkpoint_unchanged": True, "output": str(args.output)}
    args.record.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if difference or any(abs(r["scale"]-r["recorded_scale"]) > 1e-5 for r in records):
        raise SystemExit("Reproduction mismatch; inspect recorded numeric differences")


if __name__ == "__main__":
    main()
