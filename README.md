# Selecting Reliable LLM Reasoning

DASE7506 Mini-project 2. The selector ranks four supplied solutions using fixed embeddings, answer-group models, cross-candidate numerical relations and a small text CNN trained from scratch. It combines six branches containing ten trained learners. Inference is offline and requires only NumPy.

The selected checkpoint scores **392/467 (83.94%) on the final test** and **750/798 (93.98%) on validation**. Difficult validation questions account for 307/355. The complete fresh prediction process plus independent scoring process took **15.806 seconds** locally with four threads. All checkpoint contents total **9,466,445 uncompressed bytes**; their actual files occupy **9,477,679 bytes on disk**. The 16MiB limit applies to the former total.

The [frozen checkpoint](code/runs/optimized_bundle/), [selection record](code/runs/optimization_final_selection/selection.json), [official test score](code/runs/optimization_evaluation/test_score.json) and [resource record](code/runs/optimization_evaluation/test_resources.json) identify this result. The local CPU is an AMD Ryzen 5 7600; eligibility is determined on the course reference Xeon Platinum 8457C. The test result was recorded after freezing this method and did not select a rollback.

The [report](report/report.pdf) presents the method and findings. [Detailed experiments](docs/optimization_results.md) document matched controls and ablations. Model selection uses a fixed menu of 48 combinations and the [recorded selection policy](docs/optimization_selection_policy.md). OOF, validation and test results are reported separately.

## Reproduce inference

Use Python 3.12. From the repository root, create a local environment, then work inside `code/`. The release omits duplicate course data; the fetch command retrieves only files listed in the supplied manifest and verifies their exact SHA-256 hashes. It is pinned to [upstream commit bb801f4](https://github.com/xudongwu-0/DASE7506-MP2/tree/bb801f4f2e7629d270df765ab12c333b2dc616bb), whose package manifest was verified byte-for-byte against this supplied copy. Existing matching files are retained. A mismatching local file is never overwritten.

```bash
python3.12 -m venv ../.local/reproduce
source ../.local/reproduce/bin/activate
cd code
python -m pip install -r requirements.txt
python fetch_provided_data.py --splits train validation test --download
python check_data.py
python freeze_selected.py --verify
python measure_selected.py --output-dir runs/reproduction_evaluation
```

Windows PowerShell, from the repository root:

```powershell
py -3.12 -m venv ..\.local\reproduce
$python = (Resolve-Path "..\.local\reproduce\Scripts\python.exe").Path
Set-Location code
& $python -m pip install -r requirements.txt
& $python fetch_provided_data.py --splits train validation test --download
& $python check_data.py
& $python freeze_selected.py --verify
& $python measure_selected.py --output-dir runs/reproduction_evaluation
```

Use a fresh evaluation output directory. `measure_selected.py` sets four thread limits before either child starts, verifies source and asset hashes, and includes both processes' startup, imports, loading, preprocessing, prediction and scoring in the total time. After installation and data retrieval, prediction and scoring require no network.

For new, input-only audit questions supplied by the evaluator:

```bash
python predict.py --method submission.py --checkpoint runs/optimized_bundle --split audit --data-dir /path/to/audit_inputs --output runs/audit_predictions.json
```

The public `predict(bank, arrays, checkpoint_dir)` computes fresh scores from the provided inputs. It supports variable batch size and fresh question IDs, uses the official answer parser, and reads neither reference answers nor saved predictions during inference.

## Reproduce the selected training recipe

The permanent [configuration](code/configs/selected_recipe.json) fixes all six weights. Run training in a new output/work directory; no validation/test evaluation or additional search is part of this command. Install training dependencies in the environment used above, with a separate CPU Torch installation command:

```bash
python -m pip install -r requirements-train.txt
python -m pip install --index-url https://download.pytorch.org/whl/cpu -r requirements-sequence.txt
python train_optimized.py --recipe configs/selected_recipe.json --output runs/reproduced_optimized --work-dir runs/reproduced_training_work
```

If using separate numeric and Torch environments, pass `--core-python /path/to/numeric/python` and `--sequence-python /path/to/torch/python`. The command defaults to its current interpreter for both. Only the supplied training split is needed for this refit (`fetch_provided_data.py --splits train --download`). Output contains portable inference assets; logs, fitting predictions and records stay in the distinct work directory.

| Component (configuration key) | Fixed weight | Learners |
|---|---:|---:|
| Group scorer with noise and EMA (`Anoise_5`) | 0.09375 | 5 |
| Candidate-comparison kernel (`K`) | 0.046875 | 1 |
| Quantitative tree with cross-candidate features (`Qcross`) | 0.046875 | 1 |
| Conditional pair scorer (`P`) | 0.375 | 1 |
| Condition tree with cross-candidate features (`Ccross`) | 0.1875 | 1 |
| Numeric-role CNN (`roles_cnn`) | 0.25 | 1 |

The five `Anoise_5` members use fixed seeds, Gaussian input noise and exponential weight averaging. The other branches combine a candidate-comparison kernel, quantitative and condition trees, a conditional pair scorer, and a numeric-role CNN. Every score scale is fitted on training data. The CNN epoch count follows the rounded median of six internal stopping decisions; the compact source-study protocol, summary, fit records and source hashes are included so this rule can be reproduced without historical fold models. See [numeric refitting](docs/core_refit_protocol.md) and [sequence refitting](docs/optimization_refit_protocol.md).

The [experiment index](docs/optimization_results.md) maps each reported comparison to its `optimization_*.py` driver, grouped folds, settings and paired results. Run the recorded prerequisites into fresh directories to regenerate omitted OOF arrays and fold models. Experimental fit counts and the independent ten-learner reproduction are accounted for separately.

## Files and evidence

| File or directory | Purpose |
|---|---|
| `code/submission.py`, `code/optimized_selection.py` | Public interface, score fusion and answer-group selection |
| `code/optimization_core_refit.py`, `code/optimization_sequence_refit.py` | Training adapters and NumPy-only branch inference |
| `code/cross_candidate_features.py` | Directed derived-number relations and intermediate consensus |
| `code/configs/selected_recipe.json` | Permanent fixed recipe |
| `code/runs/optimized_bundle/` | Complete selected checkpoint |
| `code/runs/optimized_freeze.json` | Exact source, asset and decision hashes |
| `code/runs/optimization_final_selection/` | Frozen validation decision and policy evidence |
| `code/runs/optimization_evaluation/` | Final measured score and complete process/resource evidence |
| `code/fetch_provided_data.py`, `code/PACKAGE_MANIFEST.json` | Official data retrieval and integrity checks |
| `code/package_optimized_release.py`, `OPTIMIZED_RELEASE_ALLOWLIST.txt` | Reviewed public release inventory and exclusive ZIP creation |

The submitted inference artifact is `optimized_bundle`. The evaluation protocol records validation reuse and prior aggregate test exposure; these limitations apply to interpretation of the results.

## Provenance and submission

The project reuses the course data, embeddings, parser, runner and scorer. The original [data license](code/data/LICENSE.txt), [notice](code/data/NOTICE.txt) and [package manifest](code/PACKAGE_MANIFEST.json) are retained. No additional dataset, pretrained checkpoint or evaluation-time model service is used.

Codex assisted with implementation, experiments, debugging, analysis and report drafting. Claude provided methodological suggestions. [Provenance](docs/provenance.md) records this assistance and reused materials.

Submission requires code and checkpoint/configuration links, a report of at most ten pages, and reproduction instructions. The MP2 website prepares a GitHub issue, which must be created on GitHub to complete submission. Deadline: 31 October 2026, 23:59 UTC+8. The release contains the selected model and reproducibility materials; private notes, environments, caches and rejected models are excluded.
