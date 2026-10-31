"""Generate the formal README from finished selection/evaluation evidence.

Preparation does not write README. --write requires the frozen checkpoint,
actual final score/resources, and a checked <=10-page report. No evaluation,
model fitting or data download is performed. The former README is preserved
in the private worklogs before its first replacement.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from freeze_selected import verify
from package_optimized_release import validate_report

ROOT=Path(__file__).resolve().parent.parent


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compose(report_dir):
    frozen=verify(ROOT/'code/runs/optimized_freeze.json')
    decision=read(ROOT/'code/runs/optimization_final_selection/selection.json')
    selected=decision['selected']
    score=read(ROOT/'code/runs/optimization_evaluation/test_score.json')
    resource=read(ROOT/'code/runs/optimization_evaluation/test_resources.json')
    recipe=read(ROOT/'code/configs/selected_recipe.json')
    configuration=read(ROOT/'code'/frozen['checkpoint']/'config.json')
    checkpoint_weights={Path(c['manifest']).parts[1]:c['weight'] for c in configuration['components']}
    validate_report(report_dir)
    if selected['name']!=frozen['selected_name'] or resource['selected_name']!=frozen['selected_name'] or recipe['name']!=selected['name']:
        raise ValueError('Selected identity mismatch')
    if recipe['weights']!=checkpoint_weights:raise ValueError('Permanent recipe weights differ from checkpoint')
    if score['split']!='test' or resource['split']!='test' or resource.get('selection_changed_after_test') is not False:
        raise ValueError('Missing fixed final test evidence')
    if resource['asset_inventory']!=frozen['asset_inventory'] or resource['source_sha256']!=frozen['source_sha256']:
        raise ValueError('Test resources differ from frozen artifact')
    if not 0<=score['correct']<=score['n'] or score['n']<=0 or abs(score['accuracy']-100*score['correct']/score['n'])>1e-6:
        raise ValueError('Inconsistent official score')
    report=report_dir.relative_to(ROOT).as_posix()
    lines=[
        '# Selecting Reliable LLM Reasoning','',
        'DASE7506 Mini-project 2. The selector ranks four supplied solutions using fixed embeddings, answer-group models, cross-candidate numerical relations and a small text CNN trained from scratch. It combines six branches containing ten trained learners. Inference is offline and requires only NumPy.','',
        f"The selected checkpoint scores **{score['correct']}/{score['n']} ({score['accuracy']:.2f}%) on the final test** and **{selected['validation_correct']}/{selected['validation_n']} ({100*selected['validation_correct']/selected['validation_n']:.2f}%) on validation**. Difficult validation questions account for {selected['validation_hard_correct']}/{selected['validation_hard_n']}. The complete fresh prediction process plus independent scoring process took **{resource['total_fresh_process_seconds']:.3f} seconds** locally with four threads. All checkpoint contents total **{resource['asset_uncompressed_bytes']:,} uncompressed bytes**; their actual files occupy **{resource['asset_file_bytes']:,} bytes on disk**. The 16MiB limit applies to the former total.", '',
        'The [frozen checkpoint](code/runs/optimized_bundle/), [selection record](code/runs/optimization_final_selection/selection.json), [official test score](code/runs/optimization_evaluation/test_score.json) and [resource record](code/runs/optimization_evaluation/test_resources.json) identify this result. The local CPU is an AMD Ryzen 5 7600; eligibility is determined on the course reference Xeon Platinum 8457C. The test result was recorded after freezing this method and did not select a rollback.', '',
        f'The [report]({report}/report.pdf) presents the method and findings. [Detailed experiments](docs/optimization_results.md) document matched controls and ablations. Model selection uses a fixed menu of 48 combinations and the [recorded selection policy](docs/optimization_selection_policy.md). OOF, validation and test results are reported separately.', '',
        '## Reproduce inference','',
        'Use Python 3.12. From the repository root, create a local environment, then work inside `code/`. The release omits duplicate course data; the fetch command retrieves only files listed in the supplied manifest and verifies their exact SHA-256 hashes. It is pinned to [upstream commit bb801f4](https://github.com/xudongwu-0/DASE7506-MP2/tree/bb801f4f2e7629d270df765ab12c333b2dc616bb), whose package manifest was verified byte-for-byte against this supplied copy. Existing matching files are retained. A mismatching local file is never overwritten.','',
        '```bash',
        'python3.12 -m venv ../.local/reproduce',
        'source ../.local/reproduce/bin/activate',
        'cd code',
        'python -m pip install -r requirements.txt',
        'python fetch_provided_data.py --splits train validation test --download',
        'python check_data.py',
        'python freeze_selected.py --verify',
        'python measure_selected.py --output-dir runs/reproduction_evaluation',
        '```','',
        'Windows PowerShell, from the repository root:','',
        '```powershell',
        'py -3.12 -m venv ..\\.local\\reproduce',
        '$python = (Resolve-Path "..\\.local\\reproduce\\Scripts\\python.exe").Path',
        'Set-Location code',
        '& $python -m pip install -r requirements.txt',
        '& $python fetch_provided_data.py --splits train validation test --download',
        '& $python check_data.py',
        '& $python freeze_selected.py --verify',
        '& $python measure_selected.py --output-dir runs/reproduction_evaluation',
        '```','',
        'Use a fresh evaluation output directory. `measure_selected.py` sets four thread limits before either child starts, verifies source and asset hashes, and includes both processes\' startup, imports, loading, preprocessing, prediction and scoring in the total time. After installation and data retrieval, prediction and scoring require no network.','',
        'For new, input-only audit questions supplied by the evaluator:','',
        '```bash',
        'python predict.py --method submission.py --checkpoint runs/optimized_bundle --split audit --data-dir /path/to/audit_inputs --output runs/audit_predictions.json',
        '```','',
        'The public `predict(bank, arrays, checkpoint_dir)` computes fresh scores from the provided inputs. It supports variable batch size and fresh question IDs, uses the official answer parser, and reads neither reference answers nor saved predictions during inference.','',
        '## Reproduce the selected training recipe','',
        'The permanent [configuration](code/configs/selected_recipe.json) fixes all six weights. Run training in a new output/work directory; no validation/test evaluation or additional search is part of this command. Install training dependencies in the environment used above, with a separate CPU Torch installation command:','',
        '```bash',
        'python -m pip install -r requirements-train.txt',
        'python -m pip install --index-url https://download.pytorch.org/whl/cpu -r requirements-sequence.txt',
        'python train_optimized.py --recipe configs/selected_recipe.json --output runs/reproduced_optimized --work-dir runs/reproduced_training_work',
        '```','',
        'If using separate numeric and Torch environments, pass `--core-python /path/to/numeric/python` and `--sequence-python /path/to/torch/python`. The command defaults to its current interpreter for both. Only the supplied training split is needed for this refit (`fetch_provided_data.py --splits train --download`). Output contains portable inference assets; logs, fitting predictions and records stay in the distinct work directory.','',
        '| Component (configuration key) | Fixed weight | Learners |', '|---|---:|---:|']
    counts={'Anoise_5':5,'K':1,'Qcross':1,'P':1,'Ccross':1,'roles_cnn':1}
    labels={'Anoise_5':'Group scorer with noise and EMA','K':'Candidate-comparison kernel',
            'Qcross':'Quantitative tree with cross-candidate features','P':'Conditional pair scorer',
            'Ccross':'Condition tree with cross-candidate features','roles_cnn':'Numeric-role CNN'}
    for branch,weight in recipe['weights'].items():lines.append(f'| {labels[branch]} (`{branch}`) | {weight:g} | {counts[branch]} |')
    lines.extend(['',
        'The five `Anoise_5` members use fixed seeds, Gaussian input noise and exponential weight averaging. The other branches combine a candidate-comparison kernel, quantitative and condition trees, a conditional pair scorer, and a numeric-role CNN. Every score scale is fitted on training data. The CNN epoch count follows the rounded median of six internal stopping decisions; the compact source-study protocol, summary, fit records and source hashes are included so this rule can be reproduced without historical fold models. See [numeric refitting](docs/core_refit_protocol.md) and [sequence refitting](docs/optimization_refit_protocol.md).','',
        'The [experiment index](docs/optimization_results.md) maps each reported comparison to its `optimization_*.py` driver, grouped folds, settings and paired results. Run the recorded prerequisites into fresh directories to regenerate omitted OOF arrays and fold models. Experimental fit counts and the independent ten-learner reproduction are accounted for separately.','',
        '## Files and evidence','',
        '| File or directory | Purpose |','|---|---|',
        '| `code/submission.py`, `code/optimized_selection.py` | Public interface, score fusion and answer-group selection |',
        '| `code/optimization_core_refit.py`, `code/optimization_sequence_refit.py` | Training adapters and NumPy-only branch inference |',
        '| `code/cross_candidate_features.py` | Directed derived-number relations and intermediate consensus |',
        '| `code/configs/selected_recipe.json` | Permanent fixed recipe |',
        '| `code/runs/optimized_bundle/` | Complete selected checkpoint |',
        '| `code/runs/optimized_freeze.json` | Exact source, asset and decision hashes |',
        '| `code/runs/optimization_final_selection/` | Frozen validation decision and policy evidence |',
        '| `code/runs/optimization_evaluation/` | Final measured score and complete process/resource evidence |',
        '| `code/fetch_provided_data.py`, `code/PACKAGE_MANIFEST.json` | Official data retrieval and integrity checks |',
        '| `code/package_optimized_release.py`, `OPTIMIZED_RELEASE_ALLOWLIST.txt` | Reviewed public release inventory and exclusive ZIP creation |','',
        'The submitted inference artifact is `optimized_bundle`. The evaluation protocol records validation reuse and prior aggregate test exposure; these limitations apply to interpretation of the results.','',
        '## Provenance and submission','',
        'The project reuses the course data, embeddings, parser, runner and scorer. The original [data license](code/data/LICENSE.txt), [notice](code/data/NOTICE.txt) and [package manifest](code/PACKAGE_MANIFEST.json) are retained. No additional dataset, pretrained checkpoint or evaluation-time model service is used.','',
        'Codex assisted with implementation, experiments, debugging, analysis and report drafting. Claude provided methodological suggestions. [Provenance](docs/provenance.md) records this assistance and reused materials.','',
        'Submission requires code and checkpoint/configuration links, a report of at most ten pages, and reproduction instructions. The MP2 website prepares a GitHub issue, which must be created on GitHub to complete submission. Deadline: 31 October 2026, 23:59 UTC+8. The release contains the selected model and reproducibility materials; private notes, environments, caches and rejected models are excluded.',''])
    return '\n'.join(lines)


def public_guides(report_dir):
    score=read(ROOT/'code/runs/optimization_evaluation/test_score.json')
    resources=read(ROOT/'code/runs/optimization_evaluation/test_resources.json')
    pages=read(report_dir/'layout_checks.json')['pages']
    report=report_dir.relative_to(ROOT).as_posix()
    submission=f'''# Submission contents

The submission artifact is [optimized_bundle](../code/runs/optimized_bundle/), bound by [optimized_freeze.json](../code/runs/optimized_freeze.json). Its official final score is **{score['correct']}/{score['n']}**. Enter **{score['accuracy']}** in the website's accuracy field: this is a percentage, not a fraction.

The final materials are:

- A public code link identifying the submitted revision, with the [README](../README.md), fixed [configuration](../code/configs/selected_recipe.json), requirements and reproduction tools.
- All nested files in the selected checkpoint, or the matching checkpoint ZIP. Preserve the branch directory structure and manifests.
- The [report](../{report}/report.pdf), {pages} pages and within the ten-page limit.
- [Score](../code/runs/optimization_evaluation/test_score.json), [resource evidence](../code/runs/optimization_evaluation/test_resources.json), [selection decision](../code/runs/optimization_final_selection/selection.json) and exact source/asset hashes.
- Accurate [tool and reuse disclosure](provenance.md), original data [license](../code/data/LICENSE.txt) and [notice](../code/data/NOTICE.txt).

Complete checkpoint contents occupy **{resources['asset_uncompressed_bytes']:,} uncompressed bytes**, versus **{resources['asset_file_bytes']:,} actual file bytes**. The course asset limit applies to the uncompressed total. Fresh prediction plus independent scoring took **{resources['total_fresh_process_seconds']:.3f} seconds** on the recorded local CPU with four threads; the reference Xeon determines eligibility.

The [selection policy](optimization_selection_policy.md) preceded the fixed fusion readout and two validation evaluations. The selected checkpoint was frozen before the final test. Validation reuse and prior aggregate test exposure are disclosed in the evaluation protocol.

## Local checks and reproduction

Use the README's installation and provided-data retrieval instructions. `python freeze_selected.py --verify` verifies the deployed files. For independent score reproduction, use `measure_selected.py --output-dir runs/reviewer_evaluation` with a new directory. The training-only `train_optimized.py` command refits the fixed ten-learner recipe without searching or opening validation/test data.

## Complete the GitHub submission

The [MP2 website](https://xudongwu-0.github.io/courses/dase7506/mp2/) needs the student ID, percentage accuracy, public code link and matching checkpoint/configuration link. The final deadline is **31 October 2026, 23:59 Hong Kong/Beijing time (UTC+8)**.

The website prepares a GitHub issue. **Create that issue on GitHub** to finish submission. Preparing a URL or ZIP alone does not submit the work. Include the student ID and matching code/checkpoint links. Private learning materials, worklogs, cache files and rejected models remain outside the publication allowlist.
'''
    release='''# Local release packaging

`OPTIMIZED_RELEASE_ALLOWLIST.txt` enumerates the exact public files. The generated `.gitignore` permits those files and their parent directories and ignores other files by default.

After the report has been rendered and checked, prepare and inspect the inventory from the repository root:

```bash
python code/package_optimized_release.py --refresh-allowlist
```

This verifies the selected freeze, report PDF/hash/page evidence, supplied immutable sources, report dependencies and full nested checkpoint. It writes the exact allowlist and matching `.gitignore`; the previous ignore file is preserved locally. It performs no fitting, evaluation, downloads or upload.

Build the release:

```bash
python code/package_optimized_release.py --build
python code/package_optimized_release.py --audit ../materials/submission/submission.zip
```

`materials/submission/submission.zip` contains code, the selected checkpoint, permanent recipe, compact experiment and evaluation evidence, report/source, provenance and reproduction instructions. This single archive is self-contained apart from supplied course data and software dependencies. Its nested `code/runs/optimized_bundle/` directory provides the checkpoint and configuration link target; a second archive is unnecessary.

The builder uses exclusive ZIP and sidecar creation and never replaces an existing release. Every member has a SHA-256 hash and byte length in the sidecar. Safe relative paths, exact membership, duplicates, symlinks and member hashes are checked without executing the ZIP. Use a new `--output-dir` for another reviewed release.

Course data is not duplicated. `fetch_provided_data.py --splits train validation test --download` retrieves only files from the official course repository at commit `bb801f4f2e7629d270df765ab12c333b2dc616bb`, using the supplied `PACKAGE_MANIFEST.json` hashes. The commit's manifest was verified byte-for-byte; [verification metadata](../code/configs/upstream_data.json) records the check. Training reproduction can request `--splits train` alone. The original license, notice and data manifest are included. Existing mismatching data is never overwritten. No extra dataset or pretrained weights are downloaded.

Only the selected checkpoint is distributed. Fold/rejected models, raw OOF arrays, fitting-score caches, temporary fixtures, environments, private learning documents and worklogs are excluded. Compact CNN epoch-selection metadata is included because the selected training adapter needs it. Omitted research arrays can be regenerated with the corresponding recorded drivers.

Extract the source ZIP into a new local directory for acceptance checks: run `freeze_selected.py --verify` and `build_optimization_report.py --inputs runs/optimization_final_analysis/report_inputs.json --validate-only` from `code/`. Neither command evaluates the test. Optional inference smoke checks use copied training inputs only. No official test rerun is needed to inspect or release the package.

Creating an archive does not submit the project. Submission requires the public artifact links and creation of the course GitHub issue.
'''
    return {'docs/submission.md':submission,'docs/release.md':release}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir',type=Path,default=ROOT/'report')
    parser.add_argument('--write',action='store_true')
    args=parser.parse_args();content=compose(args.report_dir.resolve())
    if not args.write:
        print(content);return
    archive=ROOT.parent/'worklogs/README_before_optimized.md'
    current=ROOT/'README.md'
    if not archive.exists():
        archive.parent.mkdir(parents=True,exist_ok=True)
        with archive.open('xb') as stream:stream.write(current.read_bytes())
    current.write_text(content,encoding='utf-8')
    for name,text in public_guides(args.report_dir.resolve()).items():
        target=ROOT/name
        historical=ROOT.parent/'worklogs'/('before_optimized_'+Path(name).name)
        if target.exists() and not historical.exists():
            with historical.open('xb') as stream:stream.write(target.read_bytes())
        target.write_text(text,encoding='utf-8')
    print(json.dumps({'written':'README.md','sha256':sha(current),'historical_readme_preserved':str(archive)}))


if __name__=='__main__':main()
