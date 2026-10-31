# Local release packaging

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
