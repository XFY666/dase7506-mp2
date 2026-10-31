# Submission contents

The submission artifact is [optimized_bundle](../code/runs/optimized_bundle/), bound by [optimized_freeze.json](../code/runs/optimized_freeze.json). Its official final score is **392/467**. Enter **83.94004282655246** in the website's accuracy field: this is a percentage, not a fraction.

The final materials are:

- A public code link identifying the submitted revision, with the [README](../README.md), fixed [configuration](../code/configs/selected_recipe.json), requirements and reproduction tools.
- All nested files in the selected checkpoint, or the matching checkpoint ZIP. Preserve the branch directory structure and manifests.
- The [report](../report/report.pdf), 9 pages and within the ten-page limit.
- [Score](../code/runs/optimization_evaluation/test_score.json), [resource evidence](../code/runs/optimization_evaluation/test_resources.json), [selection decision](../code/runs/optimization_final_selection/selection.json) and exact source/asset hashes.
- Accurate [tool and reuse disclosure](provenance.md), original data [license](../code/data/LICENSE.txt) and [notice](../code/data/NOTICE.txt).

Complete checkpoint contents occupy **9,466,445 uncompressed bytes**, versus **9,477,679 actual file bytes**. The course asset limit applies to the uncompressed total. Fresh prediction plus independent scoring took **15.806 seconds** on the recorded local CPU with four threads; the reference Xeon determines eligibility.

The [selection policy](optimization_selection_policy.md) preceded the fixed fusion readout and two validation evaluations. The selected checkpoint was frozen before the final test. Validation reuse and prior aggregate test exposure are disclosed in the evaluation protocol.

## Local checks and reproduction

Use the README's installation and provided-data retrieval instructions. `python freeze_selected.py --verify` verifies the deployed files. For independent score reproduction, use `measure_selected.py --output-dir runs/reviewer_evaluation` with a new directory. The training-only `train_optimized.py` command refits the fixed ten-learner recipe without searching or opening validation/test data.

## Complete the GitHub submission

The [MP2 website](https://xudongwu-0.github.io/courses/dase7506/mp2/) needs the student ID, percentage accuracy, public code link and matching checkpoint/configuration link. The final deadline is **31 October 2026, 23:59 Hong Kong/Beijing time (UTC+8)**.

The website prepares a GitHub issue. **Create that issue on GitHub** to finish submission. Preparing a URL or ZIP alone does not submit the work. Include the student ID and matching code/checkpoint links. Private learning materials, worklogs, cache files and rejected models remain outside the publication allowlist.
