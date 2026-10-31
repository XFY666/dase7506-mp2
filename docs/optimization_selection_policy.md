# Final selection policy for the new optimization round

Registered on 26 October 2026 before computing the 48 joint fusion results and before any new validation evaluation. Earlier protocols and failed selection gates remain historical records. This policy changes the deployment decision prospectively; it does not retroactively certify an earlier rejected method.

## Candidate set and resource screening

The fixed 48-row menu is recorded in `code/runs/optimization_fusion/protocol.json`. It uses fitting-side normalized predictions from two three-fold partitions of the supplied training set. Rank recipes by mean difficult-question correct count, lower mean group cross entropy, fewer positive-weight branches, then menu order. A recipe must be no worse than the fixed historical F recipe in either repeat to enter the ordered shortlist. This is a descriptive stability filter, not a significance test. Two recipes giving the same parsed answers across both repeats occupy one shortlist position.

Fit complete-training checkpoints in that order using the fixed full-refit adapters. Reuse identical trained branches between recipes, without choosing individual random seeds. Before validation, count every uncompressed inference array, statistic, index, vocabulary and configuration. A checkpoint exceeding 16 MiB is ineligible. Measure one fresh prediction process and the independent scorer on a fixed 467-question fixture made from the first 467 supplied training questions. A local total above 25 seconds is ineligible for this selection round; this reserves five seconds against the course's 30-second reference limit. Local timing does not certify performance on the reference Xeon. Preserve failed resource checks and advance to the next already registered recipe, without changing its weights. Numerical/export or source-integrity failures must be repaired and documented before proceeding.

Select at most two distinct resource-eligible checkpoints. Record their exact source, configuration and asset hashes before either new validation evaluation. The complete-training fit uses supplied training inputs and references only. A final recipe represented in old experiments is still evaluated afresh if its checkpoint or transformations differ.

## Validation and the deployment decision

The choice set is the two eligible new checkpoints (or fewer if unavailable) and the unchanged deployed D checkpoint. Evaluate each new checkpoint once on validation. Report full-validation accuracy and difficult-question accuracy separately. The existing D result is 754/798 overall and 311/355 difficult; its repeated training OOF difficult counts are 1729 and 1726.

For a new checkpoint to enter the final choice set, require at least **294/355** difficult validation answers correct. This is an engineering guard against a decline exceeding five percentage points from the known D result: `ceil(311 - 0.05 * 355) = 294`. It is **not** a statistical non-inferiority test, and no confidence-interval or p-value claim is attached to passing it.

Among D and the new checkpoints that pass the guard, maximize the **deployment index**

`mean difficult OOF correct count across the two repeats + difficult validation correct count`.

The two repeats average each of the 2114 training questions' outcomes; they do not create 4228 independent observations. The validation set contributes its 355 difficult questions once. The index deliberately gives each unique development question equal total numerical weight, while allowing the limited new validation evidence to rerank the two contenders. Resolve an exact index tie by smaller measured uncompressed inference size, then lower measured local runtime, then prefer D, then the registered menu order. Freeze the winner before final test scoring. Neither old test score nor a new test result may select a rollback.

## Interpretation and known prior information

The deployment index is an engineering choice criterion, **not an accuracy estimate or calibrated probability**. Report its OOF and validation components separately. Training OOF reflects historically and currently selected recipes trained on two-thirds of training questions, whereas validation evaluates complete-training fits and has already been consulted extensively. Equal numerical weight does not imply equal reliability of the two sources. Searching 48 combinations adds selection bias; the question-cluster bootstrap intervals attached to fixed predictions exclude model-search and training uncertainty.

This policy was written with knowledge of the historical D validation count311 and historical F count303, as well as all completed single-family training screens. The old F count is not assigned to any new full-training checkpoint. The earlier proposal `difference >= -2*SE` is not used or described as non-inferiority. No novel validation result or new fusion outcome informed this policy, and its weights, floor and tie rules will not be altered after observing them.

After the winner is frozen and its interface/integrity checks pass, one final official test evaluation is permitted. Record the result even if it decreases. Publication and submission still require the user's identity, repository choice and explicit publication authorization.
