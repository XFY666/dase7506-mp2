# Independent-advice experiments: prospective protocol

Registered 26 October 2026, before the new model fits. The user authorized practical testing of the external review's suggestions. Earlier frozen checkpoints, scores, source snapshots and failed gates remain historical evidence and are not rewritten.

## Shared comparison

All new screens fit supplied training questions only. Two fixed three-fold assignments use seeds 20261027 and 20261028: `fold[default_rng(seed).permutation(N)] = arange(N) % 3`, over all 4755 questions. Each question's candidates stay together. Learned transforms, vocabularies, statistics and stopping decisions fit within the outer training boundary. A second split of the same questions is a stability check, not another independent sample of 2114 difficult questions.

Primary screen measures are difficult-question accuracy averaged over the two repeats, paired repairs/regressions for each repeat, and answer-group cross entropy. Report all six fold results. Do not require every fold to improve, and do not treat nominal McNemar significance after model selection as confirmatory. If reporting uncertainty, aggregate repeated outcomes by question and state that fixed-prediction bootstrap uncertainty excludes training and search uncertainty. New configurations must be registered before their own scores are observed; implementation repairs are recorded separately.

Initial parallel screens:

1. **A2 reference reconstruction:** fixed D (three deployed heads), T (pairwise plus text), F and G; fresh outer refits for all constituents. D neural heads use their frozen five-epoch recipes; A/P use previously fixed14/27 epochs. Record these differences rather than calling stopping policies identical. Eight unique learners per partition, six partitions,48 fits.
2. **B2 and D1/D2/D3:** same original75-feature,31-leaf,200-round HGBT controls. Four sampling policies: all candidates, all valid candidates, difficult valid candidates, difficult valid candidates with equal question weight. Three cumulative cross-candidate feature extensions on the last policy. Seven recipes,42 fits. No reference information enters features.
3. **E1 and G2:** four matched joint-CNN input variants: original question prefix and `<num>`; question head/tail; head/tail with input-derived numeric roles; head/tail with formatting normalization. Training-only stopping and fresh outer vocabulary/normalizers for every fit. Four recipes × six partitions × stopping/refit =48 fits. E2/C2 are separate subsequent comparisons.
4. **C1/C3:** add weak supervision from training-reference intermediate quantities to an otherwise matched group network. Reference overlap is a noisy proxy, not step correctness. Register auxiliary weights, masks and stopping before fitting; keep final-answer labels unchanged. Inference never reads references. This family has its own detailed protocol.

## Remaining review directions

A1/A4 selection policy, A3 stopping/EMA, B1 seed averaging, B3 out-of-sample scales, B4 support-pattern fusion, C2/E2 token supervision, C4 training-reference perturbations, D4 bounded expression search, E3 small scratch Transformer, E4 equation graph, F1 projected neighbors, F2 input noise/dropout, and G1 candidate-subset augmentation remain on the experiment register. Each receives a bounded, meaningful comparison or a documented dependency/resource stop; an unperformed direction is not called a failure. Register details before running each family. External data and pretrained weights remain prohibited. No paid service or new hardware is assumed.

## Selection and validation

The proposed rule `difference >= -2*SE` is not a non-inferiority test and will not be used or described as one. Existing D validation311/355 and F303/355 are already known; existing validation has undergone broad selection. New training screens must complete before a final selection protocol names at most two candidates for a new validation check. That protocol must be written before those validation outcomes. It must disclose historical selection and avoid retroactively changing old gates. A full new validation protocol is a separate prospective stage, not permission to repeatedly adjust a candidate until it passes.

No script in this initial screen accesses test inputs, references or scores. A later chosen method must be frozen, verified under the16MiB/four-thread/30-second conditions, and selected independently of test before any final scoring. A final score is reported even if it is lower; test cannot choose a rollback. Formal deployment stays unchanged during screens.

## Records

Each family writes its recipe/source hashes before fits, progress per completed fit, per-fold assets and fit counts, fitting-side transforms/scales, raw and normalized OOF predictions, and a summary. The progress page reads these records and distinguishes implementation, running, complete, failed and pending work. Fit counts are workload accounting, not the number of independent scientific ideas.
