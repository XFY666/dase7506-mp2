# Optimization results and final evaluation

Evidence snapshot: **2026-10-26 10:51 UTC**. This index is generated from saved protocols, summaries and verification records; human review for submission remains pending.

The completed matched experiments show useful gains from cross-candidate derived-number features in trees, EMA/input-noise training, and candidate-subset augmentation. Numeric-role CNN inputs also improve their matched control. Several plausible alternatives have not improved their matched controls. Training-only out-of-fold (OOF) comparisons and full-refit resource qualification appear first; the separate final sections report prospective selection, independent reproduction and frozen test evaluation.

## Shared design and interpretation

In the development screens, every candidate of a question remains in the same outer fold. The two three-fold assignments use seeds `20261027` and `20261028` with `fold[rng.permutation(N)] = arange(N) % 3` over all 4,755 training questions. Each repeat contains the same 2,114 difficult questions. Two repetitions therefore do not create 4,228 independent examples. Normalizers, projections, vocabularies, early stopping and score scales are fitted within the appropriate training boundary. The later finalists stage uses fixed full-training refits and resource measurements, not another OOF screen.

Tables report difficult-question correct counts for repeat 1 / repeat 2, denominator **2,114 each**, and mean normalized answer-group cross entropy (CE; lower is better). Accuracy improvements are judged against the specified matched control. CE is useful within a matched comparison; score conventions differ across some families. New hypotheses followed historical search, and configuration selection within this round is exploratory. Paired question-level resampling describes the saved predictions and omits training/search uncertainty.

The [prospective round protocol](optimization_protocol.md) controls this work. The historical [method notes](method_notes.md) and [extended evaluation protocol](extended_protocol.md) remain unchanged. Earlier development used validation repeatedly and includes prior aggregate test exposure; this round does not make that history disappear. No new validation or test data are read by the experiment and refit drivers indexed here. Deployment promotion is a separate step governed by the [selection policy](optimization_selection_policy.md).

## Experiment ledger

A fit means a completed learner training trajectory, not a saved epoch, a readout, a transform, or an independent scientific finding. Planned counts are upper bounds where identical refits may be reused.

| Study | Directions | State | Completed / planned new learner fits | Evidence |
|---|---|---|---:|---|
| baselines | A1/A2 | complete | 48 / 48 | [protocol](../code/runs/optimization_baselines/protocol.json) · [summary](../code/runs/optimization_baselines/summary.json) · [verification](../code/runs/optimization_baselines/verification.json) · [fit records](../code/runs/optimization_baselines/fit_records.json) · [OOF schema](../code/runs/optimization_baselines/oof_schema.json) |
| trees | B2/D1/D2/D3 | complete | 42 / 42 | [protocol](../code/runs/optimization_trees/protocol.json) · [summary](../code/runs/optimization_trees/summary.json) · [fit records](../code/runs/optimization_trees/fit_records.json) · [OOF schema](../code/runs/optimization_trees/oof_schema.json) |
| process | C1/C3 | complete | 90 / 90 | [protocol](../code/runs/optimization_process/protocol.json) · [summary](../code/runs/optimization_process/summary.json) · [verification](../code/runs/optimization_process/verification.json) · [fit records](../code/runs/optimization_process/fit_records.json) · [OOF schema](../code/runs/optimization_process/oof_schema.json) |
| sequence | E1/G2 | complete | 48 / 48 | [protocol](../code/runs/optimization_sequence/protocol.json) · [summary](../code/runs/optimization_sequence/summary.json) · [fit records](../code/runs/optimization_sequence/fit_records.json) |
| regularization | A3/F2 | complete | 24 / 24 | [protocol](../code/runs/optimization_regularization/protocol.json) · [summary](../code/runs/optimization_regularization/summary.json) · [verification](../code/runs/optimization_regularization/verification.json) · [fit records](../code/runs/optimization_regularization/fit_records.json) |
| interactions | D1/D2 | complete | 60 / 60 | [protocol](../code/runs/optimization_interactions/protocol.json) · [summary](../code/runs/optimization_interactions/summary.json) · [verification](../code/runs/optimization_interactions/verification.json) · [fit records](../code/runs/optimization_interactions/fit_records.json) · [OOF schema](../code/runs/optimization_interactions/oof_schema.json) |
| alternatives | F1/D4 | complete | 12 / 12 | [protocol](../code/runs/optimization_alternatives/protocol.json) · [summary](../code/runs/optimization_alternatives/summary.json) · [verification](../code/runs/optimization_alternatives/verification.json) · [fit records](../code/runs/optimization_alternatives/fit_records.json) · [OOF schema](../code/runs/optimization_alternatives/oof_schema.json) |
| sequence_aux | C2/E2 | complete | 24 / 24 | [protocol](../code/runs/optimization_sequence_aux/protocol.json) · [summary](../code/runs/optimization_sequence_aux/summary.json) · [verification](../code/runs/optimization_sequence_aux/verification.json) · [fit records](../code/runs/optimization_sequence_aux/fit_records.json) |
| calibration | B3/B4 | complete | 144 / 144 | [protocol](../code/runs/optimization_calibration/protocol.json) · [summary](../code/runs/optimization_calibration/summary.json) · [verification](../code/runs/optimization_calibration/verification.json) · [fit records](../code/runs/optimization_calibration/fit_records.json) · [OOF schema](../code/runs/optimization_calibration/oof_schema.json) |
| subset | G1 | complete | 36 / 36 | [protocol](../code/runs/optimization_subset/protocol.json) · [summary](../code/runs/optimization_subset/summary.json) · [verification](../code/runs/optimization_subset/verification.json) · [fit records](../code/runs/optimization_subset/fit_records.json) · [OOF schema](../code/runs/optimization_subset/oof_schema.json) |
| bagging | B1 | complete | 36 / 36 | [protocol](../code/runs/optimization_bagging/protocol.json) · [summary](../code/runs/optimization_bagging/summary.json) · [verification](../code/runs/optimization_bagging/verification.json) · [readiness audit](../code/runs/optimization_bagging/readiness_verification.json) · [fit records](../code/runs/optimization_bagging/fit_records.json) |
| reference_augmentation | C4 | complete | 24 / 24 | [protocol](../code/runs/optimization_reference_augmentation/protocol.json) · [summary](../code/runs/optimization_reference_augmentation/summary.json) · [verification](../code/runs/optimization_reference_augmentation/verification.json) · [fit records](../code/runs/optimization_reference_augmentation/fit_records.json) |
| graph | E4 | complete | 24 / 24 | [protocol](../code/runs/optimization_graph/protocol.json) · [summary](../code/runs/optimization_graph/summary.json) · [verification](../code/runs/optimization_graph/verification.json) · [fit records](../code/runs/optimization_graph/fit_records.json) · [OOF schema](../code/runs/optimization_graph/oof_schema.json) |
| stopping | A3 | complete | 56 / 72 | [protocol](../code/runs/optimization_stopping/protocol.json) · [summary](../code/runs/optimization_stopping/summary.json) · [verification](../code/runs/optimization_stopping/verification.json) · [fit records](../code/runs/optimization_stopping/fit_records.json) |
| transformer | E3 | complete | 24 / 24 | [protocol](../code/runs/optimization_transformer/protocol.json) · [summary](../code/runs/optimization_transformer/summary.json) · [verification](../code/runs/optimization_transformer/verification.json) · [fit records](../code/runs/optimization_transformer/fit_records.json) |
| fusion | A1 | complete | 0 / 0 | [protocol](../code/runs/optimization_fusion/protocol.json) · [summary](../code/runs/optimization_fusion/summary.json) |
| finalists | A1 | complete | 13 / 13 | [protocol](../code/runs/optimization_finalists/protocol.json) · [summary](../code/runs/optimization_finalists/summary.json) · [readiness audit](../code/runs/optimization_finalists/readiness_verification.json) |
| independent reproduction | Final recipe | complete | 10 / 10 | [summary](../code/runs/optimized_reproduction_work/summary.json) · [verification](../code/runs/optimization_reproduction_verification/verification.json) |

Completed studies account for **715 valid new learner fits** at this snapshot. Running or unresolved studies have **0 additional recorded completed fits**; their partial outcomes are not treated as final results.

Distinct workload accounting:

- Reference reconstruction adds 12 PCA and 12 PLS fits. Calibration adds 144 inner learner fits, 36 PCA and 36 PLS fits, while reusing the existing 48 outer learner fits; the reused 48 are not counted again.
- Interactions adds 6 PLS fits. Projected neighbors adds 6 PLS fits and builds 6 retrieval indexes, shared by three neighbor-count readouts; it also fits 12 trees.
- The regularization study has 24 trajectories and 48 saved readouts. Bagging reuses the first two fixed seeds and adds at most 36 trajectories / 54 saved readouts. Bag-size comparisons do not select favorable individual seeds.
- CNN/graph/attention studies count both the inner early-stopping trajectory and the fresh outer refit. Vocabulary and numeric-statistic fits are separately recorded in their summaries. A3 stopping aggregation permits reuse only when one head/outer partition requests the identical epoch count.
- The interrupted baseline attempt has **34 recorded completed invalid fits**, plus one unconfirmed in-progress fit. Boolean labels had made a PLS target constant; no OOF metric or choice was computed before interruption. All 48 valid baseline fits were restarted after an explicit float32-label repair. This invalid workload is excluded from valid counts. See [preserved attempt](../code/runs/optimization_baselines/attempt0_boolean_labels/) and [baseline summary](../code/runs/optimization_baselines/summary.json).
- The token-auxiliary Windows UTF-8 repair occurred before any learner or transform fit: **0 invalid fits**. It changed file decoding only; [repair record](../code/runs/optimization_sequence_aux/encoding_repair.json).

## Direction-by-direction disposition

The IDs below refer to the external review's candidate-direction table; its separate static findings F1–F6 use a different numbering scheme.

| Direction | Implemented comparison and current interpretation |
|---|---|
| A1 | D/T/F/G reconstruction is complete. A finite fusion menu and full-refit resource driver are separately registered. Final promotion follows the [selection policy](optimization_selection_policy.md); this index does not promote a model. |
| A2 | Shared 2×3 grouped folds, paired outcomes and group CE are used across studies; repeated questions are not pooled as independent observations. |
| A3 | Fixed 14/40-epoch, EMA readouts are complete; three-inner-split first/median/rounded-mean stopping is a separate registered study. |
| A4 | The separate predeclared selection protocol evaluated exactly two fixed validation contenders. The suggested rule `delta >= -2*SE` was rejected as a non-inferiority test; the deployed guard is an engineering condition. |
| B1 | Five fixed seeds, with first-two reuse and no best-seed choice; follow the bagging status above. |
| B2 | Four exact-base75 sampling controls show no repeat-stable gain from excluding easy/invalid rows or question weighting alone. |
| B3 | True inner-OOF score scales slightly change decisions and improve CE; this does not explain the much larger F/G-versus-D gap. |
| B4 | Nested support-pattern weights improve the first repeat but not the second against global weights. No B3+B4 combined recipe was run. |
| C1 | Reference-intermediate coverage auxiliary weights 0/0.1/0.3 and hard-only masking were tested with fresh matched controls. No repeat-stable accuracy gain. First-deviation and near-miss variants are not separate tested targets. |
| C2 | Token-level reference-number membership is tested as weak supervision, not ground-truth step validity. |
| C3 | Reference-coverage weighting lowers CE but loses difficult-answer accuracy in both repeats; official final labels are unchanged. |
| C4 | `reference_corruption01` gives 1313/1324 versus matched `digit_text_control` 1296/1327 (+17/-3). Mean group CE 0.784023 → 0.780120. This is the registered matched comparison, not a cross-architecture causal claim. |
| D1 | Derived-number directed containment and early-stop proxies improve the weighted tree control. Question numbers and explicit final markers are excluded. |
| D2 | Distinct-answer-group intermediate consensus adds to D1. The strongest repeat-stable transfers are to trees; group-network accuracy transfer is mixed. Numeric occurrence is not proof of semantic implication. |
| D3 | Target-keyword × last-operation features reduce accuracy relative to D1+D2 in both repeats. |
| D4 | Bounded depth≤3 question-number expression features produce −8/+16 versus matched base; no repeat-stable gain. Search is incomplete, and reachable values need not solve the word problem. |
| E1 | Head/tail context alone loses against the original prefix CNN; adding input-derived numeric roles gains +11/+7 against original and +19/+10 against head/tail. |
| E2 | Token auxiliary versus the same all-question CNN control gives −7/+6, with worse mean CE; different all-question batching prevents attribution to the historical hard-only CNN. |
| E3 | `token_set_attention` gives 1724/1730 versus matched `token_attention` 1732/1747 (-8/-17). Mean group CE 0.441244 → 0.450197. This is the registered matched comparison, not a cross-architecture causal claim. |
| E4 | `equation_gcn` gives 1732/1762 versus matched `node_mlp` 1745/1763 (-13/-1). Mean group CE 0.438507 → 0.438203. This is the registered matched comparison, not a cross-architecture causal claim. |
| F1 | PLS8 projected kNN at k=7/21/63 loses to the matched weighted tree in both repeats. Own-question index entries are excluded for fitting-side calibration. |
| F2 | EMA improves the 14-epoch control; fixed 0.05 Gaussian input noise adds a small accuracy gain. Standalone longer training is inconsistent. Existing dropout remains fixed. |
| G1 | Drop-one views, eligible-only at 0.5 probability, improve the three-seed control by +12/+4 with identical initialization/shuffle/dropout RNG and update budgets. |
| G2 | Formatting normalization gives mixed results against original prefix CNN; its direct head/tail comparison is positive. It is a deterministic normalization comparison, not stochastic augmentation. |

## Completed result readouts

Full fold-level repairs, regressions and traces remain in each linked summary/fit record. These compact tables retain all registered per-study readouts; the finite fusion menu is linked in full and its first two ranked entries are shown.

### Fixed D/T/F/G reference reconstruction

[protocol](../code/runs/optimization_baselines/protocol.json) · [summary](../code/runs/optimization_baselines/summary.json) · [verification](../code/runs/optimization_baselines/verification.json) · [fit records](../code/runs/optimization_baselines/fit_records.json) · [OOF schema](../code/runs/optimization_baselines/oof_schema.json)

| Recipe | Repeat 1 correct | Repeat 2 correct | Mean group CE |
|---|---:|---:|---:|
| `D` | 1729 | 1726 | 0.471037 |
| `T` | 1737 | 1754 | 0.455661 |
| `F` | 1759 | 1770 | 0.405129 |
| `G` | 1762 | 1767 | 0.415277 |

### Matched sampling policies and cross-candidate features

[protocol](../code/runs/optimization_trees/protocol.json) · [summary](../code/runs/optimization_trees/summary.json) · [fit records](../code/runs/optimization_trees/fit_records.json) · [OOF schema](../code/runs/optimization_trees/oof_schema.json)

| Recipe | Repeat 1 correct | Repeat 2 correct | Mean group CE |
|---|---:|---:|---:|
| `base_all_candidates` | 1661 | 1650 | 0.505005 |
| `base_all_valid` | 1654 | 1652 | 0.508491 |
| `base_hard_valid` | 1647 | 1663 | 0.507214 |
| `base_hard_valid_question_weighted` | 1659 | 1645 | 0.504100 |
| `weighted_D1` | 1685 | 1672 | 0.492867 |
| `weighted_D1_D2` | 1693 | 1676 | 0.490150 |
| `weighted_D1_D2_D3` | 1678 | 1669 | 0.491415 |

### Reference-process weak supervision

[protocol](../code/runs/optimization_process/protocol.json) · [summary](../code/runs/optimization_process/summary.json) · [verification](../code/runs/optimization_process/verification.json) · [fit records](../code/runs/optimization_process/fit_records.json) · [OOF schema](../code/runs/optimization_process/oof_schema.json)

| Recipe | Repeat 1 correct | Repeat 2 correct | Mean group CE |
|---|---:|---:|---:|
| `control_allq` | 1764 | 1762 | 0.410519 |
| `coverage01_allq` | 1764 | 1761 | 0.410576 |
| `coverage03_allq` | 1765 | 1759 | 0.410468 |
| `coverage01_hardq` | 1767 | 1758 | 0.410593 |
| `coverage_weight` | 1762 | 1758 | 0.409269 |

### Question context, numeric roles and formatting

[protocol](../code/runs/optimization_sequence/protocol.json) · [summary](../code/runs/optimization_sequence/summary.json) · [fit records](../code/runs/optimization_sequence/fit_records.json)

| Recipe | Repeat 1 correct | Repeat 2 correct | Mean group CE |
|---|---:|---:|---:|
| `head64` | 1722 | 1743 | 0.445177 |
| `headtail` | 1714 | 1740 | 0.446885 |
| `headtail_roles` | 1733 | 1750 | 0.437092 |
| `headtail_format` | 1727 | 1742 | 0.444651 |

### Longer training, EMA and input noise

[protocol](../code/runs/optimization_regularization/protocol.json) · [summary](../code/runs/optimization_regularization/summary.json) · [verification](../code/runs/optimization_regularization/verification.json) · [fit records](../code/runs/optimization_regularization/fit_records.json)

| Recipe | Repeat 1 correct | Repeat 2 correct | Mean group CE |
|---|---:|---:|---:|
| `epoch14` | 1749 | 1750 | 0.412336 |
| `epoch40` | 1768 | 1746 | 0.415062 |
| `epoch40_ema` | 1769 | 1759 | 0.408941 |
| `epoch40_ema_noise` | 1772 | 1760 | 0.408966 |

### Transfer to group networks and PLS trees

[protocol](../code/runs/optimization_interactions/protocol.json) · [summary](../code/runs/optimization_interactions/summary.json) · [verification](../code/runs/optimization_interactions/verification.json) · [fit records](../code/runs/optimization_interactions/fit_records.json) · [OOF schema](../code/runs/optimization_interactions/oof_schema.json)

| Recipe | Repeat 1 correct | Repeat 2 correct | Mean group CE |
|---|---:|---:|---:|
| `A_base` | 1755 | 1759 | 0.409420 |
| `A_D1D2` | 1770 | 1755 | 0.405241 |
| `Q_base` | 1732 | 1728 | 0.439151 |
| `Q_D1D2` | 1746 | 1748 | 0.430327 |
| `C_base` | 1723 | 1709 | 0.448371 |
| `C_D1D2` | 1737 | 1727 | 0.437432 |

### Projected neighbors and bounded arithmetic search

[protocol](../code/runs/optimization_alternatives/protocol.json) · [summary](../code/runs/optimization_alternatives/summary.json) · [verification](../code/runs/optimization_alternatives/verification.json) · [fit records](../code/runs/optimization_alternatives/fit_records.json) · [OOF schema](../code/runs/optimization_alternatives/oof_schema.json)

| Recipe | Repeat 1 correct | Repeat 2 correct | Mean group CE |
|---|---:|---:|---:|
| `PLS8_neighbors_k7` | 1520 | 1530 | 0.652597 |
| `PLS8_neighbors_k21` | 1564 | 1547 | 0.592103 |
| `PLS8_neighbors_k63` | 1560 | 1558 | 0.574943 |
| `weighted_base75` | 1659 | 1645 | 0.504100 |
| `weighted_base75_expression` | 1651 | 1661 | 0.504175 |

### Token process auxiliary supervision

[protocol](../code/runs/optimization_sequence_aux/protocol.json) · [summary](../code/runs/optimization_sequence_aux/summary.json) · [verification](../code/runs/optimization_sequence_aux/verification.json) · [fit records](../code/runs/optimization_sequence_aux/fit_records.json)

| Recipe | Repeat 1 correct | Repeat 2 correct | Mean group CE |
|---|---:|---:|---:|
| `control_allq` | 1752 | 1744 | 0.434641 |
| `token_aux01` | 1745 | 1750 | 0.436743 |

### Nested score scales and support-pattern fusion

[protocol](../code/runs/optimization_calibration/protocol.json) · [summary](../code/runs/optimization_calibration/summary.json) · [verification](../code/runs/optimization_calibration/verification.json) · [fit records](../code/runs/optimization_calibration/fit_records.json) · [OOF schema](../code/runs/optimization_calibration/oof_schema.json)

| Recipe | Repeat 1 correct | Repeat 2 correct | Mean group CE |
|---|---:|---:|---:|
| `D` | 1729 | 1726 | 0.471037 |
| `T` | 1737 | 1754 | 0.455661 |
| `F` | 1759 | 1770 | 0.405129 |
| `G` | 1762 | 1767 | 0.415277 |
| `D_crossfit_scale` | 1729 | 1735 | 0.466051 |
| `T_crossfit_scale` | 1735 | 1755 | 0.450163 |
| `F_crossfit_scale` | 1760 | 1770 | 0.402093 |
| `G_crossfit_scale` | 1761 | 1770 | 0.412482 |
| `global_fusion` | 1760 | 1770 | 0.410988 |
| `pattern_shrink_fusion` | 1767 | 1768 | 0.414443 |

### Matched candidate-subset augmentation

[protocol](../code/runs/optimization_subset/protocol.json) · [summary](../code/runs/optimization_subset/summary.json) · [verification](../code/runs/optimization_subset/verification.json) · [fit records](../code/runs/optimization_subset/fit_records.json) · [OOF schema](../code/runs/optimization_subset/oof_schema.json)

| Recipe | Repeat 1 correct | Repeat 2 correct | Mean group CE |
|---|---:|---:|---:|
| `original` | 1755 | 1759 | 0.409106 |
| `drop_one_half` | 1767 | 1763 | 0.404963 |

### Five fixed seeds; reuse first two

[protocol](../code/runs/optimization_bagging/protocol.json) · [summary](../code/runs/optimization_bagging/summary.json) · [verification](../code/runs/optimization_bagging/verification.json) · [readiness audit](../code/runs/optimization_bagging/readiness_verification.json) · [fit records](../code/runs/optimization_bagging/fit_records.json)

| Recipe | Repeat 1 correct | Repeat 2 correct | Mean group CE |
|---|---:|---:|---:|
| `epoch14_bag1` | 1748 | 1760 | 0.417132 |
| `epoch14_bag2` | 1749 | 1750 | 0.412336 |
| `epoch14_bag3` | 1755 | 1759 | 0.409420 |
| `epoch14_bag5` | 1764 | 1749 | 0.408577 |
| `epoch40_ema_bag1` | 1769 | 1758 | 0.411682 |
| `epoch40_ema_bag2` | 1769 | 1759 | 0.408941 |
| `epoch40_ema_bag3` | 1771 | 1761 | 0.407950 |
| `epoch40_ema_bag5` | 1776 | 1760 | 0.406986 |
| `epoch40_ema_noise_bag1` | 1771 | 1758 | 0.412098 |
| `epoch40_ema_noise_bag2` | 1772 | 1760 | 0.408966 |
| `epoch40_ema_noise_bag3` | 1768 | 1760 | 0.407935 |
| `epoch40_ema_noise_bag5` | 1773 | 1761 | 0.406907 |

### Pure-text reference perturbation auxiliary

[protocol](../code/runs/optimization_reference_augmentation/protocol.json) · [summary](../code/runs/optimization_reference_augmentation/summary.json) · [verification](../code/runs/optimization_reference_augmentation/verification.json) · [fit records](../code/runs/optimization_reference_augmentation/fit_records.json)

| Recipe | Repeat 1 correct | Repeat 2 correct | Mean group CE |
|---|---:|---:|---:|
| `digit_text_control` | 1296 | 1327 | 0.784023 |
| `reference_corruption01` | 1313 | 1324 | 0.780120 |

### Equation-node MLP versus GCN

[protocol](../code/runs/optimization_graph/protocol.json) · [summary](../code/runs/optimization_graph/summary.json) · [verification](../code/runs/optimization_graph/verification.json) · [fit records](../code/runs/optimization_graph/fit_records.json) · [OOF schema](../code/runs/optimization_graph/oof_schema.json)

| Recipe | Repeat 1 correct | Repeat 2 correct | Mean group CE |
|---|---:|---:|---:|
| `node_mlp` | 1745 | 1763 | 0.438507 |
| `equation_gcn` | 1732 | 1762 | 0.438203 |

### Aggregate three inner stopping decisions

[protocol](../code/runs/optimization_stopping/protocol.json) · [summary](../code/runs/optimization_stopping/summary.json) · [verification](../code/runs/optimization_stopping/verification.json) · [fit records](../code/runs/optimization_stopping/fit_records.json)

| Recipe | Repeat 1 correct | Repeat 2 correct | Mean group CE |
|---|---:|---:|---:|
| `D_fixed5` | 1729 | 1726 | 0.471037 |
| `D_first` | 1735 | 1735 | 0.469654 |
| `D_median3` | 1738 | 1734 | 0.469137 |
| `D_rounded_mean3` | 1733 | 1734 | 0.469400 |

### Scratch attention model and candidate-set attention

[protocol](../code/runs/optimization_transformer/protocol.json) · [summary](../code/runs/optimization_transformer/summary.json) · [verification](../code/runs/optimization_transformer/verification.json) · [fit records](../code/runs/optimization_transformer/fit_records.json)

| Recipe | Repeat 1 correct | Repeat 2 correct | Mean group CE |
|---|---:|---:|---:|
| `token_attention` | 1732 | 1747 | 0.441244 |
| `token_set_attention` | 1724 | 1730 | 0.450197 |

### Finite fusion readouts; no new learner fits

[protocol](../code/runs/optimization_fusion/protocol.json) · [summary](../code/runs/optimization_fusion/summary.json)

The first two distinct entries in the fixed ranking are listed here; all 48 menu rows remain in the summary. Their resource assessment appears below. OOF ranking alone does not promote a deployed model.

| Recipe | Repeat 1 correct | Repeat 2 correct | Mean group CE |
|---|---:|---:|---:|
| `F_subset_cross` | 1783 | 1777 | 0.397342 |
| `G_noise_cross__roles_cnn_0.25` | 1775 | 1782 | 0.410571 |

### Fixed full-training resource qualification

[protocol](../code/runs/optimization_finalists/protocol.json) · [summary](../code/runs/optimization_finalists/summary.json) · [readiness audit](../code/runs/optimization_finalists/readiness_verification.json)

The two ranked recipes passed the fixed 467-training-question resource fixture. Times include a fresh prediction process plus a fresh independent scoring process, with four threads. The fixture checks execution and resources; its accuracy is not used for selection. Entire uncompressed checkpoint bytes include manifests and configuration as well as NPZ contents.

| Candidate | Mean OOF difficult correct | Uncompressed bytes | Full probe seconds |
|---|---:|---:|---:|
| `F_subset_cross` | 1780.0 | 5,601,397 | 7.085 |
| `G_noise_cross__roles_cnn_0.25` | 1778.5 | 9,466,445 | 13.741 |

There were 13 completed full-training learner fits across 7 unique branches. The immutable [shortlist](../code/runs/optimization_finalists/shortlist.json) records exact checkpoint and inference-source hashes before validation. Neither resource pass is a validation or test score.

## Matched conclusions and verification

D1+D2 improves the exact weighted base75 tree from 1659/1645 to 1693/1676 (+34/+31), with gains in all six folds and mean CE 0.504100 → 0.490150. Adding D3 reduces this result to 1678/1669. The strong-model transfer is +14/+20 for Q and +14/+18 for C, while the three-seed group model changes +15/−4. This supports a useful tree feature block, not universal gains for every architecture.

The G1 control and treatment score 1755/1759 and 1767/1763. Independent artifact verification reconstructs every one of the 36 members, all six fitting-only normalizers, member and ensemble scales, augmentation counts, and matched initialization/RNG/update fingerprints. The control also reproduces the separately fitted interaction-control member predictions. [Subset verification](../code/runs/optimization_subset/verification.json).

The alternatives audit reconstructs all six projected indexes, their valid-hard fitting rows and labels, excludes held questions, replays all neighbor scores and fitting-side leave-own-question-out scales, and replays all 12 trees. Its weighted base75 raw scores, normalized scores and scales are bitwise identical to the B2 control. The arithmetic search retains at most 971 states and uses at most 5922 attempt slots in this bank; 99.16% of questions hit its depth-2 retention cap. Its inconclusive result applies to this bounded search. [Alternatives verification](../code/runs/optimization_alternatives/verification.json).

The graph audit verifies all 12 saved outer assets and all 24 inner/refit boundaries, including fitting-only normalizers, early-stop choices, source/input hashes and 32 reconstructed graph/numeric feature rows. Every held-out NumPy score and fitting scale reproduces exactly. The matched graph edges change hard-answer counts by −13/−1 versus node pooling. [Graph verification](../code/runs/optimization_graph/verification.json).

A completed run's own export check and a separate saved-artifact verification are different evidence levels. A link labeled readiness audit covers pre-run checks only. Pending verification is not silently promoted to completed verification. All checks in this index were local automated checks; human scientific review remains pending.

## Reproduction and handoff

The sole formal repository is this repository. Executable sources live in `code/`; each `code/runs/optimization_*` directory holds its immutable protocol, hashes, fold assets, records and OOF output. Never overwrite a completed run to change a recipe. Preserve failed-attempt evidence and register a new comparison when the method changes.

The local queue and environment paths are private operational assets outside the repository. `optimization_structure_refit.py` and `optimization_sequence_refit.py` expose explicit full-training refit and NumPy-only inference adapters; importing them does not fit a model. Refits must follow the selected recipe and later resource protocol. No full-training asset, new validation result or test result is created by this index generator.

This index supplements the formal report and does not replace it. Submit only the selected source/configuration/checkpoint, report and reproduction instructions after review. Private learning material, caches, full working logs and every historical checkpoint do not belong in a public submission by default. Existing [provenance/disclosure](provenance.md) remains applicable.


## Full-refit logging recovery

The first finalists attempt stopped while writing P's fit metadata: the dictionary supplied `seconds` twice. P's learner and asset export had already completed. The saved asset was recovered without another fit; the missing original duration/curve remain unavailable rather than being estimated. Cumulative completed full fits at recovery were 11, correcting the driver's earlier 10. This is one completed fit with a logging failure, not a discarded model comparison.

The original driver/core sources, protocol, failed status and branch records are preserved. The minimal core change affects the logging block; inference AST equality and the exact old/new hashes bind the compatibility exception. Existing branch assets, the first assembled checkpoint and its first resource timing remain unchanged. The explicit resume verifies the already-assessed ranked prefix and does not rerun its prediction/scoring timing. [Recovery evidence](../code/runs/optimization_finalists/repair_logging/P_recovery.json), [core compatibility](../code/runs/optimization_finalists/repair_logging/core_compatibility.json), [resume readiness](../code/runs/optimization_finalists/repair_logging/driver_resume_readiness.json).


## Final selection: separate validation evidence

The prospective policy was fixed before the 48-combination outcomes and the two new validation evaluations. Its engineering index adds mean difficult-OOF correct count across the two repeats to difficult-validation correct count, subject to a 294/355 validation guard and resource limits. It is not an independent accuracy estimate or a non-inferiority test. Historical D=311/355 and F=303/355 validation results were already known and disclosed.

| Fixed candidate | Mean OOF difficult correct | Validation difficult / 355 | Full validation / 798 | Deployment index |
|---|---:|---:|---:|---:|
| `F_subset_cross` | 1780.0 | 304 | 747 | 2084.0 |
| `G_noise_cross__roles_cnn_0.25` | 1778.5 | 307 | 750 | 2085.5 |
| Historical D | 1727.5 | 311 | 754 | 2038.5 |

The selected recipe is `G_noise_cross__roles_cnn_0.25`. Its index is 2085.5; its validation result is 750/798 overall and 307/355 difficult. This is four difficult-validation questions below historical D. The policy combines training and validation evidence rather than maximizing the previously observed validation score. [Selection record](../code/runs/optimization_final_selection/selection.json), [prevalidation freeze](../code/runs/optimization_final_selection/prevalidation_freeze.json).

## Independent reproduction of the selected recipe

A fresh portable run of `train_optimized.py` fitted all 10 selected learners. All ten NPZ files are byte-identical: 91 arrays, including the vocabulary and 17 scale arrays, match exactly. Fresh predictions and scores on 32 training-input questions also match exactly (maximum difference 0). Timing and documented logging-repair provenance can differ; no validation or test evaluation was used for this comparison. [Reproduction verification](../code/runs/optimization_reproduction_verification/verification.json), [training summary](../code/runs/optimized_reproduction_work/summary.json).

The complete round contains **692 development + 13 finalist + 10 independent reproduction = 715 valid new learner fits**, plus 34 separately recorded invalid completed fits. The seven distinct finalist branches contain 13 learners because one three-seed branch and one five-seed branch are alternatives; the selected six-branch recipe contains ten learners. Transform fits and readout snapshots remain separate.

## Frozen final test and resources

The selected method obtained **392/467 = 83.940043%** in the sole new official test evaluation. Historical D scored 388/467, so the observed change is +4 questions (+0.856531 percentage points). This is a descriptive comparison on this test set, not a claim of statistically significant improvement. No model or source changes followed the test.

Complete four-thread evaluation took **15.805992 seconds** locally, including fresh prediction and independent scoring processes, startup/imports, asset loading and preprocessing. The complete checkpoint contains **9,466,445 uncompressed bytes (9.0279 MiB)**, below 16 MiB. These are local measurements on AMD Ryzen 5 7600; the course reference CPU determines official eligibility.

Selection at 10:26:59 UTC preceded source/asset freeze at 10:44:16 and test start at 10:44:23 on 2026-10-26. All 38 prevalidation model sources and all 45 final frozen source files remain exact. [Official score](../code/runs/optimization_evaluation/test_score.json), [complete resources](../code/runs/optimization_evaluation/test_resources.json), [freeze](../code/runs/optimized_freeze.json), [final audit](../code/runs/optimization_final_analysis/final_audit.json).
