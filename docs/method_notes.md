# Methods and interpretation

These notes specify the implemented model families and the meaning of their comparisons. The submitted selector combines a pairwise neural head, a multi-positive listwise neural head, and a structured text classifier. Its [configuration](../code/runs/model_bundle/config.json) and matching checkpoint define the reproducible method. Model development uses training and validation; the [evaluation protocol](extended_protocol.md) documents the earlier baseline test exposure and the subsequent freeze and scoring.

## Task and supervision

Question $q$ has four candidate traces, embeddings $e_{qj}\in\mathbb{R}^{1024}$, and canonical parsed answers $a_{qj}$. The unchanged `common.parse_answer` maps supported numeric forms to a rational-number string; failed extraction returns `None`. For training reference $a_q^*$, the candidate label is

$$y_{qj}=\mathbf{1}[a_{qj}\ne\mathrm{None}\ \land\ a_{qj}=a_q^*].$$

Several candidates may have label one. The official outcome is whether the selected candidate's final answer is correct. It does not certify the validity of every intermediate reasoning step.

All learned parameters and transformations are fitted on training inputs and references. Validation selects configurations. The primary development subset contains at least two distinct valid parsed answers: 355 of the 798 validation questions. This criterion depends only on inputs and matches the documented test-bank construction. Report both complete-validation and disagreement-subset accuracy; their denominators differ.

## Embedding scorers

Per-dimension training means and population standard deviations standardize the embeddings: $x_{qj}=(e_{qj}-\mu)/\sigma$. The pairwise head floors each standard deviation at $10^{-8}$; the listwise head uses $10^{-6}$. Each head fits its own normalization on the permitted fitting questions.

The candidate-classification baseline fits a regularized logistic model to all candidate labels. The pairwise model instead compares every correct candidate with every incorrect candidate whose answer parses. Define $P_q=\{(i,j):y_{qi}=1,y_{qj}=0,a_{qj}\ne\mathrm{None}\}$. For eligible questions $Q'=\{q:|P_q|>0\}$, its data objective is

$$\mathcal{L}_{\mathrm{pair}}=\frac{1}{|Q'|}\sum_{q\in Q'}\frac{1}{|P_q|}\sum_{(i,j)\in P_q}\log\!\left(1+\exp(r_{qj}-r_{qi})\right).$$

Each eligible question has equal total weight. Correct candidates are never contrasted against one another as a positive/negative pair. Linear ranking uses $r_{qj}=w^\top x_{qj}$; the code creates both signs of every pair and rescales sample weights to mean one. The grid varies the scikit-learn inverse regularization parameter $C$.

The nonlinear pairwise head uses $r^{(p)}_{qj}=v^\top\tanh(W^\top x_{qj}+b)$. It trains with Adam and adds L2 gradients for $W$ and $v$, leaving $b$ unpenalized. The recorded pairwise-loss curves exclude the L2 penalty. Early stopping uses an 85/15 split of **training questions**, followed by a fresh full-training refit for the selected number of epochs. The early-stop normalizer uses only the internal fit questions; the final refit normalizer uses all training questions. Candidate rows of one question never cross that internal split.

The listwise head adds a trainable linear skip:

$$r^{(l)}_{qj}=v^\top\tanh(W^\top x_{qj}+b)+u^\top x_{qj}.$$

For valid candidate indices $V_q$ and correct indices $C_q$, the multi-positive loss is

$$\mathcal{L}_{\mathrm{list}}=\frac{1}{|Q'|}\sum_{q\in Q'}\left[\log\sum_{j\in V_q}\exp(r^{(l)}_{qj})-\log\sum_{i\in C_q}\exp(r^{(l)}_{qi})\right].$$

The loss increases the total probability assigned to correct candidates without forcing one correct trace to defeat another. Both neural heads optimize questions containing at least one correct and one incorrect valid candidate. The listwise head uses Adam, with L2 gradients on $W$, $v$ and $u$, and the same training-question stopping/refit procedure. Its recorded loss curves exclude L2. Although the deployment kind is named `residual`, the selected head uses raw standardized embeddings plus the linear skip; the separately tested question-centered contextual input is disabled.

| Deployed head | Training recipe | Asset |
|---|---|---|
| Pairwise neural | tanh width 64, L2 0.01, seed 7506, five full-training epochs | `seed_7506.npz` |
| Listwise neural with linear skip | tanh width 128, L2 0.01, seed 7506, five full-training epochs | `residual.npz` |
| Structured text classifier | 75 features, 200 boosting rounds, at most 31 leaves, L2 10, seed 20261013 | `text.npz` |

The two epoch counts were selected inside training. Their final training executions do not reselect epochs using validation.

For inference, normalization is folded into the weights. A linear model uses $w_{\mathrm{raw}}=w/\sigma$ and $b_{\mathrm{raw}}=b-\mu^\top w_{\mathrm{raw}}$. The nonlinear first layer applies the analogous row-wise scaling and bias correction. The listwise skip also saves $u/\sigma$ and the offset $-\mu^\top(u/\sigma)$. Deployment therefore needs only small matrix operations and the saved arrays.

## Text and relational features

The extractor supplies 75 deterministic features: 21 answer/context features, 23 presentation features, and 31 numerical, lexical, or relational features. Histogram gradient boosting fits candidate correctness labels on these features. The training implementation is scikit-learn; the saved numeric trees are evaluated by `tree_model.py` using NumPy. Exported tree scores are checked against the training library.

Arithmetic features inspect a bounded subset of explicit numeric equalities. The interpreter accepts literal arithmetic through an AST whitelist and does not execute Python expressions. Python-style powers, caret powers, common LaTeX operators, fractions, and Markdown emphasis are normalized where supported. A checked equality passes when its relative error is at most $10^{-6}$. Correct arithmetic can still implement the wrong interpretation of a question; rounding or unsupported notation can also affect these features. Their empirical contribution is measured by ablation.

| Recorded model | Precise interpretation |
|---|---|
| `question_answer_only` | 17 features from the question, parsed answers, and answer-group statistics; four trace-dependent answer features are removed. |
| `answer_context` | All 21 answer/context features, including answer repetition, final-number agreement, conflicting boxes, and same-answer text similarity. |
| `style_only` | A 23-feature presentation scorer, evaluated with the shared parsing and answer-group aggregation. It is not an answer-blind selector. |
| `answer_style` | The 21 answer/context features plus the 23 presentation features. |
| `no_answer_block` | Removes the 21-feature answer/context block. Group-relative features and candidate similarities remain elsewhere, so this is not a complete removal of candidate comparisons. |
| `no_arithmetic` | Removes 12 equality-related features, leaving 63 features. Numeric counts, arithmetic-symbol counts, and fraction indicators remain; describe this as removing equality checks. |
| `text_full` | All 75 features with the base tree configuration. |
| `text_regularized`, `text_deeper` | Fixed comparisons of tree capacity and regularization. |
| `text_pca`, `text_pca_regularized` | The 75 features plus 64 training-fitted embedding projections and 64 within-question centered projections: 203 total inputs. |

The PCA variants use the same saved float32 map $(e-\mu_{\mathrm{PCA}})V^\top$ for training, validation, and deployment. The recorded `features` field in the original text experiment rows counts handcrafted features only; the experiment index explicitly reports the full input width.

## Fusion and answer selection

Each component $k$ has a training-derived score scale

$$\tau_k=\max\!\left(\operatorname{std}\{r^{(k)}_{qj}-\tfrac14\sum_{l=1}^{4}r^{(k)}_{ql}:a_{qj}\ne\mathrm{None}\},10^{-6}\right).$$

Centering uses all four candidate scores; the population standard deviation (`ddof=0`) is then computed over valid parsed training candidates. These scales are fitted on full-training predictions after refitting each head. The submitted selector uses

$$s_{qj}=\frac13\left(\frac{r^{(p)}_{qj}}{1.110068678855896}+\frac{r^{(l)}_{qj}}{1.0948938131332397}+\frac{r^{(t)}_{qj}}{1.6705745566533514}\right).$$

The centered scores determine each scale; inference combines the uncentered scores divided by that scale. Subtracting the same question-level constant from all four scores would not change this answer-group comparison. The three coefficients are equal. Their combination and mean aggregation were selected from the recorded finite development comparisons; no continuous validation weight optimization is used.

For each valid answer group $G_a=\{j:a_{qj}=a\}$, the submitted selector calculates

$$S_q(a)=\frac{1}{|G_a|}\sum_{j\in G_a}s_{qj}.$$

It selects the answer group with highest $S_q(a)$ and returns its highest-scoring candidate. There is no consensus bonus. Deterministic ties prefer the smaller candidate index. Candidates without parsed answers are excluded whenever a valid answer exists; if none parses, the method returns the highest-scoring candidate. Numeric equivalence, rather than text-string equality, defines each group. The output is one native Python integer per question.

Candidate permutation is not claimed to preserve floating-point scores exactly: in the [deployment review](../code/runs/deployment_review.json), small feature changes near tree thresholds produced four same-answer candidate-index changes and no parsed-answer changes across 798 validation questions. This is an observed validation check, not a universal answer-invariance guarantee.

## Historical baselines and fusion comparisons

The original linear/text grid used $s=(1-\alpha)r_{\mathrm{linear}}/\tau_{\mathrm{linear}}+\alpha r_{\mathrm{text}}/\tau_{\mathrm{text}}$. Its nonlinear extension used

$$s=\alpha\frac{r_{\mathrm{text}}}{\tau_{\mathrm{text}}}+(1-\alpha)\left[(1-\beta)\frac{r_{\mathrm{linear}}}{\tau_{\mathrm{linear}}}+\beta\frac{r_{\mathrm{nonlinear}}}{\tau_{\mathrm{nonlinear}}}\right].$$

Weights, aggregation, and any consensus bonus were selected from recorded finite validation grids. The archived compact baseline has $\alpha=0.5$, $\beta=1$, mean aggregation and zero consensus bonus: it combines only the pairwise neural and text heads. Its [configuration and assets](../code/runs/baseline_bundles/compact) remain distinct from the submitted three-head checkpoint. The general experimental selector supported

$$S_q(a)=\operatorname{aggregate}_{j\in G_a}s_{qj}+\gamma\log|G_a|,$$

where aggregation was mean or maximum. This formula describes the comparison family; the submitted method fixes mean and $\gamma=0$. The broader study also fits stacking parameters on training OOF predictions. Those are rejected alternatives, not parameters of the equal-weight submitted method.

## Evidence and resource interpretation

The [research inventory](../code/runs/research_inventory.json), its [CSV companion](../code/runs/research_inventory.csv), and [experiment catalogue](experiments.md) distinguish 333 recorded research model-fit executions from 400 validation evaluation rows and 216 training-only/OOF rows. An early-stop execution and a fresh full-training refit count separately; max/mean aggregation, copied OOF predictions and reused heads do not create fits. Transform fitting is recorded separately where relevant. Interrupted API-error attempts, a smoke probe, private duplicate historical fits and unlogged reruns are outside these scientific-comparison totals, as documented in the inventory. These counts are not a claim to cover every debugging execution.

An independent replay of the already selected recipe performs **three additional verification fits**, not three new research configurations. Its [record](../code/runs/selected_training_reproduction.json) reports zero parameter-array differences, identical score scales and zero choice differences across 798 validation inputs. It reads training references only. From `code/`, install `requirements-train.txt` and run `python train_selected.py --output runs/replay_bundle --record runs/replay_training.json` with a fresh output directory; see the [root README](../README.md) for environment and checkpoint evaluation commands.

OOF experiments use five question-level outer folds, with stopping and preprocessing fitted inside the applicable training boundary. Training-only meta-selection chooses a design before its reported validation score; the best diagnostic validation design is not substituted. Fixed-seed studies average all predefined seeds and do not select a favorable new seed. With mean aggregation, the selected three-head combination obtains 1748/2114 hard training OOF questions versus 1770/2114 for pairwise neural + text ([source](../code/runs/oof_fusion/advanced_extension/base_oof_metrics.json)). Thus its validation improvement is not corroborated by this OOF comparison. The five-seed listwise average is also below the original selected three-head validation score.

The [selection record](../code/runs/final_selection.json) chose the three-head checkpoint at 311/355 hard and 754/798 full validation questions before its subsequent test evaluation. The historical two-head baseline had already been scored at 390/467. The submitted three-head method scores 388/467; it was retained under the pre-scoring commitment, and no test improvement is claimed. The entire study therefore cannot be described as having an untouched test holdout.

Selection margins are score differences, not calibrated probabilities. Risk/coverage curves and empirical accuracy bins should identify their evaluated subset, especially when including unanimous-answer groups. Paired validation differences describe the available evidence after configuration selection; they should not be presented as an untouched confirmatory test.

Official resource measurement includes a fresh prediction process and a fresh independent scoring process, including startup and loading. Only deployed inference assets count toward the asset limit, but every deployed weight, learned transform, statistic, and configuration must be included. Per-model array payload, archive bytes, and full deployed-bundle bytes are distinct quantities. Local timing documents the measured machine; eligibility is determined on the course reference CPU.

Implementation sources: [ensemble.py](../code/ensemble.py), [selection.py](../code/selection.py), [nonlinear_experiment.py](../code/nonlinear_experiment.py), [advanced_neural.py](../code/advanced_neural.py), [train_selected.py](../code/train_selected.py), [train_text.py](../code/train_text.py), [text_features.py](../code/text_features.py), and [tree_model.py](../code/tree_model.py). Historical comparison sources include [embedding_experiment.py](../code/embedding_experiment.py), [finalize_validation.py](../code/finalize_validation.py) and [compare_nonlinear_fusion.py](../code/compare_nonlinear_fusion.py). Task definitions are in the [guide](../guide/GUIDE.md), [official running instructions](../code/README.md), and unchanged [parser](../code/common.py).
