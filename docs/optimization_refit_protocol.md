# Fixed full-training refit policy for sequence branches

This protocol defines implementation and refitting only. It does not select a branch, authorize a validation/test evaluation, or replace the final selection policy managed separately. Preparation performs zero full-data fits.

## Supported branches

| Branch | Completed-source study required | Full-fit optimizer questions | Inference |
|---|---|---|---|
| `headtail_roles` | `optimization_sequence`, matching recipe | Eligible hard questions | Role-token joint CNN |
| `control_allq` / `token_aux01` | `optimization_sequence_aux`, matching configuration | All training questions; main loss on eligible questions | Same role-token joint CNN; auxiliary head omitted |
| `token_attention` / `token_set_attention` | `optimization_transformer`, matching configuration | Eligible hard questions | Token Transformer, optionally candidate-set attention |
| `digit_text_control` / `reference_corruption01` | `optimization_reference_augmentation`, matching configuration | All training questions; main loss on eligible questions | Digit-preserving pure-text CNN; auxiliary head omitted |

Every source study must have its complete registered fit count, six unique outer-fold records for the branch, and matching frozen source hashes. A partially completed or resource-stopped study cannot supply full-refit epochs. Supporting a branch in code does not mean it has been selected or fully trained.

## Epoch and fitting rule

Use the six selected epoch counts from two repeats × three outer folds. Set

`epochs = max(1, floor(median(six selected epochs) + 0.5))`.

This rule is fixed before full fitting. It adds no new stopping split, sweep, seed selection or heldout evaluation. Initialize a fresh model with seed 7506 and shuffle seed 7507. Reuse the original branch's training function, optimizer, batches and objective; fit once for the fixed epoch count. All models run on CPU with four compute threads. Existing two-hour training guards remain on Transformer and reference-corruption calls.

Fit the vocabulary and, where applicable, numeric mean/scale on all 4,755 supplied training questions. For the digit-text study, both arms retain the registered actual-candidate-plus-training-reference-pair vocabulary policy. Auxiliary targets are derived only from those training references and are used only in training. Original final-answer candidate labels are unchanged. The API checks the supplied bank, labels, references and dense embeddings against the course training files before fitting.

## Artifacts and inference

`code/optimization_sequence_refit.py` provides:

```python
manifest_path = train_branch(name, output_directory, training_bank, emb, labels, refs=None)
raw_logits = score_branch(bank, emb, manifest_path)
normalized_logits = score_branch(bank, emb, manifest_path, normalize=True)
```

Training requires a new output directory and exports `model.npz`, `manifest.json`, the fixed plan, training curve and progress. The manifest records source/evidence hashes, all six epoch counts, one model fit, vocabulary/normalizer counts, full-fit wall time and process CPU time, training-only data access and a fresh NumPy export replay. It preserves the fitting-side score scale used by the original OOF pipeline: standard deviation of scores centered over the four candidates, measured only at valid training candidates.

Inference computes fresh scores from the supplied question/candidates/embeddings and exported parameters. It reads no training data, references or stored predictions. Import and scoring require NumPy and the standard library, with no Torch/scikit-learn training imports. `score_branch` returns raw logits by default; `normalize=True` divides by the recorded training-side score scale. The root selection/fusion code must choose the convention consistent with its OOF inputs. Checksum checks bind the asset and inference source to the manifest.

The standalone default `predict` uses mean aggregation over parsed answer groups. It does not specify ensemble weights or acceptance criteria. A single branch larger than 16 MiB is rejected; total ensemble asset size and fresh-process runtime still require separate final qualification.

## Explicit launch, after selection

From `repo/code`, in the CPU training environment:

```powershell
& '..\..\.local\sequenceenv\Scripts\python.exe' -u optimization_sequence_refit.py --fit --branch headtail_roles --output runs/selected_sequence/headtail_roles
```

The branch and output above are examples, not an automatic selection. Preparation and source-study inspection use `--self-check --output runs/optimization_sequence_refit` and `--describe --branch NAME`; neither trains a model. A branch whose source study is incomplete is rejected before fitting.
