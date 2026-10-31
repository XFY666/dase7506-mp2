# Core branch refit protocol

The new `code/optimization_core_refit.py` prepares complete-training versions of fixed core branches. Preparation does not run training or access validation/test data. A later explicit `train_branch(name, outdir, bank, embeddings, labels)` call creates one new branch directory. The inputs must be the unchanged complete official training bank. Existing output directories are refused.

Supported names are `A`, `K`, `Q`, `P`, `C`, `A5_epoch14`, `A5_ema40`, `A5_noise_ema40`, `A_D1D2`, `Q_D1D2`, `C_D1D2`, and `drop_one_half`. PCA/bandwidth transformation seed is fixed at 7506. Tree fitting keeps its original seed 20261015. Historical group and pair networks retain their original fixed initialization 7506.

The five-member bags use seeds 7506, 17001, 17002, 17003, 17004. They use the exact registered `optimization_regularization.checkpoint_fit` trajectory and its epoch14 or epoch40 EMA readout. EMA starts after epoch5 with decay0.99; the noise branch adds Gaussian input noise with standard deviation0.05. All five members are retained. The three-member interaction and subset branches use the first three seeds.

`A_D1D2` appends the same24 relation features before answer-group pooling. `Q_D1D2` and `C_D1D2` append these features to their original repaired PLS8 tree designs. Tree leaves remain15/31 and rounds300. PLS targets are explicitly float32, with signed nonconstant centered targets checked before fitting. Every transform and score scale uses only full-training inputs and, where needed, training supervision.

The subset branch reconstructs all four drop-one views with the original empty-slot/zero-embedding convention. It recomputes group support and text features, keeps only correct-and-incorrect retaining views for training, and uses the existing50% view replacement rule. Its full-training augmentation seed is `model_seed + 20261026`, the declared repeat0/fold0 convention. Normalizer statistics use original training groups; augmented views never alter these statistics.

`score_branch(bank, embeddings, manifest_path, normalize=True)` uses only NumPy and standard-library inference dependencies. Each member is divided by its own fitting score scale in its original dtype, then uniformly averaged in float64. Only `drop_one_half` receives the additional ensemble scale present in its OOF experiment. Other bags have ensemble scale1. `normalize=False` returns the plain mean of raw member scores; this differs from the subset OOF field named `raw`, which already contains member-normalized scores.

Each output contains a relative-path `manifest.json`, NumPy member assets, a protocol written before fitting, source/input hashes, fit and transformation records, fitting-score evidence, and timing/size/replay checks. The NumPy adapter reads the manifest and member assets only. Fitting-score evidence is private experimental material, not a required inference artifact. Total checkpoint feasibility must be enforced after assembling a fused release; individual branch size checks do not establish that a combined checkpoint meets the assignment limit.

Example explicit training command (not executed during preparation):

```powershell
python -B optimization_core_refit.py --branch A5_noise_ema40 --output runs/core_release/A5_noise_ema40
```

No validation or test stage is implemented. Final recipe choice, release assembly and any prospective validation policy are managed separately. These branches remain conditioned on the prior training-side development search.
