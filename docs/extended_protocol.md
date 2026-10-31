# Additional validation-only research

The compact pairwise/text baseline was frozen and independently scored before the extended optimization study. Its checkpoint, inference source, score and freeze manifest are retained as a scored baseline. The complete project therefore includes prior aggregate test exposure; its test bank is not an untouched holdout for the extended study.

All additional method development uses the supplied training and validation splits only. Test inputs, reference answers, per-question outcomes and scores are excluded from new experiment scripts and model selection. Additional methods are judged by training-only holdout or cross-validation evidence and the previously defined validation disagreement metric. Existing test results do not choose architecture, features, seeds, fusion weights or checkpoints.

## Hypotheses and finite comparisons

1. **Training objective and context:** compare pairwise and multi-positive listwise losses, with raw versus question-centered embedding information, using training-question early stopping.
2. **Direct group ranking:** compare compact boosted ranking trees with candidate classification, using structured text features and train-fitted PCA representations.
3. **Additional text information:** test train-fitted word/character TF-IDF pairwise scorers for complementary lexical evidence.
4. **Learned combination:** generate five-fold out-of-fold training predictions from base models and fit a small nonnegative fusion model using training labels only. Model-selection folds remain grouped by question.
5. **Joint features and auxiliary supervision:** learn embedding/text interactions in one network; separately compare an auxiliary classification loss that can use easy training groups.
6. **Capacity and stability:** select a larger hidden layer inside training before one validation evaluation, and assess fixed five-seed averages without choosing a favorable new seed.

The resource constraints remain unchanged: no external training data or pretrained weights, CPU-only offline inference, at most four evaluation threads, at most 16 MiB uncompressed inference assets and at most 30 seconds for the fresh prediction and scoring processes on the reference machine.

The current scored checkpoint remains the deployable version until another method has been selected on permissible evidence, inspected and explicitly frozen. A new method must not replace it merely because of a favorable test comparison. Additional final scoring, if performed, must be documented as subsequent scoring rather than claiming no prior test exposure.
