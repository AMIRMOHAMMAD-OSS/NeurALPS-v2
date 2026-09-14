# Evaluation protocol

The purpose of evaluation is to determine whether natural NRPS pretraining adds transferable information beyond generic frozen ESM-C features and simple biological baselines.

## Bode construct activity

Use Bode-1 for supervised development. Freeze donor-held-out outer folds before comparing models. Donor membership is set-valued: remove every training construct containing a held-out engineered donor. Keep the shared recipient scaffold separate from those donor sets.

Use inner donor splits for hyperparameters, checkpoint selection and stopping. Build nested 10%, 25%, 50% and 100% label budgets; labels used for validation count against the budget. Pair the splits, subsets and five seeds across comparisons.

Report macro donor-fold AUROC and fold-level results, average precision, calibration metrics and selection precision at a predefined budget. Flag one-class or otherwise unevaluable folds. Report seed variability separately from uncertainty due to the limited correlated data.

Bode-2 is a retrospective external benchmark. Fix the model, stopping rule and threshold using Bode-1, then export external predictions under the locked protocol. Existing familiarity with Bode-2 means it should not be described as a newly untouched prospective test.

## Required controls

| Control | Question |
|---|---|
| Frozen domain ESM-C plus a nonlinear head | What does the generic PLM already provide? |
| Frozen domains plus connection features | What do direct B/J inputs contribute? |
| Same architecture without NRPS SSL | What does pretraining add beyond architecture? |
| A with additional updates matched to C | Is the gain due to context or additional training? |
| Pooled and residue features with compatible heads | Does residue detail help under a controlled comparison? |
| TE similarity and simple position/class covariates | What does biological relatedness explain? |
| ESM-C with the same source-pathway inputs | Does an enriched model gain from its encoder or additional information? |

## Quantitative T-domain transfer

Create an independent task copy from the natural checkpoint. Preserve actual sequence lengths, assay/background identity and parent/generation metadata. Do not replace missing or not-tested production with zero.

The supplied study reports a 55/30 train/test split for 85 sequences. Recover those IDs before claiming a reproduction. Otherwise declare a new split, report nearest train/test distances and use appropriate parent/group controls. The full worksheet inventory is not the published benchmark split.

Use Spearman correlation as the primary quantitative metric, with a declared training transform such as `log1p(percent WT)`. Report uncertainty, ties, nondetection handling and a frozen-ESM baseline. This is a separate assay task, not a conversion of all T-domain observations into Bode binary labels.

## Prediction records

Every prediction should retain construct/sequence ID, task, cohort, outer fold, held-out donor group, seed, label budget, input track, model/checkpoint identity, label and score. Keep scores unrounded in saved records.

See the [full specification](architecture.md) for published evidence and the [execution plan](execution-plan.md) for the production evaluator tasks.
