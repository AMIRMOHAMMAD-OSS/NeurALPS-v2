# Reproducibility

Record the software version (`0.2.0.dev0`), research-design version (`phase2.2`) and exact Git commit separately. A configuration hash cannot substitute for a source commit, and neither identifies the data by itself.

## Environment and checks

From the repository root:

```bash
python scripts/check_repository.py
python scripts/check_contracts.py --json artifacts/contracts.json
python scripts/check_contracts.py --require-torch --json artifacts/contracts-torch.json
```

The strict command returns nonzero if required runtime checks are skipped. Preserve the output from the environment where training will actually run. The `reports/reference/` directory records the earlier flat reference package; `reports/repository_verification.json` records the repository reorganization. Neither is a hosted CI result.

Save installed package versions, Python, platform, accelerator, CUDA/ROCm and relevant precision settings with each production run. The CPU CI recipe pins PyTorch 2.8.0 as an initial reference. An experiment should save its fully resolved environment rather than assume future dependency resolution is identical.

## Data and extraction identity

Retain dataset release/hash, manifest and shard hashes, natural split identity, engineered donor memberships, label joins and cohort maps. Record frozen ESM-C weight identity, tokenizer/extraction recipe, pooling coordinates, cap policy and feature context.

Use separate identities for owned sequence, contextual feature extraction, file integrity and occurrence position. Known exact-copy aliases belong in the masking provenance. Every derived training statistic and candidate panel must come from the permitted training data.

## Checkpoints

The reference helper stores model, configuration fingerprint, stage, epoch/update, optional optimizer/scheduler state and Python/NumPy/Torch RNG state. Complete production resumption also needs sampler state, gradient-accumulation position and distributed execution state; these remain runner tasks.

Natural checkpoints and supervised task copies must be distinguishable. A checkpoint selected with activity labels is no longer a purely natural-data selection. Keep frozen natural diagnostics on their original checkpoint.

## Experiment comparison

Save all registered seeds and retained folds, label-budget identities, head settings, input tracks, optimizer updates, wall time and peak memory. Report selection rules before scoring the external set. Use paired comparisons on the same samples and distinguish seed SD from data uncertainty.

The [execution plan](execution-plan.md) defines the outputs needed at each stage. Actual predictions and checkpoint artifacts should accompany a future result release; a design document alone is not a reproduction package for a trained model.
