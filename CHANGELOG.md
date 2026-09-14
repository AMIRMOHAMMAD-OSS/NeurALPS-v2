# Changelog

## 0.2.0.dev0 — repository scaffold, 2026-09-14

- Organize the phase2.2 reference into an installable `neuralps_v2` package with a `src/` layout.
- Add explicit command-line entry points for package preflight, canonical manifest validation and the published TE-threshold calculation.
- Preserve the A/C/D reference, masking closure, current-forward losses, complete inheritance and separate score semantics.
- Add synthetic examples, CPU CI, issue forms and contribution guidance.
- Adapt the full architecture and execution plan to portable repository paths.
- Preserve the prior evidence and verification records separately from repository validation.

This is a development snapshot. It does not add a trained checkpoint, new biological benchmark result, or complete production training runner.

## Design lineage

The research specification is **phase2.2**. It introduces pretrained-residue comparisons, a quantitative T-domain task, explicit TE/source-context baselines and conditional D refinement. The software version and research-design version identify different things; record both in experiments.
