# Execution roadmap

Detailed tasks and acceptance criteria live in the [execution plan](docs/execution-plan.md). The machine-readable stage graph is in [experiment_registry.json](configs/experiment_registry.json).

| Stage | Deliverable | Current status |
|---|---|---|
| Repository | Installable reference, documentation, examples and CI definition | Prepared; local verification is recorded in `reports/` |
| E0 | Current-data exports, repaired production evaluator and fixed donor folds | Integration pending |
| E1 | Matched ESM-C, direct-B and TE/context baselines | Current-data runs pending |
| F1 | A_pool natural pretraining and reusable checkpoint export | Reference helpers available; full runner pending |
| F2 | Pretrained residue extraction and controlled span-objective comparison | Encoder component available; extraction/corruption integration pending |
| F3 | C context warm-up and joint adaptation | Reference component available; controlled experiments pending |
| T1 | Bode-1 learning curves and locked Bode-2 evaluation | Pending |
| T2 | Quantitative T-domain transfer on a declared split | Assay/background integration pending |
| P1 | Optional donor-source context adapter | Conditional, not implemented |
| I1 | Additive interpretation and regenerated candidate-edit tables | Head available; biological validation and regeneration pending |
| F4 | D refinement under matched compute | Conditional; no automatic promotion |
| R1 | Selected foundation/task checkpoints, reproducible predictions and claims | Pending measured results |

The next production task is E0. A smaller supervised baseline remains useful evidence within the foundation-model programme. Optional branches do not delay an otherwise supported A/C release.
