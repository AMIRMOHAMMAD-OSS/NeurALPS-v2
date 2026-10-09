# NeurALPS: frozen compatibility and supervised activity

This release includes a fitted supervised head trained on all **494 reviewed
engineered assemblies**: 324 BODE1, 105 BODE2 and 65 T-domain. It also ports the
original SSL26 encoder and its local/full-context reconstruction scoring to a
small Python package and a Colab notebook.

The notebook is `notebooks/NeurALPS.ipynb`. It has a saved-feature demonstration
that works now, followed by new-sequence inference using the small runtime
asset bundle exported from Jean Zay. New-sequence inference has not yet been
run in Colab with the actual model assets. The repository URL is not yet set.

## What is included

| Item | Location |
| --- | --- |
| Fitted all-494 supervised model | `models/all494/supervised_head.npz` |
| Exact training row identities and labels | `models/all494/training_rows.json` |
| Reproducible frozen training features | `data/training_features.npz` |
| Regularization selection evidence | `data/head_selection.json` |
| Colab notebook | `notebooks/NeurALPS.ipynb` |
| CPU training command | `scripts/fit_all_labels.py` |
| Minimal runtime exporter | `scripts/export_runtime.py` |
| Original-versus-port numerical check | `scripts/verify_runtime.py` |
| Source parity and validation results | `docs/` |
| Real annotated input example | `examples/AI_1_annotated.json` |

## Finish the runtime transfer

Follow `RUN_ON_JEAN_ZAY.txt`. The exporter reads the SHA-pinned checkpoint and
training-only centering means. It writes portable NPZ weights, historical
embedding references and three golden fixtures. It compares the port against
the original implementation on BODE1, BODE2 and T-domain examples before
creating the final ZIP.

The exported bundle also preserves the installed Transformers 4.57.6 fork,
including its metadata and licenses. The recorded Git source URL returned 404
during porting; the actual installed implementation is therefore exported.
The exporter must run in the existing `envs/esmc_connections_hf` environment.
It performs CPU inference for the check and does not run training or ESMC.

The 2.3 GB ESMC weights are downloaded in Colab from the pinned Hugging Face
revision and verified by SHA-256. Existing local copies can also be supplied.
The large natural-data cache is not required for default inference.

## Run locally

```bash
python -m pip install -e '.[inference]'
python scripts/verify_runtime.py --assets /path/to/runtime_assets
```

Install the preserved Transformers wheel and its declared dependencies as shown
in the notebook. New-sequence extraction uses CUDA BF16 with the original
pooling, tokenizer and historical-reference check. CPU-only use supports the
saved-feature demonstration, final-head fitting and cached-input encoder tests.

```bash
neuralps examples/AI_1_annotated.json \
  --assets /path/to/runtime_assets \
  --esmc-model /path/to/ESMC-600M \
  --head models/all494 \
  --output results/AI_1.json
```

The JSON input supplies proteins in biosynthetic order with mature physical
sequences and already resolved domain annotations. Coordinates are **0-based,
end-exclusive**, separately for each protein. The package reconstructs covalent
windows and physical protein breaks from those inputs. It does not infer
domain annotations or chain order from an unannotated FASTA. Ambiguous domain
types such as a generic `Condensation` are rejected rather than assigned an
invented subtype. No silent sequence truncation is permitted; each independently
embedded physical feature must fit the original 2046-residue limit.

## The pretrained output

`pretrained_map` queries each physical domain or boundary independently while
hiding all overlapping and identical sequence features. For a protein break,
the two physical termini are masked together and their losses are averaged
once. Both local-only and full-context results use the same frozen weights.

`joint_domain_indices` selects a contiguous exchanged region. The package masks
that region and all touching boundaries together, including their physical
masking closure, and reports domain mean, boundary mean, balanced mean and
neighborhood mean. Those regional results are distinct from the single-object
map. The local balanced joint-region score is the deployment default for an
identified exchange; the notebook retains full context for comparison.

Scores reproduce the original `1 - mean(slot_loss)` implementation:
25% raw cosine agreement plus 75% agreement after subtracting the training-only
type mean, with the original degenerate-target fallback. Raw values are in
[-1, 1]. They are not activity probabilities. Queries with missing physical
features or fewer than two remaining independent contextual objects retain an
explicit unavailable status; scores are not fabricated for them.

No natural-reference percentiles are supplied for an unsupported score/mask
contract. The release does not conflate single-object and joint-region results.

## The supervised output

The head uses 2822 full-context typed features, training-only standardization
and the original deterministic L-BFGS logistic solver. The encoder is frozen.
L2 = **10** minimizes mean archived inner-validation BCE on each BODE donor
benchmark separately and on their equal-weight average. T-domain outcomes did
not select the penalty; all 65 are included in the final fit. Each labelled
assembly has equal weight in that fit, matching the original solver objective.

The native BODE2 control is recorded separately and is not among the 494
engineered training cases. Unreviewed T-domain hybrids are not relabelled.
All six BODE1 cases with incomplete joint-mask scores have complete typed
features and are included. No joint-mask scores are needed by this head.

`activity_score` is the sigmoid of the assembly logit. It is uncalibrated. The
optional per-object output is an exact additive decomposition of that logit,
with count, absent-type normalization and intercept effects retained in an
assembly context term. These are signed classifier contributions, not causal
defect scores or individual domain probabilities.

To reproduce the final fit:

```bash
python scripts/fit_all_labels.py --output results/refit_all494
```

The trainer refuses missing, duplicated, relabelled or mismatched training rows.
`training_predictions.json` is explicitly in sample. This combined refit has
no independently measured held-out AUROC. Prior dataset-specific evaluations
are preserved separately in `docs/ARCHIVED_SUPERVISED_RESULTS.json`.

## Validation status

The portable typed pooling exactly matches 88 saved T-domain feature rows.
The portable head exactly replays 1077 held-out predictions from 56 archived
donor-fold models. Local tests check overlap and alias masking, two-slot
protein breaks, missing-input handling, the loss formula, head optimization,
training membership, NPZ weight loading and score decomposition.

The actual SSL checkpoint is not in the received source handoff. Its numerical
parity check is run by the Jean Zay exporter and repeated by the notebook.
ESMC GPU extraction must pass its historical reference check in the selected
Colab runtime. A BF16-capable GPU is required for the exact extraction mode;
a T4 does not satisfy that hardware requirement.

## Repository contents and provenance

Commit the source, notebook, modest training-feature NPZ and fitted head.
Keep runtime bundles and ESMC weights in versioned downloadable assets. The
`.gitignore` excludes them and generated results. No repository has been pushed
from this workspace. Once the target URL is supplied, the notebook's GitHub
source option can be set to an actual commit.

Three original modules and the domain vocabulary are preserved byte for byte;
their hashes are in `docs/ORIGINAL_SOURCES.json`. The logistic solver changes
only its import path. Existing project and upstream rights are retained; no
new license grant is asserted. The supplied data and results remain research
development evidence, with the original studies' endpoint definitions.
