# Interpreting connection compatibility

A compatibility table must state what each entry measures. NeurALPS keeps three kinds of outputs separate.

| Output | Conditioning and computation | Interpretation |
|---|---|---|
| Contextual features | Clean encoder forward before heads | Learned representation of the observed domain or connection |
| Natural-support table | Masked target prediction compared with valid candidate features | Support under the natural-data objective in the declared background |
| Supervised edit table | Activity-head difference between complete original and rebuilt edited constructs | Predicted activity change for a specified edit |

Raw contrastive logits can contain arbitrary context-dependent offsets. They do not form a calibrated activity scale. Candidate-reference differences remove a common offset within a compatible conditional comparison; they do not make unrelated contexts or tasks directly comparable.

## Additive activity factors

The optional `AdditiveActivityHead` exposes domain, connection and background terms that sum to its construct logit. It uses shared functions across positions and must be compared against the common pooled head under the same encoder and splits.

This is an exact decomposition of a model output. Correlated biological factors can still produce unstable or noncausal attributions. A connection contribution is not a measured probability that the physical seam works.

## Candidate tables

Each table should retain the original background, edited coordinates, ordered partners, candidate identity, validity, score definition and reference. For activity-edit tables, rebuild every affected sequence/window and contextual feature before rescoring. Swapping a single cached vector while keeping other contaminated features fixed does not implement that edit.

Chain-break candidates contain two ordered termini and never a through-translated sequence. Missing and invalid candidates stay explicit; they cannot enter an average as dummy scores.

## Validation

Check orientation, candidate permutation invariance, duplicate positive handling and reference-offset invariance. Confirm that additive factors sum to the reported logit. Then evaluate ranking consistency across seeds, known engineered scar positions and appropriate controls. Scar-location agreement can support interpretation, but it does not establish causal activity prediction by itself.

See [the architecture specification](architecture.md) for the score equations and [the execution plan](execution-plan.md) for the candidate-regeneration implementation stage.
