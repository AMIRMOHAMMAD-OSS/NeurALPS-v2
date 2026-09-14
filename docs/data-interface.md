# Canonical data interface, phase2.2

This is the normalized exchange format for the reference, not an assertion about the current production filenames. Export current audited occurrence tables into it. `manifest_adapter.py` validates topology and pooled features; the exporter owns biological provenance, vocabulary, cap policy and split correctness.

## Record schema

Each JSONL record requires `id` and ordered `tokens` starting/ending in a domain, alternating D/connection/D. Keep labels, engineered-donor memberships, invariant recipient/scaffold and source context outside neural Batch fields. Optional binary `label` is validated but is not included by collation.

Token fields:

| Field | Meaning |
|---|---|
| `kind` | `domain`, `covalent` or `break`; B/J is display annotation only |
| `chain` | Nonnegative physical-chain equality ID; break is -1 and joins different chains |
| `domain_type` | Canonical integer vocabulary ID; non-D uses 0 |
| `slots` | Exactly three dictionaries; domain/covalent uses 0, break uses ordered 1/2 |
| `meta` | Six finite values in the design's fixed order; omitted values default to zeros |

Unused slots have `state: na`. Applicable states are `observed`, `empty`, `missing`. Masked state is created by preparation, not accepted as an observed clean source record. An observed slot requires:

- `hash`: lowercase SHA-256 of the owned sequence, used for exact-copy masking.
- `feature_key`: nonempty extraction identity used for vector lookup. This must distinguish different PLM contexts/recipes even when `hash` is identical.
- `owned`: nonempty list of `[translated_protein_id, start, stop]` spans, zero-based half-open.
- `dependencies`: all physical sequence intervals seen by the PLM input producing that vector, covering the ownership spans.

Optional record `aliases` contains objects with `a` and `b` spans for exact corresponding residue copies. The exporter must supply known aliases, including source contexts if used by an objective; the adapter cannot infer copy identity without sequence data.

The metadata order is log1p intended sequence length, retained/intended fraction, empty-core indicator, missing-required-slot indicator, coordinate uncertainty, route confidence. Defaults of zero do not turn missing confidence into observed certainty. Any metadata dependency on a target must be represented by its token's slot provenance or explicitly zeroed by the exporter. The masked adapter zeros all metadata at a token with a removed sequence slot. The primary sensitivity control zeros every metadata channel.

## Cache format

The JSON index maps `feature_key` to `{"file":"shard.npy","row":0}`. Each shard is float32 [rows,1152], loaded read-only with memory mapping. Paths are relative to the index directory and cannot escape it. Record sequence/content hashes and extraction metadata in the surrounding manifest; sequence SHA alone is not a general contextual feature key. Shard file hashes are a separate storage-integrity identity.

This version deliberately requires `feature_key`; old canonical records need an explicit exporter update. Do not auto-fill it from `hash` unless the audited recipe proves that identical owned sequences necessarily had identical extraction inputs and pooling coordinates.

## Preparation and collation

`prepare_record(record, store, primary=[(token_index,slot_index)])` returns NumPy graph fields, clean `teacher`, boolean `primary_target` and forbidden residue spans. It masks every contaminated observed channel, including exact aliases. Only primary targets contribute supervised reconstruction loss. Collateral removals do not add teacher targets.

`collate(prepared, device)` pads graphs and returns the reference Batch, teacher and primary mask. Supply training-only mean tensors separately, shape [B,T,3,1152]. Raw-cosine recipes may use zeros. Attach ID-aligned activity labels outside the graph when running activity training. Never include labels, donor IDs or dataset identity as input embeddings.

Trainable residue features are built online from the current corrupted view and scattered to [B,T,3,256]. Use the `raw_features` factory interface described in the execution plan; the name is historical and also accepts pooled pretrained-residue states. The adapter does not implement the raw/residue extractor or prove that a feature was recomputed correctly.

## Extended task schemas to implement

`donor_membership.jsonl` has `id`, `engineered_donors` and a separate scaffold/background identity. Set-valued purging uses engineered donors, with controls explicitly included/excluded by cohort. The scalar TE baseline additionally needs source-pathway TE sequence identity, recipient TE identity, score definition, coverage and score availability. Source IDs remain join keys, not learned categorical features.

The T-domain task manifest needs sequence, actual assay-background graph, parent/generation grouping, numeric production, units/WT reference, missing/not-tested state and declared split. Reconstruct the author's split only from actual sequence IDs. Unrolled residue baselines require a valid aligned fixed-length cohort; variable-length rows cannot be silently flattened together.

Candidate edit records specify replacement sequence, coordinates, ordered partners, background, molecular stage and source-input policy. Rebuild and re-extract all affected features. A new feature key is required whenever the PLM input or pooling region changes, including contextual effects outside the edited owned sequence.

## Available check

```bash
neuralps-validate-data graphs.jsonl --index embeddings/feature_index.json
```

Without `--index`, it checks record structure only. Even with cache vectors present, this command does not prove homology separation, biological coordinate correctness or masking-view freshness. Those gates are listed in the execution plan.
