# Local data

Biological records, embedding shards and model weights are supplied separately. The repository contains a schema, a path template, and synthetic examples; it does not redistribute the underlying manuscripts, assay spreadsheets or trained PLM weights.

Copy `configs/data_paths.template.json` to `configs/data_paths.local.json` from the repository root. Fill all required paths using the current audited data. Relative paths resolve from the configuration file's directory, so a manifest under this directory would use `../data/natural_train.jsonl`.

Keep these artifacts distinct:

| Artifact | Purpose |
|---|---|
| Graph JSONL | Domain/connection occurrences, ordered physical chains and provenance |
| Feature index and `.npy` shards | Context-aware feature keys to frozen vectors |
| Labels | Assay outcomes joined once by construct ID |
| Donor memberships | Engineered donor sets, separate from the shared scaffold |
| Split manifest | Fixed natural and engineered development/evaluation membership |
| Extraction manifest | PLM weight/recipe identity, parent context, pooling coordinates and hashes |

For each future public dataset release, provide its source, version, permitted distribution terms and a reproducible acquisition/export procedure. Audited cache counts alone are not a download endpoint.

See the [canonical data interface](../docs/data-interface.md) and [reproducibility guide](../docs/reproducibility.md).
