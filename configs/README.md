# Configuration

| File | Role |
|---|---|
| `phase2_final_config.json` | Phase2.2 architecture, loss and evaluation starting settings |
| `experiment_registry.json` | Stage dependencies and acceptance criteria |
| `data_paths.template.json` | Local integration paths to populate |

Copy the template to `data_paths.local.json`, which Git ignores. Resolve paths relative to that file's directory or supply absolute local paths.

The reference factory supports explicit fixed architecture dimensions and rejects unsupported settings. Editing JSON alone does not implement a new architecture. For example, a width-128 comparator requires a corresponding parameterized model change and validation.

`domain_types: 32` is the reference embedding capacity; the actual canonical domain-to-index mapping must come from the audited manifest. The configuration does not invent that vocabulary or the extraction crop policy.

Run preflight from the repository root:

```bash
neuralps-preflight --config configs/phase2_final_config.json
neuralps-preflight --config configs/phase2_final_config.json --paths configs/data_paths.local.json
```

Package-only preflight does not validate data, labels, feature-space consistency or training.
