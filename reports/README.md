# Verification and evidence records

`repository_verification.json` records checks run after reorganizing the source into this repository. Its values describe the authoring environment and do not substitute for the first hosted CI result.

`reference/` preserves the phase2.2 reference package's evidence ledger and verification:

- `contract_results.json`: 33 discovered checks, 22 passes and 11 PyTorch skips in the original authoring environment.
- `verification.json`: original code, PDF and integration status; paths and page counts refer to that earlier report package.
- `evidence_summary.json`: reported manuscript counts, quantitative benchmark values and workbook inventory.
- `source_inventory.json`: names and SHA-256 hashes of supplied source files. Source bytes are not redistributed here.

Temporary build/test outputs belong in ignored `artifacts/` or `runs/` directories. Add future benchmark summaries here only when their method, split, input track and underlying prediction artifact are available.
