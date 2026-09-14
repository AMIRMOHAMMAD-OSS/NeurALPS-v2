# Contributing

NeurALPS v2 is under active research development. Start with the [architecture](docs/architecture.md), [data contract](docs/data-interface.md), and [execution plan](docs/execution-plan.md).

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python scripts/check_repository.py
python scripts/check_contracts.py
```

Install a suitable PyTorch build and `.[model]` for neural changes. Run `python scripts/check_contracts.py --require-torch` before claiming those changes are runtime-validated.

## Changes and review

Use a focused branch and explain the problem, the resulting behavior, and the evidence that verifies it. For experiment changes, identify the stage, fixed comparison, split identity and expected decision criterion. Keep scientific claims separate from software checks.

Preserve these invariants:

- Physical domains and connections define the graph; module annotations do not affect neural computation.
- Chain breaks use two independent termini.
- Hidden targets cannot re-enter through contextual caches, residue states, metadata or reused sequence aliases.
- Donor membership and labels remain outside neural inputs.
- Candidate validity, orientation and actual-candidate identity are explicit.
- Representation features, natural-support scores and supervised activity predictions retain separate meanings.

Add a targeted regression check when changing one of these behaviors. Documentation-only edits need link and example review; they do not require a new mirrored test.

## Data and results

Keep large data, model weights and per-run outputs outside Git. Add a sanitized manifest or a small synthetic fixture when a reproduction requires input structure. Report skipped tests, missing data and incomplete evaluations explicitly. Do not treat a new best seed as a model-level performance claim.

## Writing and style

Use Python 3.11-compatible syntax, four-space indentation, explicit tensor shapes at interfaces, and clear exception messages. Prefer small functions and preserve the readable reference implementation until a profiled production replacement passes the same contracts.

Use GitHub issues for reproducible bugs or concrete research proposals. Discuss the technical evidence respectfully and avoid posting credentials, unpublished data or account-specific cluster configuration.
