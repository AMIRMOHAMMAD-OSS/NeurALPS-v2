# Synthetic examples

These examples contain artificial protein strings and random vectors. They illustrate the software interface and are not biological predictions or cached ESM-C embeddings.

From an installed repository checkout:

```bash
python examples/synthetic_manifest.py
python examples/synthetic_manifest.py --write artifacts/synthetic
neuralps-validate-data artifacts/synthetic/graphs.jsonl --index artifacts/synthetic/feature_index.json
```

The route contains a covalent connection and a physical chain break. Masking the covalent target removes all clean features that saw its residues, including the dependent terminus on the same protein. The other protein's terminus remains observable, and the detached teacher stays clean.

With PyTorch installed:

```bash
python examples/model_smoke.py --variant A
python examples/model_smoke.py --variant C
python examples/model_smoke.py --variant D
```

These commands build untrained pooled-input models. Residue extraction and complete training runs are work items in the [execution plan](../docs/execution-plan.md).
