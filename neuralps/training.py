"""Refit the selected deployment head on the exact reviewed 494-row cohort."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from .contracts import require, dump, sha, SSL_SHA, FEATURE_SCHEMA
from .head import fit, predict, save_model

def validate_training(rows, matrix, expected):
    keys = [(r["dataset"], r["assembly_id"]) for r in rows]
    wanted = {(r["dataset"], r["assembly_id"]): r["observed_activity"] for r in expected["rows"]}
    require(len(keys) == len(set(keys)), "Duplicate training identities")
    require(set(keys) == set(wanted), "Training rows do not exactly cover the reviewed cohort")
    require(len(keys) == expected["expected_total"] == 494, "Expected all 494 reviewed engineered rows")
    for r, key in zip(rows, keys):
        require(type(r["observed_activity"]) is int and r["observed_activity"] == wanted[key], "Training label mismatch")
    x = np.asarray(matrix, dtype=np.float64)
    require(x.shape == (len(rows), 2822) and np.isfinite(x).all(), "Invalid or incomplete training features")
    return x, np.array([r["observed_activity"] for r in rows], dtype=np.float64)

def train(root, output):
    root, output = Path(root), Path(output)
    require(not output.exists(), "Use a new training output directory")
    contract = json.loads((root/"data/training_manifest.json").read_text())
    expected = json.loads((root/"data/expected_labels.json").read_text())
    rows = json.loads((root/"data/training_rows.json").read_text())
    for name, digest in contract["files_sha256"].items():
        require(sha(root/name) == digest, "Training input changed: "+name)
    require(contract["checkpoint_sha256"] == SSL_SHA and contract["feature_schema"] == FEATURE_SCHEMA,
            "Training feature provenance differs")
    with np.load(root/"data/training_features.npz", allow_pickle=False) as z:
        require(set(z.files) == {"features"}, "Unexpected training archive schema")
        x, y = validate_training(rows, z["features"], expected)
    require(hashlib.sha256(x.astype("<f8").tobytes()).hexdigest() == contract["matrix_sha256"],
            "Training feature matrix changed")
    selection = json.loads((root/"data/head_selection.json").read_text())
    model, diagnostics = fit(x, y, selection["selected_l2"])
    output.mkdir(parents=True)
    save_model(output/"supervised_head.npz", model)
    dump(output/"training_rows.json", rows)
    logits = predict(model, x)
    dump(output/"fit_diagnostics.json", diagnostics)
    dump(output/"training_predictions.json", [dict(r, logit=float(z), activity_score=float(p),
         evaluation="IN_SAMPLE_DEPLOYMENT_FIT") for r, z, p in zip(rows, logits, expit(logits))])
    manifest = dict(schema="neuralps_all494_typed_context_v1", feature_schema=FEATURE_SCHEMA,
                    checkpoint_sha256=SSL_SHA, trained_rows=len(rows), positives=int(y.sum()),
                    by_dataset=dict(Counter(r["dataset"] for r in rows)), l2=model["l2"],
                    selection=selection["rule"], encoder_updates=0,
                    fit_scope="All 494 reviewed engineered examples; BODE2 native control excluded",
                    score_interpretation="Uncalibrated sigmoid activity score",
                    validation="This combined refit has no independent held-out performance estimate",
                    model_sha256=sha(output/"supervised_head.npz"),
                    training_manifest_sha256=sha(root/"data/training_manifest.json"),
                    rows_sha256=sha(output/"training_rows.json"))
    dump(output/"manifest.json", manifest)
    return manifest
