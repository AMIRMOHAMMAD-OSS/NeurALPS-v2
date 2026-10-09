#!/usr/bin/env python3
"""Create the small runtime asset ZIP on Jean Zay, without copying model caches.

Loads only the SHA-pinned original checkpoint, converts weights to pickle-free
NPZ, and records independent original-code golden predictions for port checks.
Does not train or modify the existing project. Requires NumPy and CPU PyTorch.
"""
import argparse
import ast
import base64
import csv
import gzip
import hashlib
import importlib.metadata
import importlib.util
import io
import json
from pathlib import Path
import sqlite3
import sys
import zipfile
import numpy as np

RELEASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE))
from neuralps.contracts import SSL_SHA, TARGETS_SHA, FORK_COMMIT, require, sha, dump

def export_transformers_wheel(out):
    """Preserve the installed pure-Python fork, metadata and license files."""
    dist = importlib.metadata.distribution("transformers")
    require(dist.version == "4.57.6", "Use the original esmc_connections_hf Python environment")
    origin = json.loads(dist.read_text("direct_url.json") or "{}")
    require(origin.get("vcs_info", {}).get("commit_id") == FORK_COMMIT, "Installed Transformers fork differs")
    sources = {}
    for entry in dist.files or []:
        rel = Path(str(entry))
        if rel.is_absolute() or ".." in rel.parts or "__pycache__" in rel.parts or rel.suffix == ".pyc":
            continue
        if not (rel.parts[0] == "transformers" or rel.parts[0].startswith("transformers-") and rel.parts[0].endswith(".dist-info")):
            continue
        if rel.name in {"RECORD", "INSTALLER", "REQUESTED", "WHEEL"}:
            continue
        p = Path(dist.locate_file(entry))
        require(p.is_file(), "Installed fork file missing: "+str(entry))
        sources[rel.as_posix()] = p.read_bytes()
    require("transformers/models/esmc/modeling_esmc.py" in sources, "Installed fork lacks ESMC source")
    info = "transformers-4.57.6.dist-info"
    require(info+"/METADATA" in sources, "Installed distribution metadata absent")
    sources[info+"/WHEEL"] = b"Wheel-Version: 1.0\nGenerator: NeurALPS installed-runtime export\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    record = io.StringIO(newline="")
    writer = csv.writer(record, lineterminator="\n")
    for name, data in sorted(sources.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        writer.writerow([name, "sha256="+digest, len(data)])
    writer.writerow([info+"/RECORD", "", ""])
    sources[info+"/RECORD"] = record.getvalue().encode()
    wheel = out/"transformers-4.57.6-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in sorted(sources.items()):
            z.writestr(name, data)
    return dict(version=dist.version, original_vcs_commit=FORK_COMMIT, wheel=wheel.name,
                source_sha256={name.removeprefix("transformers/"): hashlib.sha256(data).hexdigest()
                               for name, data in sources.items() if name.startswith("transformers/") and name.endswith(".py")},
                origin=origin)

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj

def functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    require(len(nodes) == len(names) and not any(n.decorator_list for n in nodes), "Original function contract changed")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace

def readonly(path):
    return sqlite3.connect(Path(path).resolve().as_uri()+"?mode=ro", uri=True)

def read_parents(path):
    parents, parts, current = {}, [], None
    with gzip.open(path, "rt") as stream:
        for line in stream:
            line = line.strip()
            if line.startswith(">"):
                if current is not None:
                    value = "".join(parts)
                    from neuralps.contracts import seqsha
                    parents[seqsha(value)] = value
                current, parts = line[1:], []
            elif line:
                parts.append(line)
    if current is not None:
        from neuralps.contracts import seqsha
        value = "".join(parts)
        parents[seqsha(value)] = value
    return parents

def export(root, out):
    import torch
    from collections import defaultdict
    root, out = root.resolve(), out.resolve()
    require(not out.exists() and not out.with_suffix(".zip").exists(), "Use a new output directory")
    checkpoint = root/"e1_ssl_continue30_v1_530757/best.pt"
    require(sha(checkpoint) == SSL_SHA, "Selected SSL checkpoint changed")
    target_dir = root/"e1_ssl_targets_497189"
    require(sha(target_dir/"targets.npz") == TARGETS_SHA, "Original target mean bank changed")
    expected_source = json.loads((RELEASE/"docs/ORIGINAL_SOURCES.json").read_text())
    for item in expected_source:
        require(sha(root/item["original"]) == item["sha256"], "Original source changed: "+item["original"])
    golden_sources = json.loads((RELEASE/"docs/GOLDEN_SOURCE_HASHES.json").read_text())
    for relative, digest in golden_sources.items():
        require(sha(root/relative) == digest, "Golden reference source changed: "+relative)
    # Legacy optimizer checkpoints contain NumPy/Python state. Deserialization
    # is allowed only after matching the fixed known original SHA above.
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    require(saved["step"] == 26000, "Selected step differs")
    config = saved["config"]["settings"]["encoder"]
    manifest = json.loads((target_dir/"manifest.json").read_text())
    require(saved["targets_manifest"] == manifest, "Checkpoint and target manifests disagree")
    with np.load(target_dir/"targets.npz", allow_pickle=False) as z:
        means = {name: z[name].copy() for name, _ in manifest["means"].values()}
    for group, (name, _) in manifest["means"].items():
        require(np.array_equal(saved["target_means"][group].numpy(), means[name]), "Checkpoint mean differs: "+group)
    out.mkdir(parents=True)
    np.savez_compressed(out/"encoder_weights.npz", **{k: v.detach().cpu().numpy() for k, v in saved["model"].items()})
    np.savez_compressed(out/"target_means.npz", **means)
    dump(out/"target_mean_map.json", manifest["means"])
    dump(out/"encoder_config.json", config)
    original = module("neuralps_original_encoder", root/"NeurALPS_Assembly_v1_20261002/src/assembly_model.py")
    mask = module("neuralps_original_mask", root/"NeurALPS_Assembly_v1_20261002/src/masking_contract.py")
    namespace = functions(root/"NeurALPS_ContextAudit_20261007/frozen_scoring.py", {"predict_mode"}, {"require": require})
    plan_ns = functions(root/"NeurALPS_ContextAudit_20261007/context_plans.py",
                        {"slots_by_object", "remaining_context", "make_plan"}, {"require": require, "defaultdict": defaultdict})
    loss_ns = functions(root/"NeurALPS_Assembly_v1_20261002/run_ssl.py", {"slot_loss"}, {})
    torch.set_num_threads(1)
    model = original.AssemblyEncoder(original.Config(**config))
    model.load_state_dict(saved["model"], strict=True)
    model.eval().requires_grad_(False)
    del saved
    db = readonly(root/"e0_training_prep_461027/training_data.sqlite3")
    idx = readonly(root/"e0_cache_final_491894/feature_index.sqlite3")
    base = np.memmap(root/"e0_cache_final_491894/features.f32", mode="r", dtype="<f4", shape=(548713, 1152))
    troot = root/"e10_tdomain_embed_696333"
    tidx = readonly(troot/"feature_index.sqlite3")
    tz = np.memmap(troot/"features.f32", mode="r", dtype="<f4", shape=(290, 1152))
    tparents = read_parents(troot/"inputs/parents.fasta.gz")
    t_records = json.loads(gzip.open(root/"e11_tdomain_transfer_699466/scores/records.json.gz", "rt").read())
    labels = json.loads((RELEASE/"data/expected_labels.json").read_text())["rows"]
    fixtures = []
    for dataset in ("bode1", "bode2", "tdomain"):
        aid = next(r["assembly_id"] for r in labels if r["dataset"] == dataset)
        if dataset == "tdomain":
            record = t_records[aid]
            parents = {h: tparents[h] for h in set(record["parent_sequence_hashes"].values())}
        else:
            row = db.execute("SELECT payload FROM assemblies WHERE dataset=? AND assembly_id=?", (dataset, aid)).fetchone()
            require(row is not None, "Golden assembly absent: "+aid)
            record = json.loads(row[0])
            parents = {h: db.execute("SELECT sequence FROM proteins WHERE hash=?", (h,)).fetchone()[0]
                       for h in set(record["parent_sequence_hashes"].values())}
        vectors = {}
        for slot in record["model_slots"]:
            if slot["state"] != "OBSERVED":
                continue
            h = slot["sequence_hash"]
            table, values = (tidx, tz) if dataset == "tdomain" else (idx, base)
            found = table.execute("SELECT feature_row FROM features WHERE hash=? AND status='covered'", (h,)).fetchone()
            require(found is not None, "Golden feature absent")
            vectors[h] = np.array(values[found[0]], copy=True)
        def mean(slot):
            item = manifest["means"].get(slot["group"])
            if item is None or item[1] < 100:
                broad = ("D" if slot["kind"] == 0 else "C") if slot["kind"] != 2 else slot["group"]
                item = manifest["means"][broad]
            return means[item[0]]
        clean = original.collate_records([record], vectors.__getitem__)
        with torch.inference_mode():
            h = model.features(clean)[0].numpy()
        by = plan_ns["slots_by_object"](record)
        targets = {"single": [2], "joint": [1, 2, 3]}
        expected = {}
        for name, indices in targets.items():
            plan = plan_ns["make_plan"](record, indices, parents.__getitem__, mask.masking_closure)
            objects = {}
            if plan["status"] == "READY":
                batch = original.collate_records([record], vectors.__getitem__)
                for j, role in plan["hidden"]:
                    batch["state"][0, j, role] = 4
                    batch["x"][0, j, role] = float("nan")
                with torch.inference_mode():
                    for mode in ("local", "full"):
                        p = namespace["predict_mode"](model, batch, local=(mode == "local"))
                        for j in indices:
                            slots = by[j]
                            target = torch.as_tensor(np.stack([vectors[s["sequence_hash"]] for s in slots]))
                            mu = torch.as_tensor(np.stack([mean(s) for s in slots]))
                            loss, _ = loss_ns["slot_loss"](p[0, j, [s["slot"] for s in slots]], target, mu)
                            objects.setdefault(str(j), {})[mode] = float((1-loss.mean()).item())
            expected[name] = dict(plan=plan, objects=objects)
        prefix = "golden_"+dataset
        np.savez_compressed(out/(prefix+".npz"), object_states=h, **vectors)
        # Strip assay fields from the inference fixture.
        record.pop("label", None)
        dump(out/(prefix+".json"), dict(record=record, parents=parents, expected=expected))
        fixtures.append(prefix)
    db.close(); idx.close(); tidx.close()
    backend = module("neuralps_original_embedding", root/"NeurALPS_TDomain_Embed_20261006/embedding_backend.py")
    refs = json.loads((root/"data/esmc_bode2_domains_v2/manifest.json").read_text())["references"]
    cached, provenance = backend._reference_vectors(root, refs, torch)
    np.savez_compressed(out/"esmc_reference_vectors.npz", **{k: v.numpy() for k, v in cached.items()})
    dump(out/"esmc_references.json", refs)
    dump(out/"esmc_reference_provenance.json", provenance)
    transformer_distribution = export_transformers_wheel(out)
    payload = dict(schema="neuralps_portable_runtime_v1", source_checkpoint_sha256=SSL_SHA,
                   selected_step=26000, original_targets_sha256=TARGETS_SHA, golden_fixtures=fixtures,
                   files_sha256={p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()},
                   original_code_golden_predictions=True, training_performed=False,
                   esmc_weights_included=False, transformers_distribution=transformer_distribution,
                   original_golden_sources_sha256=golden_sources)
    dump(out/"manifest.json", payload)
    verifier = module("neuralps_port_verifier", RELEASE/"scripts/verify_runtime.py")
    report = verifier.verify(out)
    dump(out/"PORT_PARITY.json", report)
    payload["files_sha256"]["PORT_PARITY.json"] = sha(out/"PORT_PARITY.json")
    dump(out/"manifest.json", payload)
    archive = out.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            z.write(p, p.name)
    print("Runtime assets:", archive)
    print("SHA256:", sha(archive))
    print("Bytes:", archive.stat().st_size)
    return archive

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    export(args.root, args.output_dir)
