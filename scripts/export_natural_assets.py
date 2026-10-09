#!/usr/bin/env python3
"""Export the existing reference and ALL natural candidate features, without training.

Run on Jean Zay with the existing NumPy environment. The original project is
opened read-only. Float32 embeddings are preserved exactly and distributed in
small release assets; no ESMC or encoder inference is performed.
"""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import zipfile
import numpy as np

RELEASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE))
from neuralps.contracts import require, sha, seqsha, SSL_SHA, dump
from neuralps.repertoire import signature

STAMP = "20261010"
FEATURE_SHA = "5a7c2f3afd527785ce3c12eaa3b97138034f4dfc811d840b5b44cf1534b9ef90"
EXPECTED_COUNTS = {"train": 27913, "dev": 3499, "internal_test": 3515}


def readonly(path):
    return sqlite3.connect(Path(path).resolve().as_uri()+"?mode=ro", uri=True)


def archive(path, members):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for source, name in members:
            z.write(source, name)


def export_reference(root, destination):
    proof = json.loads((RELEASE/"neuralps/data/reference_provenance.json").read_text())
    source = root/proof["source_run"]
    for name, digest in proof["files_sha256"].items():
        require(sha(source/name) == digest, "Original reference changed: " + name)
    output = destination/f"NeurALPS_natural_reference_{STAMP}.zip"
    archive(output, [(source/name, name) for name in proof["files_sha256"]])
    return output


def export_repertoire(root, destination, rows_per_shard=20000):
    require(type(rows_per_shard) is int and 1 <= rows_per_shard <= 100000, "Invalid shard size")
    source = readonly(root/"e0_training_prep_461027/training_data.sqlite3")
    counts = dict(source.execute("SELECT split,count(*) FROM assemblies WHERE dataset='natural' GROUP BY split"))
    require(counts == EXPECTED_COUNTS, "Natural corpus membership differs: " + json.dumps(counts))
    feature_file = root/"e0_cache_final_491894/features.f32"
    require(feature_file.stat().st_size == 548713*1152*4, "Feature matrix shape differs")
    print("Verifying the existing frozen feature matrix...", flush=True)
    require(sha(feature_file) == FEATURE_SHA, "Original feature matrix checksum differs")
    idx = readonly(root/"e0_cache_final_491894/feature_index.sqlite3")
    feature_rows = dict(idx.execute("SELECT hash,feature_row FROM features WHERE status='covered'"))
    idx.close()
    vectors = np.memmap(feature_file, mode="r", dtype="<f4", shape=(548713,1152))
    working = destination/"repertoire"
    working.mkdir()
    db = sqlite3.connect(working/"repertoire.sqlite3")
    db.executescript("""
        CREATE TABLE objects (assembly_id TEXT, j INTEGER, signature TEXT,
                              object_json TEXT, slots_json TEXT, PRIMARY KEY(assembly_id,j));
        CREATE TABLE features (hash TEXT PRIMARY KEY, shard INTEGER, row_index INTEGER, sequence TEXT);
        CREATE TABLE parents (hash TEXT PRIMARY KEY, sequence TEXT);
    """)
    known, parent_written = set(), set()
    shards, stream, shard_rows, object_count = [], None, 0, 0
    from neuralps.inputs import vocabulary
    names = {v:k for k,v in vocabulary().items()}
    @lru_cache(maxsize=512)
    def parent(h):
        value = source.execute("SELECT sequence FROM proteins WHERE hash=?", (h,)).fetchone()
        require(value is not None and seqsha(value[0]) == h, "Natural parent sequence mismatch")
        return value[0]
    def finish_shard():
        nonlocal stream, shard_rows
        if stream is None:
            return
        stream.close()
        path = working/f"vectors-{len(shards):03d}.f32"
        packed = destination/f"NeurALPS_vectors_{len(shards):03d}_{STAMP}.zip"
        archive(packed, [(path, path.name)])
        shards.append(dict(path=path.name, rows=shard_rows, sha256=sha(path),
                           archive=packed.name, archive_sha256=sha(packed), archive_bytes=packed.stat().st_size))
        stream, shard_rows = None, 0
    cursor = source.execute("SELECT assembly_id,split,payload FROM assemblies WHERE dataset='natural' ORDER BY assembly_id")
    for number, (aid, split, payload) in enumerate(cursor, 1):
        rec = json.loads(payload)
        require(rec["assembly_id"] == aid and rec["dataset"] == "natural" and rec["split"] == split,
                "Natural record identity mismatch")
        by = defaultdict(list)
        for slot in rec["model_slots"]:
            by[slot["object_index"]].append(slot)
        for j, obj in enumerate(rec["route"]):
            slots = sorted(by[j], key=lambda s:s["slot"])
            clean = dict(obj, object_index=j)
            if obj["kind"] == 0:
                clean["canonical_domain_type"] = names[obj["domain_type"]]
            db.execute("INSERT INTO objects VALUES(?,?,?,?,?)", (aid,j,signature(rec,j),json.dumps(clean),json.dumps(slots)))
            object_count += 1
            for s in slots:
                if s["state"] != "OBSERVED":
                    continue
                ph, h = s["parent_sequence_hash"], s["sequence_hash"]
                sequence = parent(ph)[s["start"]:s["end"]]
                require(len(sequence) == s["length"] and seqsha(sequence) == h, "Natural feature sequence mismatch")
                if ph not in parent_written:
                    db.execute("INSERT INTO parents VALUES(?,?)", (ph,parent(ph)))
                    parent_written.add(ph)
                if h in known:
                    continue
                require(h in feature_rows, "Natural feature is missing from the original frozen cache")
                if stream is None:
                    stream = (working/f"vectors-{len(shards):03d}.f32").open("wb")
                vector = np.asarray(vectors[feature_rows[h]], dtype="<f4")
                require(np.isfinite(vector).all(), "Nonfinite original feature")
                stream.write(vector.tobytes())
                db.execute("INSERT INTO features VALUES(?,?,?,?)", (h,len(shards),shard_rows,sequence))
                known.add(h)
                shard_rows += 1
                if shard_rows == rows_per_shard:
                    finish_shard()
        if number % 1000 == 0:
            db.commit()
            print(json.dumps(dict(stage="export", assemblies=number, total=sum(counts.values()), unique_features=len(known))), flush=True)
    finish_shard()
    db.execute("CREATE INDEX objects_signature ON objects(signature)")
    db.commit()
    db.close()
    source.close()
    manifest = dict(schema="neuralps_natural_repertoire_v1", checkpoint_sha256=SSL_SHA,
        source_feature_sha256=FEATURE_SHA, feature_dtype="float32", feature_dim=1152,
        natural_assemblies_by_split=counts, unique_features=len(known), objects=object_count,
        database_sha256=sha(working/"repertoire.sqlite3"), shards=shards,
        scope="All natural assemblies; no engineered assemblies or phenotype labels; not the calibration cohort")
    dump(working/"manifest.json", manifest)
    index_archive = destination/f"NeurALPS_repertoire_index_{STAMP}.zip"
    archive(index_archive, [(working/n,n) for n in ("manifest.json","repertoire.sqlite3")])
    require(index_archive.stat().st_size < 2*1024**3, "Index asset exceeds GitHub's per-asset limit")
    return index_archive


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference-only", action="store_true")
    args = p.parse_args()
    require(not args.output.exists(), "Choose a new output folder; existing assets will not be overwritten")
    args.output.mkdir(parents=True)
    print("Reference:", export_reference(args.root.resolve(), args.output), flush=True)
    if not args.reference_only:
        print("Repertoire:", export_repertoire(args.root.resolve(), args.output), flush=True)
    assets = [f for f in sorted(args.output.glob("*.zip"))]
    dump(args.output/"UPLOAD_MANIFEST.json", {f.name:dict(sha256=sha(f),bytes=f.stat().st_size) for f in assets})
    print("COMPLETE. Upload every ZIP in:", args.output.resolve(), flush=True)


if __name__ == "__main__":
    main()
