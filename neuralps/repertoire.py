"""Retrieve natural fragments against predictions in the user's masked assembly."""
import copy
import hashlib
import heapq
import json
from pathlib import Path
import sqlite3
import numpy as np
from .contracts import require, sha, SSL_SHA
from .scoring import slots_by_object
from .segments import masked_predictions, candidate_agreements


def signature(record, j):
    route = record["route"]
    obj = route[j]
    types = [obj["domain_type"]] if obj["kind"] == 0 else [route[j-1]["domain_type"], route[j+1]["domain_type"]]
    return json.dumps([obj["kind"], types], separators=(",", ":"))


class Repertoire:
    def __init__(self, directory, fetch_shard=None):
        self.root = Path(directory).resolve()
        self.manifest = json.loads((self.root/"manifest.json").read_text())
        require(self.manifest["schema"] == "neuralps_natural_repertoire_v1" and
                self.manifest["checkpoint_sha256"] == SSL_SHA, "Wrong natural repertoire contract")
        require(sha(self.root/"repertoire.sqlite3") == self.manifest["database_sha256"], "Repertoire index checksum mismatch")
        self.db = sqlite3.connect((self.root/"repertoire.sqlite3").as_uri()+"?mode=ro", uri=True, check_same_thread=False)
        self.fetch_shard = fetch_shard
        self.arrays = {}

    def vector(self, h):
        info = self.db.execute("SELECT shard,row_index FROM features WHERE hash=?", (h,)).fetchone()
        require(info is not None, "Candidate feature absent: " + h)
        shard, row = info
        if shard not in self.arrays:
            meta = self.manifest["shards"][shard]
            path = self.root/meta["path"]
            if not path.exists() and self.fetch_shard:
                self.fetch_shard(meta, self.root)
            require(path.exists(), "Natural-repertoire vector shard is not downloaded: " + meta["archive"])
            require(path.stat().st_size == meta["rows"]*1152*4 and sha(path) == meta["sha256"], "Candidate vector checksum mismatch")
            self.arrays[shard] = np.memmap(path, mode="r", dtype="<f4", shape=(meta["rows"], 1152))
        return self.arrays[shard][row]

    def fragment(self, aid, start, count):
        rows = self.db.execute("SELECT j,signature,object_json,slots_json FROM objects WHERE assembly_id=? AND j>=? AND j<? ORDER BY j", (aid, start, start+count)).fetchall()
        if len(rows) != count or [r[0] for r in rows] != list(range(start, start+count)):
            return None
        return dict(assembly_id=aid, start=start, count=count, signatures=[r[1] for r in rows],
                    objects=[json.loads(r[2]) for r in rows], slots=[json.loads(r[3]) for r in rows])

    def fragments(self, signatures):
        cursor = self.db.execute("SELECT assembly_id,j FROM objects WHERE signature=? ORDER BY assembly_id,j", (signatures[0],))
        for aid, start in cursor:
            frag = self.fragment(aid, start, len(signatures))
            if frag is not None and frag["signatures"] == signatures:
                yield frag

    def describe(self, frag, score):
        sequences = []
        for obj, slots in zip(frag["objects"], frag["slots"]):
            for slot in slots:
                row = self.db.execute("SELECT sequence FROM features WHERE hash=?", (slot["sequence_hash"],)).fetchone()
                sequences.append(dict(object_index=slot["object_index"], role=slot["slot"],
                    label=obj.get("canonical_domain_type", "boundary" if obj["kind"] == 1 else "protein terminus"),
                    sequence=row[0], sequence_hash=slot["sequence_hash"], protein_uid=slot.get("protein_uid"),
                    start=slot["start"], end=slot["end"]))
        return dict(candidate_id=[frag["assembly_id"], frag["start"], frag["count"]],
                    assembly_id=frag["assembly_id"], first_object=frag["start"], last_object=frag["start"]+frag["count"]-1,
                    score=float(score), sequences=sequences)

    def rank(self, runtime, record, parents, vectors, targets, mode="full", top_k=20):
        targets = sorted(targets)
        require(targets == list(range(targets[0], targets[-1]+1)),
                "Repertoire search needs a contiguous segment. Use Shift-click to select a range.")
        require(type(top_k) is int and 1 <= top_k <= 100, "top_k must be between 1 and 100")
        sigs = [signature(record, j) for j in targets]
        pred, means, current, weights, visible = masked_predictions(runtime, record, parents, vectors, targets, mode)
        incumbent = float(candidate_agreements(pred, means, current[None], weights)[0])
        own = tuple(s["sequence_hash"] for j in targets for s in slots_by_object(record)[j])
        seen, pending, top = set(), [], []
        counts = dict(eligible_unique=0, higher=0, equal=0, lower=0, identical_to_query=0,
                      duplicate_fragments=0, visible_sequence_alias=0, missing_features=0, matched_occurrences=0)
        tolerance = 2e-6
        def flush():
            if not pending:
                return
            values = np.stack([np.stack([self.vector(h) for h in hashes]) for _, hashes in pending])
            scores = candidate_agreements(pred, means, values, weights)
            require(np.isfinite(scores).all(), "Nonfinite candidate scores")
            for (frag, hashes), score in zip(pending, scores):
                score = float(score)
                counts["eligible_unique"] += 1
                key = "higher" if score > incumbent+tolerance else "lower" if score < incumbent-tolerance else "equal"
                counts[key] += 1
                # Candidate sequence identity gives deterministic ordering for ties.
                tie = hashlib.sha256("|".join(hashes).encode()).hexdigest()
                item = (score, tie, frag)
                if len(top) < top_k:
                    heapq.heappush(top, item)
                elif item[:2] > top[0][:2]:
                    heapq.heapreplace(top, item)
            pending.clear()
        for frag in self.fragments(sigs):
            counts["matched_occurrences"] += 1
            ss = [s for slots in frag["slots"] for s in slots]
            if any(s["state"] != "OBSERVED" or not s.get("sequence_hash") for s in ss):
                counts["missing_features"] += 1
                continue
            hashes = tuple(s["sequence_hash"] for s in ss)
            if hashes == own:
                counts["identical_to_query"] += 1
                continue
            if hashes in seen:
                counts["duplicate_fragments"] += 1
                continue
            seen.add(hashes)
            if visible.intersection(hashes):
                counts["visible_sequence_alias"] += 1
                continue
            require(len(hashes) == len(weights), "Candidate slot topology differs")
            pending.append((frag, hashes))
            if len(pending) == 128:
                flush()
        flush()
        n = counts["eligible_unique"]
        return dict(status="RANKED" if n else "NO_ELIGIBLE_CANDIDATES", mode=mode, targets=targets,
                    incumbent_score=incumbent, counts=counts, comparison_tolerance=tolerance,
                    repertoire_percentile_0_to_100=100*(counts["lower"]+.5*counts["equal"])/n if n else None,
                    top_candidates=[self.describe(frag, score) for score, _, frag in sorted(top, key=lambda x:x[:2], reverse=True)],
                    corpus=self.manifest["natural_assemblies_by_split"],
                    meaning="Matching natural fragments ranked against the same masked-query predictions; exact feature duplicates collapsed. A retrieval ranking, not a measured activity prediction.",
                    junctions_rebuilt=False, candidate_embeddings="original native physical features, float32")

    def variant_spec(self, spec, record, targets, candidate_id):
        """Splice complete domains on one physical chain; rebuild seams on inference."""
        require(isinstance(candidate_id, list) and len(candidate_id) == 3, "Invalid candidate identifier")
        aid, start, count = candidate_id
        require(isinstance(aid, str) and type(start) is int and type(count) is int, "Invalid candidate identifier")
        require(sorted(targets) == list(range(min(targets), max(targets)+1)), "Variant selection must be contiguous")
        frag = self.fragment(aid, start, count)
        require(frag is not None and frag["signatures"] == [signature(record, j) for j in sorted(targets)], "Candidate topology differs")
        query_domains = [record["route"][j] for j in targets if record["route"][j]["kind"] == 0]
        donor_domains = [o for o in frag["objects"] if o["kind"] == 0]
        require(query_domains and donor_domains, "Boundary-only retrieval: download its feature FASTA; a complete-domain selection is required to build a variant")
        require(len({o["protein_uid"] for o in query_domains}) == len({o["protein_uid"] for o in donor_domains}) == 1,
                "Automatic splicing supports a segment within one physical protein. Cross-protein candidates can be exported as FASTA.")
        a, b = query_domains[0]["start_aa_0based"], query_domains[-1]["end_aa_0based_exclusive"]
        da, db = donor_domains[0]["start_aa_0based"], donor_domains[-1]["end_aa_0based_exclusive"]
        slot = next(s for ss in frag["slots"] for s in ss if s["object_index"] == donor_domains[0]["object_index"])
        parent = self.db.execute("SELECT sequence FROM parents WHERE hash=?", (slot["parent_sequence_hash"],)).fetchone()
        require(parent is not None, "Donor physical protein is unavailable")
        replacement = parent[0][da:db]
        require(replacement, "Empty replacement sequence")
        result = copy.deepcopy(spec)
        result.pop("joint_domain_indices", None)
        result["assembly_id"] = spec["assembly_id"]+"__donor_"+hashlib.sha256(json.dumps(candidate_id).encode()).hexdigest()[:10]
        p = next(p for p in result["proteins"] if p["id"] == query_domains[0]["protein_uid"])
        delta = len(replacement)-(b-a)
        keep = []
        for d in p["domains"]:
            if d["end"] <= a:
                keep.append(d)
            elif d["start"] >= b:
                keep.append(dict(d, start=d["start"]+delta, end=d["end"]+delta))
            else:
                require(a <= d["start"] < d["end"] <= b, "Overlapping domain annotation crosses a splice boundary")
        keep.extend(dict(type=d["canonical_domain_type"], start=a+d["start_aa_0based"]-da,
                         end=a+d["end_aa_0based_exclusive"]-da) for d in donor_domains)
        p["domains"] = sorted(keep, key=lambda d:d["start"])
        p["sequence"] = p["sequence"][:a]+replacement+p["sequence"][b:]
        result["variant_provenance"] = dict(donor=candidate_id, recipient_object_indices=targets,
            splice="complete-domain span; recipient flanks retained; all boundary features must be rebuilt and re-embedded")
        return result
