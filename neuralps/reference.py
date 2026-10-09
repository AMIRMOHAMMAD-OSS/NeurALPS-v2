"""Apply the original Jean Zay scale only to its exact single-object contract."""
import bisect
import json
from pathlib import Path
from .contracts import require, sha, SSL_SHA
from .scoring import make_plan, slots_by_object
from .reference_scale_original import apply_one


def reference_row(record, parents, object_index, raw_score, status="SCORED"):
    j = object_index
    obj = record["route"][j]
    slots = slots_by_object(record)[j]
    plan = make_plan(record, [j], parents)
    types = ([obj["domain_type"]] if obj["kind"] == 0 else
             [record["route"][j-1]["domain_type"], record["route"][j+1]["domain_type"]])
    return dict(assembly_id=record["assembly_id"], object_index=j, kind=obj["kind"],
                domain_types=types, mode="joint_terminal_recovery" if obj["kind"] == 2 else "object_recovery",
                slots=[dict(s, length_bin=max(0, int(s.get("length") or 0).bit_length()-1)) for s in slots],
                hidden_fraction=plan.get("masked_fraction"),
                native_hashes=[s["sequence_hash"] for s in slots if s.get("sequence_hash")],
                parent_hashes=sorted({s["parent_sequence_hash"] for s in slots if s.get("parent_sequence_hash")}),
                native_score=raw_score, status=status)


class NaturalReference:
    def __init__(self, directory):
        root = Path(directory).resolve()
        proof = json.loads((Path(__file__).parent / "data/reference_provenance.json").read_text())
        for name, digest in proof["files_sha256"].items():
            require(sha(root/name) == digest, "Original natural reference changed: " + name)
        self.protocol = json.loads((root/"protocol.json").read_text())
        self.reference = json.loads((root/"reference.json").read_text())
        require(self.protocol["checkpoint_sha256"] == SSL_SHA and self.protocol["model_path"] == "full_context",
                "Reference belongs to a different encoder or scoring contract")
        self.audit = {g["reference_key"]: g for g in json.loads((root/"natural_scale_audit.json").read_text())["groups"]}
        self.provenance = proof

    def apply(self, record, parents, j, raw_score, status="SCORED", mode="full"):
        if mode != "full":
            return dict(scale_status="UNSUPPORTED_CONTEXT_MODE", natural_percentile_0_to_100=None)
        scaled = apply_one(reference_row(record, parents, j, raw_score, status), self.reference, self.protocol)
        keys = ["natural_percentile_0_to_100", "lower_tail_rank", "reference_components",
                "reference_candidate_objects", "reference_level", "tail_resolution", "warning", "scale_status",
                "reference_feature_overlap_components", "reference_parent_overlap_components"]
        out = {k: scaled[k] for k in keys}
        if scaled["reference_key"]:
            group = self.reference["groups"][scaled["reference_key"]]
            left = bisect.bisect_left(group["scores"], raw_score)
            right = bisect.bisect_right(group["scores"], raw_score)
            out.update(reference_lower=left, reference_equal=right-left, reference_higher=group["n"]-right,
                       reference_audit_status=self.audit.get(scaled["reference_key"], {}).get("rate_assessment"))
        out["meaning"] = "Percentile among matched natural objects in their original contexts; not replacement rank or activity probability"
        return out

    def annotate(self, result, record, parents):
        for obj in result["pretrained_map"]:
            obj["natural_reference"] = self.apply(record, parents, obj["object_index"],
                (obj.get("raw") or {}).get("full"), obj["status"])
        result["natural_reference_percentiles"] = "Original full-context single-object reference"
        result["reference_provenance"] = self.provenance
        return result
