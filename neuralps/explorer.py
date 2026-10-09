"""A live Colab assembly explorer and a self-contained, inspectable HTML export."""
import copy
import json
import secrets
from pathlib import Path
import threading
from scipy.special import expit
from .contracts import require, sha, SSL_SHA, FEATURE_SCHEMA
from .inputs import build_record
from .scoring import Runtime, slots_by_object
from .segments import score_segment


def cartoon_objects(record, result):
    by = slots_by_object(record)
    scored = {o["object_index"]:o for o in result["pretrained_map"]}
    out = []
    for j, obj in enumerate(record["route"]):
        o = copy.deepcopy(scored[j])
        slots = by[j]
        name = obj.get("canonical_domain_type", "Protein break" if obj["kind"] == 2 else "Boundary")
        if obj["kind"] != 0:
            name = " → ".join(record["route"][k].get("canonical_domain_type", "?") for k in (j-1,j+1))
        short = ("A" if name == "AMP-binding" else "T" if name in ("PCP", "ACP") else
                 "C" if name.startswith("Condensation") else "TE" if name == "Thioesterase" else
                 "E" if name == "Epimerization" else name[:3])
        shape = "boundary" if obj["kind"] == 1 else "break" if obj["kind"] == 2 else {
            "A":"circle", "T":"square", "C":"diamond", "TE":"hexagon", "E":"hexagon"}.get(short, "rounded")
        o.update(name=name, short=short, shape=shape, chain=obj.get("chain"),
                 protein_uid=obj.get("protein_uid"),
                 spans=[dict(protein=s.get("protein_uid"), start=s.get("start"), end=s.get("end"),
                             role=s["slot"], length=s.get("length")) for s in slots])
        out.append(o)
    return out


class ExplorerSession:
    def __init__(self, spec, record, parents, vectors, runtime, head_dir=None, reference=None,
                 repertoire=None, repertoire_factory=None, model_dir=None, assets=None):
        self.spec, self.record, self.parents, self.vectors = spec, record, parents, vectors
        self.runtime, self.head_dir, self.reference = runtime, head_dir, reference
        self.repertoire, self.repertoire_factory = repertoire, repertoire_factory
        self.model_dir, self.assets = model_dir, assets
        self.result = runtime.score(record, parents, vectors, spec.get("joint_domain_indices"))
        if reference:
            reference.annotate(self.result, record, parents)
        if head_dir:
            from .head import load_model
            from .features import contributions
            root = Path(head_dir)
            manifest = json.loads((root/"manifest.json").read_text())
            require(manifest["checkpoint_sha256"] == SSL_SHA and manifest["feature_schema"] == FEATURE_SCHEMA,
                    "Supervised model contract differs")
            require(sha(root/"supervised_head.npz") == manifest["model_sha256"], "Supervised head checksum differs")
            evidence = contributions(record, runtime.object_states(record, vectors), load_model(root/"supervised_head.npz"))
            self.result["supervised"] = dict(evidence, activity_score=float(expit(evidence["logit"])),
                model_sha256=manifest["model_sha256"], training_examples=manifest["trained_rows"], calibrated_probability=False)
        self.searches, self.segments, self.variants = {}, {}, {}
        self.lock = threading.Lock()

    @classmethod
    def from_spec(cls, spec, assets, model_dir, head_dir=None, reference=None, repertoire_factory=None):
        from .embedding import Embedder
        import torch
        record, parents, sequences = build_record(spec)
        extractor = Embedder(model_dir, device="cuda", assets=assets)
        vectors = extractor.embed(sequences)
        gate = extractor.reference_gate
        del extractor
        torch.cuda.empty_cache()
        session = cls(spec, record, parents, vectors, Runtime(assets), head_dir, reference,
                      repertoire_factory=repertoire_factory, model_dir=model_dir, assets=assets)
        session.result["embedding_reference_gate"] = gate
        return session

    def payload(self):
        return dict(schema="neuralps_explorer_v1", assembly_id=self.record["assembly_id"],
                    objects=cartoon_objects(self.record, self.result), result=self.result,
                    reference_available=self.reference is not None,
                    repertoire_available=self.repertoire is not None or self.repertoire_factory is not None,
                    segment_results=self.segments, candidate_results=self.searches,
                    source="Live model session", coordinates="0-based, end-exclusive",
                    can_test_variants=bool(self.assets and self.model_dir))

    def handle(self, request):
        require(isinstance(request, dict), "Expected an explorer request")
        require(self.lock.acquire(blocking=False), "Another model request is running; wait for it to finish")
        try:
            action = request.get("action")
            if action == "state":
                return self.payload()
            selected = request.get("indices", [])
            mode = request.get("mode", "full")
            touching = bool(request.get("touching_boundaries", False))
            segment = score_segment(self.runtime, self.record, self.parents, self.vectors, selected, mode, touching)
            key = mode+":"+",".join(map(str,segment["targets"]))
            self.segments[key] = segment
            if action == "score":
                return dict(segment=segment, key=key)
            require(segment["status"] == "SCORED", "This selection cannot be scored: " + segment["status"])
            if self.repertoire is None:
                require(self.repertoire_factory is not None, "Export and publish the natural repertoire on Jean Zay first")
                self.repertoire = self.repertoire_factory()
            if action == "rank":
                if key not in self.searches:
                    self.searches[key] = self.repertoire.rank(self.runtime, self.record, self.parents, self.vectors, segment["targets"], mode)
                return dict(segment=segment, ranking=self.searches[key], key=key)
            require(action in ("variant", "test_variant"), "Unknown explorer action")
            allowed = [x["candidate_id"] for x in self.searches.get(key, {}).get("top_candidates", [])]
            require(request.get("candidate_id") in allowed, "Rank candidates before choosing a variant")
            variant = self.repertoire.variant_spec(self.spec, self.record, segment["targets"], request["candidate_id"])
            if action == "variant":
                return dict(spec=variant)
            require(self.assets and self.model_dir, "Live ESMC extraction is required to test the rebuilt variant")
            variant_session = self.from_spec(variant, self.assets, self.model_dir, self.head_dir, self.reference, self.repertoire_factory)
            self.variants[variant["assembly_id"]] = variant_session
            scored = variant_session.result
            variant_segment = score_segment(variant_session.runtime, variant_session.record,
                variant_session.parents, variant_session.vectors, segment["targets"], mode)
            self.segments[key]["tested_variant"] = dict(spec=variant, result=scored)
            return dict(spec=variant, result=scored,
                        base_segment=segment, variant_segment=variant_segment,
                        base_activity_score=self.result.get("supervised", {}).get("activity_score"),
                        variant_activity_score=scored.get("supervised", {}).get("activity_score"),
                        model_score_not_calibrated_probability=True,
                        note="Complete-domain splice rescored after rebuilding and re-embedding its physical boundaries. No experiment was performed.")
        finally:
            self.lock.release()


def render_html(payload, callback=None):
    template = (Path(__file__).parent/"data/explorer.html").read_text()
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    bridge = json.dumps(callback)
    return template.replace("__NEURALPS_DATA__", data).replace("__NEURALPS_CALLBACK__", bridge)


def show_explorer(session):
    from IPython.display import HTML, JSON, display
    from google.colab import output
    name = "neuralps.explorer."+secrets.token_hex(8)
    def callback(request):
        try:
            return JSON(dict(ok=True, value=session.handle(request)))
        except Exception as e:
            return JSON(dict(ok=False, error=str(e)))
    output.register_callback(name, callback)
    display(HTML(render_html(session.payload(), name)))
    return name


def save_explorer(session, destination):
    destination = Path(destination)
    destination.write_text(render_html(session.payload()), encoding="utf-8")
    return destination
