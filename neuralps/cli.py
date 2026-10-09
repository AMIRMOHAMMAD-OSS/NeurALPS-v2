import argparse
import json
from pathlib import Path
from scipy.special import expit
from .contracts import require, dump, sha, SSL_SHA, FEATURE_SCHEMA

def analyze(spec, assets, head_dir=None, model_dir=None, device="cpu", modes=("local", "full")):
    from .inputs import build_record
    from .embedding import Embedder
    from .scoring import Runtime
    from .head import load_model
    from .features import contributions
    record, parents, sequences = build_record(spec)
    require(model_dir is not None, "A pinned local ESMC model directory is required")
    extractor = Embedder(model_dir, device="cuda", assets=assets)
    vectors = extractor.embed(sequences)
    gate = extractor.reference_gate
    del extractor
    import torch
    torch.cuda.empty_cache()
    runtime = Runtime(assets, device=device)
    result = runtime.score(record, parents, vectors, spec.get("joint_domain_indices"), modes=modes)
    result["embedding_reference_gate"] = gate
    if head_dir:
        hroot = Path(head_dir)
        manifest = json.loads((hroot/"manifest.json").read_text())
        require(manifest["checkpoint_sha256"] == SSL_SHA and manifest["feature_schema"] == FEATURE_SCHEMA,
                "Supervised head uses a different encoder or feature schema")
        require(sha(hroot/"supervised_head.npz") == manifest["model_sha256"], "Supervised head checksum differs")
        evidence = contributions(record, runtime.object_states(record, vectors), load_model(hroot/"supervised_head.npz"))
        result["supervised"] = dict(evidence, activity_score=float(expit(evidence["logit"])),
                                     model_sha256=manifest["model_sha256"], training_examples=manifest["trained_rows"],
                                     calibrated_probability=False)
    return result

def main():
    p = argparse.ArgumentParser(description="NeurALPS frozen compatibility and supervised activity")
    p.add_argument("input", type=Path, help="Ordered-protein annotated assembly JSON")
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--esmc-model", type=Path, required=True)
    p.add_argument("--head", type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    require(not args.output.exists(), "Choose a new output filename")
    result = analyze(json.loads(args.input.read_text()), args.assets, args.head, args.esmc_model, args.device)
    dump(args.output, result)
    print(args.output.resolve())

if __name__ == "__main__":
    main()
