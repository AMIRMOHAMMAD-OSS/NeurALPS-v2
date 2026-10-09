"""ESMC extraction pinned to the historically validated fork and weights."""
import importlib.metadata
import json
from pathlib import Path
import numpy as np
from .contracts import require, sha, ESMC_SHA, ESMC_REPO, ESMC_REVISION, FORK_COMMIT, seqsha

def download_esmc(destination):
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo_id=ESMC_REPO, revision=ESMC_REVISION, local_dir=str(destination),
                             allow_patterns=["*.json", "model.safetensors"])
    require(sha(Path(path)/"model.safetensors") == ESMC_SHA, "Downloaded ESMC checkpoint differs")
    return Path(path)

class Embedder:
    def __init__(self, model_dir, device="cuda", assets=None):
        import torch
        import transformers
        from transformers import AutoModel, AutoTokenizer
        require(torch.cuda.is_available() and str(device).startswith("cuda"), "Exact ESMC extraction needs CUDA")
        require(torch.cuda.is_bf16_supported(), "Exact historical BF16 mode requires a BF16-capable GPU")
        require(transformers.__version__ == "4.57.6", "Install the pinned Biohub Transformers fork")
        require(assets is not None, "The exported runtime and historical references are required")
        runtime_manifest = json.loads((Path(assets)/"manifest.json").read_text())
        fork = runtime_manifest["transformers_distribution"]
        require(fork["original_vcs_commit"] == FORK_COMMIT, "Wrong exported Transformers fork")
        package = Path(transformers.__file__).resolve().parent
        for relative, digest in fork["source_sha256"].items():
            source = (package/relative).resolve()
            require(source.is_relative_to(package) and sha(source) == digest,
                    "Installed Transformers source differs from the exported fork: "+relative)
        model_dir = Path(model_dir)
        require(sha(model_dir/"model.safetensors") == ESMC_SHA, "ESMC weights differ from original cache")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        require(type(self.tokenizer).__name__ == "ESMCTokenizer", "Unexpected ESMC tokenizer")
        self.model = AutoModel.from_pretrained(model_dir, dtype=torch.bfloat16, local_files_only=True)
        require(type(self.model).__name__ == "ESMCModel", "Unexpected ESMC model class")
        self.model.to(device).eval().requires_grad_(False)
        self.reference_gate = None
        require(assets is not None, "Historical ESMC reference vectors are required")
        self.verify_references(assets)

    def _batch(self, sequences):
        import torch
        require(sequences and all(isinstance(s, str) and 0 < len(s) <= 2046 and
                                   set(s) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for s in sequences),
                "Each independent input must contain 1–2046 amino acids; no silent truncation")
        encoded = self.tokenizer(sequences, return_tensors="pt", padding=True, truncation=False,
                                 add_special_tokens=True, return_special_tokens_mask=True)
        special = encoded.pop("special_tokens_mask").to(self.device)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        require(encoded["input_ids"].shape[1] <= 2048, "ESMC token limit exceeded")
        aa = encoded["attention_mask"].bool() & ~special.bool()
        for i, sequence in enumerate(sequences):
            ids = encoded["input_ids"][i][aa[i]].tolist()
            require(self.tokenizer.convert_ids_to_tokens(ids) == list(sequence), "Tokenizer residue round-trip failed")
        with torch.inference_mode():
            hidden = self.model(**encoded).last_hidden_state
            pooled = (hidden*aa.unsqueeze(-1).to(hidden.dtype)).sum(dim=1, dtype=torch.float32) / aa.sum(1, keepdim=True).float()
        require(tuple(pooled.shape) == (len(sequences), 1152) and bool(torch.isfinite(pooled).all()),
                "Invalid ESMC output")
        return pooled.cpu().numpy()

    def verify_references(self, assets):
        root = Path(assets)
        manifest = json.loads((root/"manifest.json").read_text())
        for name in ("esmc_references.json", "esmc_reference_vectors.npz"):
            require(sha(root/name) == manifest["files_sha256"][name], "Reference asset changed")
        refs = json.loads((root/"esmc_references.json").read_text())
        with np.load(root/"esmc_reference_vectors.npz", allow_pickle=False) as z:
            cached = {k: z[k] for k in z.files}
        comparisons = []
        for start in range(0, len(refs), 4):
            rows = refs[start:start+4]
            fresh = self._batch([r["seq"] for r in rows])
            for r, vector in zip(rows, fresh):
                require(seqsha(r["seq"]) == r["hash"], "Reference sequence hash differs")
                old = cached[r["hash"]]
                cosine = float(np.dot(vector, old)/(np.linalg.norm(vector)*np.linalg.norm(old)))
                relative = float(np.linalg.norm(vector-old)/np.linalg.norm(old))
                comparisons.append(dict(hash=r["hash"], cosine=cosine, relative_l2=relative))
        require(comparisons and min(r["cosine"] for r in comparisons) >= .999 and
                max(r["relative_l2"] for r in comparisons) <= .02,
                "Historical ESMC reference gate failed; embeddings are not compatible")
        self.reference_gate = dict(status="PASS", comparisons=comparisons,
                                   tolerance="Historical engineering tolerances, not bitwise equivalence")

    def embed(self, sequences, batch_size=4):
        import torch
        require(type(batch_size) is int and batch_size >= 1, "Invalid ESMC batch size")
        rows = sorted(sequences.items(), key=lambda x: (len(x[1]), x[0]))
        require(rows and all(seqsha(s) == h for h, s in rows), "Sequence identities differ")
        result, i, size = {}, 0, batch_size
        while i < len(rows):
            batch = rows[i:i+size]
            try:
                values = self._batch([s for _, s in batch])
            except torch.cuda.OutOfMemoryError:
                if size == 1:
                    raise
                torch.cuda.empty_cache()
                size = max(1, size//2)
                continue
            result.update({h: v for (h, _), v in zip(batch, values)})
            i += len(batch)
        return result
