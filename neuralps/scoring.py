"""Frozen reconstruction scores with the original physical masking closure."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from .contracts import require, sha, SSL_SHA
from .masking import masking_closure
from .inputs import region_targets

def slots_by_object(record):
    groups = defaultdict(list)
    for slot in record["model_slots"]:
        groups[slot["object_index"]].append(slot)
    return {j: sorted(ss, key=lambda s: s["slot"]) for j, ss in groups.items()}

def make_plan(record, targets, parents):
    targets = sorted(set(targets))
    by = slots_by_object(record)
    require(targets and all(j in by for j in targets), "Invalid target object indices")
    primary = [(j, s["slot"]) for j in targets for s in by[j]]
    if any(s["state"] != "OBSERVED" for j in targets for s in by[j]):
        return dict(status="MISSING_SEQUENCE_FEATURE", targets=targets, primary=primary, hidden=[])
    hidden = masking_closure(record, primary, parents.__getitem__, "independent_sequence")["hidden"]
    hidden_set = set(map(tuple, hidden))
    remaining = {}
    visible = defaultdict(set)
    for s in record["model_slots"]:
        if s["state"] == "OBSERVED" and (s["object_index"], s["slot"]) not in hidden_set:
            visible[s["object_index"]].add(s["sequence_hash"])
    for target in targets:
        seen, count = set(), 0
        for j in sorted(visible):
            if j != target and abs(j-target) <= 24 and visible[j]-seen:
                count += 1
                seen.update(visible[j])
        remaining[str(target)] = count
    n = sum(s["state"] == "OBSERVED" for s in record["model_slots"])
    return dict(status="READY" if min(remaining.values()) >= 2 else "INSUFFICIENT_VISIBLE_CONTEXT",
                targets=targets, primary=primary, hidden=hidden, masked_fraction=len(hidden)/n,
                remaining_context_by_target=remaining)

def predict_mode(model, batch, local=False):
    import torch
    if not local:
        return model.predict_masked(batch)
    require(bool(((batch["state"] == 4) & batch["valid"].unsqueeze(-1)).any()), "Explicit mask required")
    h = model.features(batch, context=False)
    b, t, _ = h.shape
    roles = model.decoder_role(torch.arange(3, device=h.device))[None, None].expand(b, t, -1, -1)
    kinds = model.decoder_kind(batch["kind"]).unsqueeze(2).expand(-1, -1, 3, -1)
    pred = model.decoder(torch.cat([h.unsqueeze(2).expand(-1, -1, 3, -1), roles, kinds], dim=-1))
    return torch.where(batch["valid"][:, :, None, None], pred, torch.zeros_like(pred))

def reconstruction_agreement(prediction, target, mean):
    import torch
    from torch.nn import functional as F
    prediction, target, mean = prediction.float(), target.float(), mean.float()
    raw = 1-F.cosine_similarity(prediction, target, dim=-1, eps=1e-6)
    centered = 1-F.cosine_similarity(prediction-mean, target-mean, dim=-1, eps=1e-6)
    loss = torch.where((target-mean).norm(dim=-1) > 1e-6, .25*raw+.75*centered, raw)
    return 1-loss.mean()

class Runtime:
    def __init__(self, assets, device="cpu"):
        import torch
        from .assembly_model import AssemblyEncoder, Config, collate_records
        root = Path(assets).resolve()
        manifest = json.loads((root/"manifest.json").read_text())
        require(manifest["source_checkpoint_sha256"] == SSL_SHA, "Wrong SSL26 checkpoint lineage")
        require(manifest.get("selected_step") == 26000, "Wrong checkpoint step")
        for name, digest in manifest["files_sha256"].items():
            p = (root/name).resolve()
            require(p.is_relative_to(root) and sha(p) == digest, "Asset checksum mismatch: "+name)
        cfg = json.loads((root/"encoder_config.json").read_text())
        self.model = AssemblyEncoder(Config(**cfg))
        with np.load(root/"encoder_weights.npz", allow_pickle=False) as z:
            state = {k: torch.from_numpy(z[k].copy()) for k in z.files}
        require(all(bool(torch.isfinite(v).all()) for v in state.values()), "Nonfinite model weights")
        self.model.load_state_dict(state, strict=True)
        self.model.to(device).eval().requires_grad_(False)
        with np.load(root/"target_means.npz", allow_pickle=False) as z:
            self.means = {k: np.asarray(z[k], dtype=np.float32) for k in z.files}
        self.mean_map = json.loads((root/"target_mean_map.json").read_text())
        require(all(v.shape == (1152,) and np.isfinite(v).all() for v in self.means.values()),
                "Invalid target mean vectors")
        self.device, self.collate, self.manifest = device, collate_records, manifest

    def target_mean(self, slot):
        specific = self.mean_map.get(slot["group"])
        if specific is not None and specific[1] >= 100:
            return self.means[specific[0]]
        broad = ("D" if slot["kind"] == 0 else "C") if slot["kind"] != 2 else slot["group"]
        require(broad in self.mean_map, "Missing training-only broad target mean")
        return self.means[self.mean_map[broad][0]]

    def object_states(self, record, vectors):
        import torch
        batch = self.collate([record], vectors.__getitem__)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.inference_mode():
            h = self.model.features(batch, context=True)[0]
        return h.cpu().numpy()

    def score_plans(self, record, parents, vectors, plans, modes=("local", "full"), microbatch=8):
        import torch
        require(set(modes) <= {"local", "full"} and bool(modes), "Unknown score mode")
        require(type(microbatch) is int and microbatch >= 1, "Invalid microbatch")
        by = slots_by_object(record)
        output, ready = {}, []
        for name, plan in plans.items():
            if plan["status"] != "READY":
                output[name] = dict(plan, objects={})
            else:
                ready.append((name, plan))
        for start in range(0, len(ready), microbatch):
            queries = ready[start:start+microbatch]
            batch = self.collate([record]*len(queries), vectors.__getitem__)
            for b, (_, plan) in enumerate(queries):
                for j, role in plan["hidden"]:
                    batch["state"][b, j, role] = 4
                    batch["x"][b, j, role] = float("nan")
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.inference_mode():
                preds = {mode: predict_mode(self.model, batch, local=(mode == "local")) for mode in modes}
                require(all(bool(torch.isfinite(v).all()) for v in preds.values()), "Nonfinite reconstruction")
                for b, (name, plan) in enumerate(queries):
                    scores = {}
                    for j in plan["targets"]:
                        slots = by[j]
                        roles = [s["slot"] for s in slots]
                        target = torch.as_tensor(np.stack([vectors[s["sequence_hash"]] for s in slots]), device=self.device)
                        mean = torch.as_tensor(np.stack([self.target_mean(s) for s in slots]), device=self.device)
                        scores[str(j)] = {mode: float(reconstruction_agreement(p[b, j, roles], target, mean).item())
                                          for mode, p in preds.items()}
                    output[name] = dict(plan, status="SCORED", objects=scores)
        return output

    def score(self, record, parents, vectors, domain_indices=None, modes=("local", "full")):
        plans = {f"object_{j}": make_plan(record, [j], parents) for j in range(len(record["route"]))}
        if domain_indices is not None:
            ds, bs = region_targets(record, domain_indices)
            plans["joint_region"] = make_plan(record, ds+bs, parents)
        scored = self.score_plans(record, parents, vectors, plans, modes=modes)
        objects = []
        for j, obj in enumerate(record["route"]):
            result = scored[f"object_{j}"]
            objects.append(dict(object_index=j, kind=obj["kind"],
                                label=obj.get("canonical_domain_type", "protein break" if obj["kind"] == 2 else "boundary"),
                                status=result["status"], raw=result["objects"].get(str(j)),
                                masking="single_physical_object_with_overlap_and_alias_closure"))
        output = dict(assembly_id=record["assembly_id"], checkpoint_sha256=SSL_SHA,
                      pretrained_map=objects, raw_score_range=[-1, 1],
                      score_meaning="Reconstruction agreement; not activity probability",
                      natural_reference_percentiles="Not provided for this score/mask contract")
        if domain_indices is not None:
            joint = scored["joint_region"]
            aggregate = {}
            if joint["status"] == "SCORED":
                for mode in modes:
                    d = float(np.mean([joint["objects"][str(j)][mode] for j in ds]))
                    b = float(np.mean([joint["objects"][str(j)][mode] for j in bs]))
                    aggregate[mode] = dict(domains=d, boundaries=b, balanced=.5*(d+b),
                                           neighborhood_mean=float(np.mean([joint["objects"][str(j)][mode] for j in ds+bs])))
            output["joint_region"] = dict(joint, domain_indices=ds, boundary_indices=bs, aggregate=aggregate)
        return output
