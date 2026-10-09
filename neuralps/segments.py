"""Joint physical-object selection, preserving overlap and alias masking."""
import numpy as np
from .contracts import require
from .scoring import make_plan, predict_mode, slots_by_object


def selection(record, indices, touching_boundaries=False):
    require(isinstance(indices, (list, tuple)) and indices and
            all(type(j) is int and 0 <= j < len(record["route"]) for j in indices),
            "Select one or more valid domains or boundaries")
    targets = set(indices)
    if touching_boundaries:
        for j in indices:
            if record["route"][j]["kind"] == 0:
                targets.update(k for k in (j-1, j+1) if 0 <= k < len(record["route"]) and record["route"][k]["kind"] != 0)
    return sorted(targets)


def object_weights(record, targets):
    domain = [record["route"][j]["kind"] == 0 for j in targets]
    nd, nb = sum(domain), len(domain)-sum(domain)
    if nd and nb:
        return np.asarray([.5/nd if d else .5/nb for d in domain]), "equal domain/boundary weight"
    return np.full(len(targets), 1/len(targets)), "equal physical-object weight"


def score_segment(runtime, record, parents, vectors, indices, mode="full", touching_boundaries=False):
    require(mode in ("full", "local"), "Unknown context mode")
    targets = selection(record, indices, touching_boundaries)
    plan = make_plan(record, targets, parents)
    result = runtime.score_plans(record, parents, vectors, {"selection": plan}, modes=(mode,))["selection"]
    weights, aggregation = object_weights(record, targets)
    result.update(mode=mode, aggregation=aggregation, score=None,
                  additional_hidden_objects=sorted({j for j, _ in result.get("hidden", [])}-set(targets)),
                  natural_percentile_0_to_100=None)
    if result["status"] == "SCORED":
        result["score"] = float(sum(w*result["objects"][str(j)][mode] for j, w in zip(targets, weights)))
    return result


def masked_predictions(runtime, record, parents, vectors, targets, mode):
    import torch
    plan = make_plan(record, targets, parents)
    require(plan["status"] == "READY", "Cannot rank this selection: " + plan["status"])
    batch = runtime.collate([record], vectors.__getitem__)
    for j, role in plan["hidden"]:
        batch["state"][0, j, role] = 4
        batch["x"][0, j, role] = float("nan")
    batch = {k: v.to(runtime.device) for k, v in batch.items()}
    by = slots_by_object(record)
    slots = [s for j in targets for s in by[j]]
    weights, _ = object_weights(record, targets)
    slot_weights = np.asarray([w/len(by[j]) for j, w in zip(targets, weights) for _ in by[j]])
    with torch.inference_mode():
        pred = predict_mode(runtime.model, batch, mode == "local")[0]
        predictions = np.stack([pred[s["object_index"], s["slot"]].cpu().numpy() for s in slots])
    means = np.stack([runtime.target_mean(s) for s in slots])
    incumbent = np.stack([vectors[s["sequence_hash"]] for s in slots])
    hidden = set(map(tuple, plan["hidden"]))
    visible_hashes = {s["sequence_hash"] for s in record["model_slots"] if s["state"] == "OBSERVED" and (s["object_index"], s["slot"]) not in hidden}
    return predictions, means, incumbent, slot_weights, visible_hashes


def candidate_agreements(predictions, means, values, slot_weights):
    """Vectorized exact loss formula, averaging break termini once per object."""
    values = np.asarray(values, dtype=np.float32)
    def cosine(a, b):
        return np.sum(a*b, axis=-1) / (np.maximum(np.linalg.norm(a, axis=-1), 1e-6)*np.maximum(np.linalg.norm(b, axis=-1), 1e-6))
    raw = cosine(predictions, values)
    centred = cosine(predictions-means, values-means)
    agreement = np.where(np.linalg.norm(values-means, axis=-1) > 1e-6, .25*raw+.75*centred, raw)
    return np.sum(agreement * slot_weights, axis=-1)
