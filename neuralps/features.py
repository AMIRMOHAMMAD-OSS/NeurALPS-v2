"""Exact 2822-column typed_context pooling from full-context frozen states."""
import numpy as np
from .contracts import require

GROUPS = ((0, 256), (256, 1536), (1536, 2816))

def object_vectors(record, states):
    h = np.asarray(states, dtype=np.float64)
    route = record["route"]
    require(h.shape == (len(route), 256) and np.isfinite(h).all(), "Invalid frozen object states")
    kinds = [o["kind"] for o in route]
    require(set(kinds) <= {0, 1, 2} and 0 in kinds and any(k != 0 for k in kinds),
            "Typed-context classifier requires domains and boundaries")
    vectors = []
    for j, k in enumerate(kinds):
        if k == 0:
            v = h[j]
        else:
            require(0 < j < len(route)-1 and kinds[j-1] == kinds[j+1] == 0, "Invalid boundary neighbors")
            left, right = h[j-1], h[j+1]
            v = np.concatenate([left, h[j], right, left*right, np.abs(left-right)])
        vectors.append((k, v))
    return vectors

def typed_context(record, states):
    groups = [[], [], []]
    for k, v in object_vectors(record, states):
        groups[k].append(v)
    means = [np.mean(g, axis=0) if len(g) else np.zeros(256 if k == 0 else 1280) for k, g in enumerate(groups)]
    extra = np.asarray([v for g in groups for v in (np.log1p(len(g)), float(bool(g)))])
    return np.concatenate([*means, extra])

def contributions(record, states, model):
    """An exact additive decomposition of the logit, not causal fault scores."""
    vectors = object_vectors(record, states)
    counts = np.bincount([k for k, _ in vectors], minlength=3)
    coef = model["weight"] * model["active"] / model["scale"]
    mu = model["mean"]
    x = typed_context(record, states)
    context = float(model["intercept"] + np.dot(coef[2816:], (x-mu)[2816:]))
    for k, (lo, hi) in enumerate(GROUPS):
        if counts[k] == 0:
            context -= float(np.dot(coef[lo:hi], mu[lo:hi]))
    evidence = []
    for j, (k, v) in enumerate(vectors):
        lo, hi = GROUPS[k]
        evidence.append(dict(object_index=j, kind=k,
                             logit_contribution=float(np.dot(coef[lo:hi], v-mu[lo:hi])/counts[k]),
                             kind_absent_from_training=bool(mu[2816+2*k+1] == 0)))
    logit = float(np.dot(coef, x-mu)+model["intercept"])
    require(abs(context+sum(e["logit_contribution"] for e in evidence)-logit) < 1e-8,
            "Supervised contributions do not reconstruct logit")
    return dict(logit=logit, assembly_context_logit_term=context, objects=evidence,
                interpretation="Signed activity evidence; not a domain probability or causal defect score")
