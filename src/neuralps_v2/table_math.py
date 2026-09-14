"""NumPy oracle for the native table equations; not a trained predictor."""
import numpy as np


def cosine(a,b):
    a,b = np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
    a = a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-6)
    b = b/np.maximum(np.linalg.norm(b,axis=-1,keepdims=True),1e-6)
    return np.sum(a*b,axis=-1)


def conditional_scores(pred,candidates,mean,valid,residual_weight=0.0):
    p,t,mu = (np.asarray(x,dtype=np.float64) for x in (pred,candidates,mean))
    valid = np.asarray(valid,dtype=bool)
    if p.ndim != 3 or t.ndim != 4 or t.shape[0] != p.shape[0] or t.shape[2:] != p.shape[1:]:
        raise ValueError('Expected [Q,S,D] and [Q,K,S,D].')
    if mu.shape != p.shape or valid.shape != t.shape[:2] or not valid.any(-1).all():
        raise ValueError('Invalid mean/validity geometry.')
    if not np.isfinite(p).all() or not np.isfinite(mu).all() or not np.isfinite(t[valid]).all():
        raise ValueError('Nonfinite valid input.')
    if not 0 <= residual_weight <= 1:
        raise ValueError('Invalid residual weight.')
    t = np.where(valid[...,None,None],t,0)
    p,mu = p[:,None],mu[:,None]
    raw = 1-cosine(p,t)
    centered = 1-cosine(p-mu,t-mu)
    loss = np.where(np.linalg.norm(t-mu,axis=-1)>1e-6,
                    (1-residual_weight)*raw+residual_weight*centered,raw)
    return np.where(valid,1-loss.mean(-1),-np.inf)


def reference_relative(scores,reference_scores,reference_valid):
    scores,ref = np.asarray(scores),np.asarray(reference_scores)
    mask = np.asarray(reference_valid,dtype=bool)
    if ref.shape != mask.shape or scores.shape[0] != ref.shape[0] or not mask.any(-1).all():
        raise ValueError('Each query needs a fixed reference panel.')
    if not np.isfinite(ref[mask]).all():
        raise ValueError('Invalid reference score.')
    return scores - (np.where(mask,ref,0).sum(-1)/mask.sum(-1))[:,None]


def candidate_rank_loss(scores,positive,valid,temperature=.07):
    scores = np.asarray(scores,dtype=np.float64)
    pos,valid = np.asarray(positive,dtype=bool),np.asarray(valid,dtype=bool)
    if scores.shape != pos.shape or scores.shape != valid.shape:
        raise ValueError('Shape mismatch.')
    if (pos & ~valid).any() or not pos.any(-1).all() or not (valid & ~pos).any(-1).all():
        raise ValueError('Need valid positives and alternatives.')
    if temperature <= 0 or not np.isfinite(scores[valid]).all():
        raise ValueError('Invalid temperature or scores.')
    z = np.where(valid,scores/temperature,-np.inf)
    z = z - np.max(z,axis=-1,keepdims=True)
    numerator = np.where(pos,np.exp(z),0).sum(-1)
    denominator = np.exp(z).sum(-1)
    return -np.log(numerator/denominator)
