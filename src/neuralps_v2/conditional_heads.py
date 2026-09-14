"""Native natural candidate scores and explicitly separate activity APIs.

PyTorch reference. Syntax checked here; runtime/GPU validation remains required.
Candidates are loss-side frozen vectors, never unmasked query inputs.
"""
import torch
from torch import nn
from torch.nn import functional as F

from neuralps_v2.neuralps_reference import NeurALPS, MaskDecoder, ActivityHead


def conditional_scores(pred, candidates, mean, valid, residual_weight=0.0):
    """Return [Q,K] scores for equally available target slots.

    pred: [Q,S,D], candidates: [Q,K,S,D], mean: [Q,S,D].
    valid: [Q,K]. S=1 for a single sequence, S=2 for paired termini.
    Missing required terminal slots invalidate a candidate, rather than changing
    the number of averaged slots. Candidate groups share the query's mean.
    Scores are conditional agreement in [-1,1], not activity probabilities.
    """
    if pred.ndim != 3 or candidates.ndim != 4:
        raise ValueError('Expected pred [Q,S,D] and candidates [Q,K,S,D].')
    if candidates.shape[0] != pred.shape[0] or candidates.shape[2:] != pred.shape[1:]:
        raise ValueError('Candidate and prediction slot geometry disagree.')
    if mean.shape != pred.shape or valid.shape != candidates.shape[:2]:
        raise ValueError('Mean or validity geometry disagrees.')
    if valid.dtype != torch.bool or not valid.any(-1).all():
        raise ValueError('Every query requires at least one valid candidate.')
    if not 0 <= residual_weight <= 1:
        raise ValueError('residual_weight must be in [0,1].')
    p = pred.float()[:, None]
    t = candidates.detach().float()
    mu = mean.detach().float()[:, None]
    if not torch.isfinite(p).all() or not torch.isfinite(mu).all():
        raise ValueError('Nonfinite prediction or training mean.')
    if not torch.isfinite(t[valid]).all():
        raise ValueError('Nonfinite valid candidate.')
    t = torch.where(valid[..., None, None], t, torch.zeros_like(t))
    raw = 1 - F.cosine_similarity(p, t, dim=-1, eps=1e-6)
    centered = 1 - F.cosine_similarity(p-mu, t-mu, dim=-1, eps=1e-6)
    use_centered = (t-mu).norm(dim=-1) > 1e-6
    loss = torch.where(use_centered,
                       (1-residual_weight)*raw + residual_weight*centered, raw)
    return (1-loss.mean(-1)).masked_fill(~valid, float('-inf'))


def candidate_rank_loss(scores, positive, valid, temperature=.07):
    """Per-query multi-positive loss. Aggregate by relation/kind/assembly outside."""
    if scores.shape != positive.shape or scores.shape != valid.shape:
        raise ValueError('Candidate masks must match [Q,K] scores.')
    if positive.dtype != torch.bool or valid.dtype != torch.bool:
        raise ValueError('Candidate masks must be boolean.')
    if (positive & ~valid).any() or not positive.any(-1).all():
        raise ValueError('Each query needs valid positive candidates.')
    if not (valid & ~positive).any(-1).all() or temperature <= 0:
        raise ValueError('Each training query needs an alternative and positive temperature.')
    if not torch.isfinite(scores[valid]).all():
        raise ValueError('Nonfinite valid candidate score.')
    logits = (scores.float()/temperature).masked_fill(~valid, float('-inf'))
    return torch.logsumexp(logits, -1) - torch.logsumexp(
        logits.masked_fill(~positive, float('-inf')), -1)


def reference_relative(scores, reference_scores, reference_valid):
    """Reference panel can be distinct from the currently displayed candidates."""
    if reference_scores.shape != reference_valid.shape:
        raise ValueError('Reference mask mismatch.')
    if scores.shape[0] != reference_scores.shape[0] or not reference_valid.any(-1).all():
        raise ValueError('Every query needs its fixed reference panel.')
    total = reference_scores.masked_fill(~reference_valid, 0).sum(-1)
    avg = total/reference_valid.sum(-1)
    return scores-avg[:, None]


class FoundationBundle(nn.Module):
    """One checkpoint identity; preserve separate natural and activity instances.

    Clean forward deliberately omits natural diagnostics. `rank_masked_slots`
    requires the caller's provenance-validated corrupted batch. No method here
    validates arbitrary biological edits or builds new ESM-C features.
    """
    def __init__(self, variant='A', domain_types=32, with_residue=False,
                 activity_head_kind='pooled'):
        super().__init__()
        self.encoder = NeurALPS(variant, domain_types=domain_types,
                               with_residue=with_residue)
        self.decoder = MaskDecoder()
        self.activity_head_kind = activity_head_kind
        if activity_head_kind == 'pooled':
            self.activity_head = ActivityHead()
        elif activity_head_kind == 'additive':
            from neuralps_v2.interpretability_head import AdditiveActivityHead
            self.activity_head = AdditiveActivityHead()
        else:
            raise ValueError('activity_head_kind must be pooled or additive.')

    def forward(self, batch, **encoder_kwargs):
        encoded = self.encoder(batch, **encoder_kwargs)
        logits = [self.activity_head(h, batch) for h in encoded['cycles']]
        return {
            'features': encoded['tokens'],
            'activity_logit': logits[-1],
            'cycle_activity_logits': logits,
            'cycle_features': encoded['cycles'],
        }

    def predict_masked(self, corrupted_batch, **encoder_kwargs):
        encoded = self.encoder(corrupted_batch, **encoder_kwargs)
        return {
            'teacher_prediction': self.decoder(encoded['tokens'], corrupted_batch.kind),
            'cycle_teacher_predictions': [
                self.decoder(h, corrupted_batch.kind) for h in encoded['cycles']],
        }

    def rank_masked_slots(self, corrupted_batch, query_rows, token_indices,
                          slot_indices, candidates, mean, valid,
                          residual_weight=0.0, **encoder_kwargs):
        """Indices [Q,S] identify each query's required target slots.

        query_rows [Q], token_indices [Q,S], slot_indices [Q,S]. The chosen
        slots must be primary masked targets in the caller's manifest, and
        candidates have shape [Q,K,S,1152].
        """
        if token_indices.shape != slot_indices.shape or token_indices.ndim != 2:
            raise ValueError('Expected matching [Q,S] token/slot indices.')
        if query_rows.shape != token_indices.shape[:1]:
            raise ValueError('Expected query_rows [Q].')
        from neuralps_v2.neuralps_reference import MASKED
        chosen_state = corrupted_batch.state[query_rows[:, None], token_indices, slot_indices]
        if not (chosen_state == MASKED).all():
            raise ValueError('Natural ranking requires explicitly masked target slots.')
        output = self.predict_masked(corrupted_batch, **encoder_kwargs)
        pred = output['teacher_prediction'][query_rows[:, None], token_indices, slot_indices]
        return conditional_scores(pred, candidates, mean, valid, residual_weight)


def initialize_child(parent, child):
    """Narrow architecture transfer; never use for legacy checkpoint migration."""
    transition = (parent.variant, child.variant)
    allowed = {('A','C'): ('global_blocks.', 'global_slots'),
               ('C','D'): ('recycle_h.', 'recycle_g.')}
    if transition not in allowed:
        raise ValueError('Only new A->C and C->D transfers are supported.')
    # Inspect keys/shapes before mutation; this function never silently ignores keys.
    source, destination = parent.state_dict(), child.state_dict()
    if any(k not in destination or destination[k].shape != v.shape for k,v in source.items()):
        raise ValueError('Parent input/core configuration does not match child.')
    missing = set(destination)-set(source)
    if not all(k.startswith(allowed[transition]) for k in missing):
        raise ValueError('Unexpected child parameters: '+str(sorted(missing)))
    child.load_state_dict(source, strict=False)
    return sorted(missing)
