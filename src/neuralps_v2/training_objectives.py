"""Phase2.2 losses and pass aggregation.

This file deliberately consumes already validated batches. It does not load
FASTA/TSV files and cannot perform provenance closure. The data adapter must
construct ``Batch`` plus detached teacher targets after applying
``masking_contract.mask_slots``.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from neuralps_v2.neuralps_reference import (
    Batch,
    DOMAIN,
    COVALENT,
    BREAK,
    masked_teacher_loss,
)
from neuralps_v2.conditional_heads import candidate_rank_loss, conditional_scores


def _validate_ssl_tensors(pred, teacher, mean, primary_target, kind):
    if pred.shape != teacher.shape or pred.shape != mean.shape:
        raise ValueError("Prediction, teacher and mean shapes must agree.")
    if pred.ndim != 4 or pred.shape[-1] != 1152:
        raise ValueError("Expected [batch,tokens,slots,1152] vectors.")
    expected = pred.shape[:3]
    if primary_target.shape != expected or kind.shape != expected[:2]:
        raise ValueError("Primary-target or kind shape mismatch.")
    if primary_target.dtype != torch.bool:
        raise ValueError("primary_target must be bool.")
    if not torch.isfinite(teacher[primary_target]).all() or not torch.isfinite(mean[primary_target]).all():
        raise ValueError('Nonfinite primary teacher/mean.')
    if not torch.isfinite(pred).all():
        raise ValueError("Student produced a nonfinite masked prediction.")


def object_ssl_loss(pred, teacher, mean, primary_target, kind,
                    residual_weight: float = 0.0) -> torch.Tensor:
    """Compute the relation/group/assembly-normalized object loss."""
    _validate_ssl_tensors(pred, teacher, mean, primary_target, kind)
    if not primary_target.any():
        raise ValueError("Batch has no primary SSL target.")
    if not 0.0 <= residual_weight <= 1.0:
        raise ValueError("residual_weight must lie in [0,1].")
    return masked_teacher_loss(
        pred.float(), teacher.detach().float(), mean.detach().float(),
        primary_target, kind, residual_weight=residual_weight,
    )


def aggregate_cycle_object_loss(predictions: Sequence[torch.Tensor], teacher,
                                mean, primary_target, kind,
                                cycle_weights=None,
                                residual_weight: float = 0.0):
    """Combine pass losses; D defaults to 0.25*pass1 + 0.75*pass2."""
    if not predictions:
        raise ValueError("At least one prediction pass is required.")
    if cycle_weights is None:
        cycle_weights = (1.0,) if len(predictions) == 1 else (0.25, 0.75)
    if len(predictions) != len(cycle_weights):
        raise ValueError("Cycle weights must match the number of predictions.")
    weights = torch.as_tensor(cycle_weights, dtype=predictions[0].dtype,
                              device=predictions[0].device)
    if (weights < 0).any() or not torch.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("Cycle weights must be finite, nonnegative and nonzero.")
    weights = weights / weights.sum()
    losses = torch.stack([
        object_ssl_loss(p, teacher, mean, primary_target, kind, residual_weight)
        for p in predictions
    ])
    return (weights * losses).sum(), losses.detach()


def rank_ssl_loss(pred, candidates, mean, candidate_positive, candidate_valid,
                  temperature: float = 0.07,
                  residual_weight: float = 0.0) -> torch.Tensor:
    """Candidate-ranking auxiliary on one masked query per row.

    ``pred`` is [Q,S,1152], candidates [Q,K,S,1152]. Candidate vectors are
    detached inside ``conditional_scores`` and must not be supplied to the
    corrupted student graph.
    """
    scores = conditional_scores(
        pred, candidates, mean, candidate_valid, residual_weight,
    )
    return candidate_rank_loss(
        scores, candidate_positive, candidate_valid, temperature,
    ).mean()


def activity_loss(logits, labels) -> torch.Tensor:
    """One BCE term per construct; no occurrence-level reweighting."""
    if logits.ndim != 1 or labels.shape != logits.shape:
        raise ValueError("Activity logits and labels must be [batch].")
    labels = labels.to(device=logits.device, dtype=logits.dtype)
    if not torch.isfinite(logits).all() or not torch.isfinite(labels).all():
        raise ValueError("Nonfinite activity input.")
    if not ((labels == 0) | (labels == 1)).all():
        raise ValueError("Activity labels must be binary.")
    return F.binary_cross_entropy_with_logits(logits.float(), labels.float())


def _training_mode(bundle):
    """Train enabled modules; deterministic frozen subtrees (especially probes)."""
    bundle.train()
    for module in bundle.modules():
        parameters = list(module.parameters())
        if parameters and not any(p.requires_grad for p in parameters):
            module.eval()


def _cycle_weights(count, supplied, device):
    values = supplied if supplied is not None else ((1.0,) if count == 1 else (0.25, 0.75))
    w = torch.as_tensor(values, device=device, dtype=torch.float32)
    if w.shape != (count,) or not torch.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("Invalid cycle weights.")
    return w / w.sum()


def current_rank_loss(prediction, batch, primary_target, rank, residual_weight):
    """Gather queries from THIS forward pass; equal relation/kind/assembly means."""
    if 'query_pred' in rank:
        raise ValueError('Pass query indices; external query predictions can be stale/disconnected.')
    rows, tokens, slots = rank['query_rows'], rank['token_indices'], rank['slot_indices']
    if tokens.ndim != 2 or tokens.shape != slots.shape or rows.shape != tokens.shape[:1]:
        raise ValueError('Expected rows [Q], token_indices/slot_indices [Q,S].')
    if not rows.numel() or tokens.shape[1] not in (1, 2):
        raise ValueError('Each nonempty query panel has one or two target slots.')
    for indices, bound in ((rows, prediction.shape[0]), (tokens, prediction.shape[1]), (slots, 3)):
        if indices.dtype != torch.long or not ((indices >= 0) & (indices < bound)).all():
            raise ValueError('Query indices must be in-range int64 tensors.')
    if not (tokens == tokens[:, :1]).all():
        raise ValueError('A query scores one relation, optionally its two terminal slots.')
    if slots.shape[1] == 2 and (slots[:, 0] == slots[:, 1]).any():
        raise ValueError('Repeated slot in terminal pair.')
    identities = torch.stack([rows, tokens[:, 0]], -1)
    if identities.unique(dim=0).shape[0] != rows.shape[0]:
        raise ValueError('Duplicate query relation changes weighting.')
    if not primary_target[rows[:, None], tokens, slots].all():
        raise ValueError('Rank targets must be primary SSL targets.')
    gathered = prediction[rows[:, None], tokens, slots]
    scores = conditional_scores(gathered, rank['candidates'], rank['mean'], rank['valid'], residual_weight)
    losses = candidate_rank_loss(scores, rank['positive'], rank['valid'], rank.get('temperature', .07))
    kinds = batch.kind[rows, tokens[:, 0]]
    assembly_losses = []
    for row in rows.unique():
        groups = [losses[(rows == row) & (kinds == k)].mean()
                  for k in (DOMAIN, COVALENT, BREAK) if ((rows == row) & (kinds == k)).any()]
        assembly_losses.append(torch.stack(groups).mean())
    return torch.stack(assembly_losses).mean()


def ssl_step(bundle, batch: Batch, teacher, mean, primary_target, *,
             optimizer, rank=None, span_loss=None, rank_weight=0.05,
             span_weight=0.0, residual_weight=0.0,
             cycle_weights=None, raw_features=None,
             raw_valid=None, grad_clip=1.0):
    """One update. Rank uses current outputs. span_loss, if enabled, is a callback.

    Call raw_features as a zero-argument factory to run a trainable residue
    branch after training mode is configured. The factory returns (features,
    valid); precomputed tensors are only suitable for frozen/external features.
    span_loss(bundle) must run its own independent corrupted span view and
    return an assembly-normalized scalar, including D pass weighting if used.
    This reference update does not implement accumulation, scheduling or AMP.
    """
    from neuralps_v2.neuralps_reference import MASKED
    if not primary_target.any() or not (batch.state[primary_target] == MASKED).all():
        raise ValueError('Primary SSL targets must be explicitly masked.')
    if not ((primary_target.any(-1)) <= batch.valid).all():
        raise ValueError('Padding cannot carry primary targets.')
    if not primary_target.flatten(1).any(-1).all():
        raise ValueError('Each batch row must contain a primary target.')
    _training_mode(bundle)
    optimizer.zero_grad(set_to_none=True)
    if callable(raw_features):
        raw_features, raw_valid = raw_features()
    output = bundle.predict_masked(batch, raw_features=raw_features, raw_valid=raw_valid)
    predictions = output['cycle_teacher_predictions']
    object_loss, pass_losses = aggregate_cycle_object_loss(
        predictions, teacher, mean, primary_target, batch.kind,
        cycle_weights=cycle_weights, residual_weight=residual_weight)
    total = object_loss
    rank_value = object_loss.new_zeros(())
    if rank is not None and rank_weight:
        w = _cycle_weights(len(predictions), cycle_weights, object_loss.device)
        rank_value = (w * torch.stack([
            current_rank_loss(pred, batch, primary_target, rank, residual_weight)
            for pred in predictions])).sum()
        total = total + float(rank_weight) * rank_value
    span_value = object_loss.new_zeros(())
    if span_loss is not None and span_weight:
        if not callable(span_loss):
            raise ValueError('span_loss must be a callback executing the current corrupted span view.')
        span_value = span_loss(bundle).float()
        if span_value.ndim != 0 or not torch.isfinite(span_value) or not span_value.requires_grad:
            raise ValueError('span_loss callback must return a finite connected scalar.')
        total = total + float(span_weight) * span_value
    if not torch.isfinite(total):
        raise FloatingPointError('Nonfinite SSL loss before backward.')
    total.backward()
    norm = torch.nn.utils.clip_grad_norm_(bundle.parameters(), grad_clip, error_if_nonfinite=True)
    optimizer.step()
    return dict(loss=total.detach(), object_loss=object_loss.detach(), rank_loss=rank_value.detach(),
                span_loss=span_value.detach(), pass_losses=pass_losses,
                grad_norm=torch.as_tensor(norm).detach())


def activity_step(bundle, batch: Batch, labels, *, optimizer,
                  raw_features=None, raw_valid=None, grad_clip=1.0, cycle_weights=None):
    """One supervised update; D trains both passes and deploys the last pass."""
    _training_mode(bundle)
    optimizer.zero_grad(set_to_none=True)
    if callable(raw_features):
        raw_features, raw_valid = raw_features()
    output = bundle(batch, raw_features=raw_features, raw_valid=raw_valid)
    logits = output['cycle_activity_logits']
    w = _cycle_weights(len(logits), cycle_weights, logits[-1].device)
    pass_losses = torch.stack([activity_loss(z, labels) for z in logits])
    loss = (w * pass_losses).sum()
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(bundle.parameters(), grad_clip, error_if_nonfinite=True)
    optimizer.step()
    return dict(loss=loss.detach(), grad_norm=torch.as_tensor(norm).detach())


def mean_metrics(records: Sequence[Mapping[str, torch.Tensor]], weights=None):
    """Weighted update metrics; preserve pass-loss axis. Not a gradient accumulator."""
    if not records:
        raise ValueError('Cannot average an empty metric list.')
    keys = tuple(records[0])
    if any(tuple(r) != keys for r in records):
        raise ValueError('Metric record keys differ.')
    w = torch.as_tensor(weights if weights is not None else [1.] * len(records), dtype=torch.float32)
    if w.shape != (len(records),) or not torch.isfinite(w).all() or (w <= 0).any():
        raise ValueError('Invalid metric weights.')
    result = {}
    for k in keys:
        values = torch.stack([torch.as_tensor(r[k]).detach().float().cpu() for r in records])
        result[k] = (values * w.reshape((-1,) + (1,) * (values.ndim - 1))).sum(0) / w.sum()
    return result
