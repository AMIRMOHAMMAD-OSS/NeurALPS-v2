"""Stage/optimizer/checkpoint helpers for the phase2.2 reference.

The caller supplies a DataLoader whose records already satisfy the physical
graph and provenance contracts. This module intentionally does not guess file
paths, split labels, or target masks.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import torch

from neuralps_v2.conditional_heads import FoundationBundle, initialize_child
from neuralps_v2.neuralps_reference import NeurALPS
from neuralps_v2.training_objectives import activity_step, mean_metrics, ssl_step


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _set_requires_grad(model, enabled=False):
    for parameter in model.parameters():
        parameter.requires_grad_(enabled)


def _enable_named(model, prefixes):
    for name, parameter in model.named_parameters():
        if any(name == p or name.startswith(p + ".") for p in prefixes):
            parameter.requires_grad_(True)


def configure_stage(bundle: FoundationBundle, stage: str):
    """Freeze/unfreeze the named phase2.2 stage and return parameter names.

    ``residue_encoder`` is optional: attach it to the bundle when raw windows
    are trained online. The core reference accepts externally computed raw
    features, so a loader may instead train that encoder in its own optimizer.
    """
    expected = 'A' if stage.startswith('A_') else 'C' if stage.startswith('C_') else 'D' if stage.startswith('D_') else None
    if stage.startswith('activity_restricted_'):
        expected = stage.rsplit('_', 1)[-1]
    if expected is not None and bundle.encoder.variant != expected:
        raise ValueError('Stage/architecture mismatch.')
    if stage == 'A_pool' and bundle.encoder.input.with_residue:
        raise ValueError('A_pool stage requires pooled-only inputs.')
    if 'residue' in stage or 'raw' in stage:
        if not hasattr(bundle, 'residue_encoder') or not bundle.encoder.input.with_residue:
            raise ValueError('Residue stage requires an attached online residue_encoder and enabled input gate.')
    _set_requires_grad(bundle, False)
    if stage == "A_pool":
        _enable_named(bundle, ("encoder.input", "encoder.local", "encoder.output_norm", "decoder"))
    elif stage in ("A_raw_warm", "A_residue_warm"):
        _enable_named(bundle, ("residue_encoder", "encoder.input.raw_gate", "decoder"))
    elif stage in ("A_raw_joint", "A_residue_joint"):
        _enable_named(bundle, ("encoder", "decoder", "residue_encoder"))
    elif stage == "C_warm":
        _enable_named(bundle, ("encoder.global_blocks", "encoder.global_slots", "decoder"))
    elif stage == "C_joint":
        _enable_named(bundle, ("encoder", "decoder", "residue_encoder"))
    elif stage == "D_warm":
        _enable_named(bundle, ("encoder.recycle_h", "encoder.recycle_g", "decoder"))
    elif stage == "D_joint":
        _enable_named(bundle, ("encoder", "decoder", "residue_encoder"))
    elif stage == "activity_probe":
        _enable_named(bundle, ("activity_head",))
    elif stage == "activity_restricted_A":
        _enable_named(bundle, ("activity_head", "encoder.local.2", "encoder.local.3"))
    elif stage == "activity_restricted_C":
        _enable_named(bundle, ("activity_head", "encoder.global_blocks"))
    elif stage == "activity_restricted_D":
        _enable_named(bundle, ("activity_head", "encoder.global_blocks", "encoder.recycle_h", "encoder.recycle_g"))
    elif stage == "activity_broad":
        _enable_named(bundle, ("encoder", "activity_head", "residue_encoder"))
    else:
        raise ValueError(f"Unknown phase2.2 stage: {stage}")
    if not stage.startswith('activity_') and hasattr(bundle, 'span_decoder'):
        _enable_named(bundle, ('span_decoder',))
    names = [name for name,p in bundle.named_parameters() if p.requires_grad]
    if not names:
        raise RuntimeError(f"Stage {stage} enabled no parameters.")
    return names


def build_optimizer(bundle, stage: str, *, lr: float, weight_decay=0.01,
                    betas=(0.9, 0.999), eps=1e-8, lr_by_prefix=None):
    """Longest prefix wins. Example: {'encoder':1e-5,'activity_head':1e-4}.

    Every override must match an enabled parameter, preventing misspelled
    branch names from silently using an inherited learning rate.
    """
    names = configure_stage(bundle, stage)
    overrides = dict(lr_by_prefix or {})
    if lr <= 0 or weight_decay < 0 or any(v <= 0 for v in overrides.values()):
        raise ValueError('Invalid optimizer rates/decay.')
    matches = {prefix: False for prefix in overrides}
    grouped = {}
    for name, parameter in bundle.named_parameters():
        if not parameter.requires_grad:
            continue
        candidates = [p for p in overrides if name == p or name.startswith(p + '.')]
        for prefix in candidates:
            matches[prefix] = True
        rate = overrides[max(candidates, key=len)] if candidates else lr
        decay = 0.0 if parameter.ndim < 2 else weight_decay
        grouped.setdefault((rate, decay), []).append(parameter)
    if not all(matches.values()):
        raise ValueError('Unmatched trainable prefixes: ' + str([p for p,v in matches.items() if not v]))
    groups = [dict(params=parameters, lr=rate, weight_decay=decay)
              for (rate,decay),parameters in grouped.items()]
    return torch.optim.AdamW(groups, betas=betas, eps=eps), names


def copy_variant(parent: FoundationBundle, child: FoundationBundle):
    """Transfer the complete shared bundle, including decoder and residue pool."""
    transition = parent.encoder.variant, child.encoder.variant
    allowed = {('A','C'): ('encoder.global_blocks.', 'encoder.global_slots'),
               ('C','D'): ('encoder.recycle_h.', 'encoder.recycle_g.')}
    if transition not in allowed or parent.activity_head_kind != child.activity_head_kind:
        raise ValueError('Expected compatible A->C or C->D bundles.')
    src, dst = parent.state_dict(), child.state_dict()
    if any(k not in dst or dst[k].shape != v.shape for k,v in src.items()):
        raise ValueError('Parent/child branch configuration mismatch.')
    missing = sorted(set(dst) - set(src))
    if not all(k.startswith(allowed[transition]) for k in missing):
        raise ValueError('Unexpected child keys: ' + str(missing))
    child.load_state_dict(src, strict=False)
    return missing


def build_bundle(config, variant='A', input_variant='pool', activity_head_kind='pooled'):
    """Build only settings supported by this explicit reference implementation."""
    from neuralps_v2.phase2_preflight import validate_config
    validate_config(config)
    if config['design_version'] != 'phase2.2':
        raise ValueError('Expected phase2.2 config.')
    local = config['local']
    if (local['width'], local['heads'], local['blocks'], local['radius_interleaved'], local['dropout']) != (256,8,4,6,0.1):
        raise ValueError('Reference uses width=256, heads=8, blocks=4, radius=6, dropout=.1.')
    if input_variant not in ('pool', 'residue', 'raw'):
        raise ValueError('Unknown input variant.')
    bundle = FoundationBundle(variant, config['inputs']['domain_types'],
                              input_variant != 'pool', activity_head_kind)
    if input_variant == 'residue':
        from neuralps_v2.pretrained_residue import PretrainedResidueEncoder
        bundle.residue_encoder = PretrainedResidueEncoder()
    elif input_variant == 'raw':
        from neuralps_v2.residue_extension import ResidueWindowEncoder
        bundle.residue_encoder = ResidueWindowEncoder()
    if input_variant != 'pool':
        from neuralps_v2.residue_extension import SpanDecoder
        bundle.span_decoder = SpanDecoder()
    return bundle


def initialize_residue_branch(pooled, residue):
    """Load learned pooled A into A_residue/A_raw, leaving only new branches fresh."""
    if pooled.encoder.variant != 'A' or residue.encoder.variant != 'A':
        raise ValueError('Initial residue extension is A_pool -> A_residue/A_raw.')
    if pooled.encoder.input.with_residue or not residue.encoder.input.with_residue:
        raise ValueError('Expected pooled source and residue-enabled destination.')
    if not hasattr(residue, 'residue_encoder') or pooled.activity_head_kind != residue.activity_head_kind:
        raise ValueError('Expected compatible head and attached residue encoder.')
    src, dst = pooled.state_dict(), residue.state_dict()
    if any(k not in dst or dst[k].shape != v.shape for k,v in src.items()):
        raise ValueError('Shared configuration mismatch.')
    missing = sorted(set(dst) - set(src))
    if not all(k.startswith(('encoder.input.raw_gate', 'residue_encoder.', 'span_decoder.')) for k in missing):
        raise ValueError('Unexpected new residue-branch keys.')
    residue.load_state_dict(src, strict=False)
    return missing


def _state_dict_cpu(model):
    return {k: v.detach().cpu() for k,v in model.state_dict().items()}


def save_checkpoint(path, *, bundle, optimizer=None, scheduler=None,
                    config, stage, epoch, update, metadata):
    """Save a self-describing checkpoint; config/metadata are JSON-safe."""
    json.dumps(config); json.dumps(metadata)
    payload = {
        "format": "neuralps_phase2.2",
        "architecture": bundle.encoder.variant,
        "activity_head_kind": bundle.activity_head_kind,
        "config_fingerprint": config_fingerprint(config),
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
        "stage": stage,
        "epoch": int(epoch),
        "update": int(update),
        "model": _state_dict_cpu(bundle),
        "config": config,
        "metadata": metadata,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


def load_checkpoint(path, bundle, *, optimizer=None, scheduler=None,
                    expected_architecture=None, map_location="cpu", restore_rng=False):
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get('format') != 'neuralps_phase2.2':
        raise ValueError('Checkpoint format mismatch; do not silently migrate old checkpoints.')
    if payload.get('config_fingerprint') != config_fingerprint(payload['config']):
        raise ValueError('Checkpoint configuration fingerprint mismatch.')
    if payload.get('activity_head_kind') != bundle.activity_head_kind:
        raise ValueError('Activity head mismatch.')
    architecture = payload.get("architecture")
    if expected_architecture is not None and architecture != expected_architecture:
        raise ValueError(f"Checkpoint architecture {architecture!r} != {expected_architecture!r}.")
    if architecture != bundle.encoder.variant:
        raise ValueError("Checkpoint/model architecture mismatch.")
    bundle.load_state_dict(payload["model"], strict=True)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    if restore_rng:
        rng = payload['rng']
        random.setstate(rng['python']); np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch'].cpu())
        if rng['cuda'] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([v.cpu() for v in rng['cuda']])
    return payload


def run_ssl_epoch(bundle, loader: Iterable[Mapping], *, optimizer,
                  rank_weight=0.05, span_weight=0.0,
                  residual_weight=0.0, cycle_weights=None,
                  grad_clip=1.0):
    records=[]; sizes=[]
    for record in loader:
        sizes.append(int(record["batch"].valid.shape[0]))
        records.append(ssl_step(
            bundle, record["batch"], record["teacher"], record["mean"],
            record["primary_target"], optimizer=optimizer,
            rank=record.get("rank"), span_loss=record.get("span_loss"),
            rank_weight=rank_weight, span_weight=span_weight,
            residual_weight=residual_weight, cycle_weights=cycle_weights,
            raw_features=record.get("raw_features"),
            raw_valid=record.get("raw_valid"), grad_clip=grad_clip,
        ))
    return mean_metrics(records, sizes)


def run_activity_epoch(bundle, loader: Iterable[Mapping], *, optimizer,
                        grad_clip=1.0):
    records=[]; sizes=[]
    for record in loader:
        sizes.append(int(record["batch"].valid.shape[0]))
        records.append(activity_step(
            bundle, record["batch"], record["labels"], optimizer=optimizer,
            raw_features=record.get("raw_features"),
            raw_valid=record.get("raw_valid"), grad_clip=grad_clip,
        ))
    return mean_metrics(records, sizes)


def config_fingerprint(config):
    blob = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()
