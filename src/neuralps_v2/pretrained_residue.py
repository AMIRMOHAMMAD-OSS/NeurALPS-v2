"""Trainable regional pooling of genuinely extracted frozen ESM-C residue states.

No PLM extraction or corruption is performed here. Every retained residue state
must come from the current valid sequence view; masking vectors after clean
contextual extraction cannot satisfy the leakage contract.
"""
import torch
from torch import nn


class PretrainedResidueEncoder(nn.Module):
    def __init__(self, output_width=256):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(1152), nn.Linear(1152, 128), nn.GELU())
        self.attention = nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1))
        self.pool = nn.Sequential(nn.Linear(387, output_width), nn.GELU(), nn.LayerNorm(output_width))

    def forward(self, residue_states, region, valid):
        if residue_states.shape[:-1] != valid.shape or region.shape != valid.shape:
            raise ValueError('Expected residue states [windows,residues,1152] and matching masks.')
        if residue_states.ndim != 3 or residue_states.shape[-1] != 1152:
            raise ValueError('Expected ESM-C 1152-dimensional residue vectors.')
        if valid.dtype != torch.bool or not valid.any(-1).all():
            raise ValueError('Each encoded view needs at least one valid residue.')
        if not ((region[valid] >= 0) & (region[valid] <= 2)).all():
            raise ValueError('Region must be left/core/right = 0/1/2.')
        if not torch.isfinite(residue_states[valid]).all():
            raise ValueError('Nonfinite observed residue vector.')
        clean = torch.where(valid[..., None], residue_states.detach(), 0)
        h = self.project(clean) * valid[..., None]
        score = self.attention(h).squeeze(-1).float()
        vectors, present = [], []
        for k in range(3):
            mask = valid & (region == k)
            # Mask before softmax; empty regions get exactly zero weight.
            weights = score.masked_fill(~mask, -1e9).softmax(-1) * mask
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
            vectors.append((h * weights.to(h.dtype)[..., None]).sum(1))
            present.append(mask.any(-1).to(h.dtype))
        return h, self.pool(torch.cat(vectors + [torch.stack(present, -1)], -1))
