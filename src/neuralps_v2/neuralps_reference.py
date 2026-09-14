"""NeurALPS phase2.2 reference: pooled-cache A, C, D encoders.

PyTorch 2.8; deliberately explicit dense attention for correctness, not a
production data loader. No module IDs, donor IDs, labels, or B/J annotation enter
the encoder. Provenance-based corruption must happen BEFORE InputEncoder.
See docs/architecture.md for training and evaluation contracts.
Use masking_contract.py for production provenance closure. This module is a
readable model reference, not a production store or complete training runner.
"""
from dataclasses import dataclass
from typing import Sequence
import math
import torch
from torch import nn
from torch.nn import functional as F

DOMAIN, COVALENT, BREAK = 0, 1, 2
NA, OBSERVED, EMPTY, MISSING, MASKED = range(5)


@dataclass
class Batch:
    x: torch.Tensor             # [batch, tokens, 3, 1152]; slots = seq, C-term, N-term
    state: torch.Tensor         # [batch, tokens, 3], enums above
    kind: torch.Tensor          # [batch, tokens], DOMAIN/COVALENT/BREAK
    domain_type: torch.Tensor   # [batch, tokens], canonical vocabulary, 0 = unknown
    pos: torch.Tensor           # [batch, tokens], D_i=2i, connection_i=2i+1
    chain: torch.Tensor         # [batch, tokens], arbitrary same-chain IDs; BREAK=-1
    meta: torch.Tensor          # [batch, tokens, 6], documented numeric features
    valid: torch.Tensor         # [batch, tokens], bool; right padding

    def validate(self):
        b, n, slots, dim = self.x.shape
        assert slots == 3 and dim == 1152
        assert self.state.shape == (b, n, 3)
        assert self.meta.shape == (b, n, 6)
        for v in (self.kind, self.domain_type, self.pos, self.chain, self.valid):
            assert v.shape == (b, n)
        assert self.valid.dtype == torch.bool and self.valid.any(1).all()
        assert ((self.kind[self.valid] >= 0) & (self.kind[self.valid] <= 2)).all()
        assert ((self.state >= NA) & (self.state <= MASKED)).all()
        is_d = (self.kind == DOMAIN) & self.valid
        is_c = (self.kind == COVALENT) & self.valid
        is_b = (self.kind == BREAK) & self.valid
        assert (self.state[..., 1:][is_d | is_c] == NA).all()
        assert (self.state[..., 0][is_b] == NA).all()
        assert (self.state[..., 1:][is_b] != NA).all()
        assert (self.state[..., 0][is_d | is_c] != NA).all()
        assert (self.chain[is_b] == -1).all()
        assert torch.isfinite(self.x[self.state == OBSERVED]).all()
        assert torch.isfinite(self.meta).all()


class FFN(nn.Module):
    def __init__(self, width=256, hidden=1024, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, width),
                                 nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class InputEncoder(nn.Module):
    def __init__(self, domain_types=32, width=256, with_residue=False):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(1152), nn.Linear(1152, width),
                                     nn.LayerNorm(width))
        self.state_embed = nn.Embedding(5, width)
        self.slot_embed = nn.Embedding(3, width)
        self.kind_embed = nn.Embedding(3, width)
        self.domain_embed = nn.Embedding(domain_types, width)
        self.fuse = nn.Sequential(nn.Linear(3 * width, width), nn.GELU(),
                                  nn.Linear(width, width))
        self.metadata = nn.Sequential(nn.Linear(6, 32), nn.GELU(), nn.Linear(32, width))
        self.norm = nn.LayerNorm(width)
        self.with_residue = with_residue
        if with_residue:
            self.raw_gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, batch, raw_features=None, raw_valid=None):
        state = torch.where(batch.valid[..., None], batch.state, NA)
        kind = torch.where(batch.valid, batch.kind, DOMAIN)
        domain_type = torch.where(batch.valid & (kind == DOMAIN), batch.domain_type, 0)
        observed = (state == OBSERVED) & batch.valid[..., None]
        # Before projection: a hidden input may even contain NaN sentinels.
        clean = torch.where(observed[..., None], batch.x, torch.zeros_like(batch.x))
        v = self.project(clean) * observed[..., None]
        if self.with_residue:
            if raw_features is None or raw_valid is None:
                raise ValueError('Residue-enabled encoder needs corrupted raw features and availability.')
            assert raw_features.shape == v.shape and raw_valid.shape == observed.shape
            raw_valid = raw_valid & batch.valid[..., None]
            assert not (raw_valid & ((state == NA) | (state == EMPTY))).any()
            v = v + 0.5*self.raw_gate.sigmoid()*torch.where(raw_valid[...,None], raw_features, 0)
        elif raw_features is not None:
            raise ValueError('Enable with_residue=True to use raw features.')
        slot_ids = torch.arange(3, device=v.device)
        v = v + self.state_embed(state) + self.slot_embed(slot_ids)
        v = v * (state != NA)[..., None]
        edge = self.fuse(v.flatten(-2))
        domain = v[..., 0, :] + self.domain_embed(domain_type)
        h = torch.where((kind == DOMAIN)[..., None], domain, edge)
        # Masked sequence-derived metadata cannot bypass feature masking.
        meta = batch.meta.masked_fill((state == MASKED).any(-1)[..., None], 0)
        meta = torch.where(batch.valid[..., None], meta, torch.zeros_like(meta))
        h = self.norm(h + self.kind_embed(kind) + self.metadata(meta))
        return h * batch.valid[..., None]


class Attention(nn.Module):
    def __init__(self, width=256, heads=8, dropout=0.1):
        super().__init__()
        assert width % heads == 0
        self.heads, self.head_dim, self.dropout = heads, width // heads, dropout
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.out = nn.Linear(width, width)

    def forward(self, query, context, allow, bias=None):
        b, nq, width = query.shape
        nk = context.shape[1]
        def split(x, n):
            return x.reshape(b, n, self.heads, self.head_dim).transpose(1, 2)
        q, k, v = split(self.q(query), nq), split(self.k(context), nk), split(self.v(context), nk)
        logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(self.head_dim)
        if bias is not None:
            logits = logits + bias.float()
        # Every row must allow at least one key. Padding-query output is zeroed by caller.
        logits = logits.masked_fill(~allow[:, None], float('-inf'))
        weights = torch.softmax(logits, -1)
        weights = F.dropout(weights, self.dropout, self.training).to(v.dtype)
        out = torch.matmul(weights, v).transpose(1, 2).reshape(b, nq, width)
        return self.out(out)


class LocalBlock(nn.Module):
    """Four default blocks, each attending within +/-6 interleaved positions."""
    def __init__(self, width=256, heads=8, dropout=0.1, radius=6):
        super().__init__()
        self.radius = radius
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attention = Attention(width, heads, dropout)
        self.ffn = FFN(width, 4 * width, dropout)
        self.relative = nn.Embedding(65, heads)  # clipped signed offset -32..32
        self.physical = nn.Embedding(3, heads)   # same chain, different chain, break endpoint
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, batch):
        n = h.shape[1]
        distance = batch.pos[:, None, :] - batch.pos[:, :, None]  # key minus query
        valid_pair = batch.valid[:, :, None] & batch.valid[:, None, :]
        allow = valid_pair & (distance.abs() <= self.radius)
        eye = torch.eye(n, device=h.device, dtype=torch.bool)[None]
        allow = allow | (eye & ~batch.valid[:, :, None])
        ci, cj = batch.chain[:, :, None], batch.chain[:, None, :]
        relation = (ci != cj).long()
        relation = torch.where((ci < 0) | (cj < 0), 2, relation)
        bias = self.relative(distance.clamp(-32, 32) + 32) + self.physical(relation)
        bias = bias.permute(0, 3, 1, 2)
        z = self.norm1(h)
        h = h + self.dropout(self.attention(z, z, allow, bias))
        h = h + self.ffn(self.norm2(h))
        return h * batch.valid[..., None]


class GlobalExchange(nn.Module):
    """Four latent assembly slots read all tokens and write back to every token."""
    def __init__(self, width=256, heads=8, dropout=0.1):
        super().__init__()
        self.gq, self.hk, self.gf = [nn.LayerNorm(width) for _ in range(3)]
        self.hq, self.gk, self.hf = [nn.LayerNorm(width) for _ in range(3)]
        self.read = Attention(width, heads, dropout)
        self.write = Attention(width, heads, dropout)
        self.ffg, self.ffh = FFN(width, 4 * width, dropout), FFN(width, 4 * width, dropout)
        # Exact A -> C initialization in eval mode after copying shared weights.
        # The read path begins receiving gradients after outward paths open.
        nn.init.zeros_(self.write.out.weight)
        nn.init.zeros_(self.write.out.bias)
        nn.init.zeros_(self.ffh.net[3].weight)
        nn.init.zeros_(self.ffh.net[3].bias)
        self.gate = nn.Parameter(torch.tensor(-2.0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, g, valid):
        b, n, _ = h.shape
        ng = g.shape[1]
        read_allow = valid[:, None, :].expand(b, ng, n)
        g = g + self.dropout(self.read(self.gq(g), self.hk(h), read_allow))
        g = g + self.ffg(self.gf(g))
        write_allow = torch.ones(b, n, ng, device=h.device, dtype=torch.bool)
        delta = self.dropout(self.write(self.hq(h), self.gk(g), write_allow))
        candidate = h + delta
        candidate = candidate + self.ffh(self.hf(candidate))
        h = h + 0.5 * self.gate.sigmoid() * (candidate - h)
        return h * valid[..., None], g


class RecycleAdapter(nn.Module):
    def __init__(self, width=256):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 64),
                                 nn.GELU(), nn.Linear(64, width))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, h):
        return 0.5 * self.gate.sigmoid() * self.net(h.detach())


class NeurALPS(nn.Module):
    """variants: input (no context control), A (local), C (assembly), D (recycled)."""
    def __init__(self, variant='A', domain_types=32, width=256, dropout=0.1, with_residue=False):
        super().__init__()
        if variant not in ('input', 'A', 'C', 'D'):
            raise ValueError(variant)
        self.variant = variant
        self.input = InputEncoder(domain_types, width, with_residue)
        self.local = nn.ModuleList([LocalBlock(width, 8, dropout) for _ in range(4)]
                                   if variant != 'input' else [])
        self.global_blocks = nn.ModuleList([GlobalExchange(width, 8, dropout) for _ in range(2)]
                                           if variant in ('C', 'D') else [])
        if variant in ('C', 'D'):
            self.global_slots = nn.Parameter(torch.randn(4, width) * 0.02)
        if variant == 'D':
            self.recycle_h, self.recycle_g = RecycleAdapter(width), RecycleAdapter(width)
        self.output_norm = nn.LayerNorm(width)

    def forward(self, batch, cycles=None, raw_features=None, raw_valid=None):
        base = self.input(batch, raw_features, raw_valid)
        if cycles is None:
            cycles = 2 if self.variant == 'D' else 1
        if cycles < 1 or cycles > 3 or (self.variant != 'D' and cycles != 1):
            raise ValueError('Only D can recycle; 1..3 passes (3 is diagnostic only).')
        base_g = self.global_slots[None].expand(base.shape[0], -1, -1) if self.global_blocks else None
        states = []
        previous_h = previous_g = None
        for cycle in range(cycles):
            h = base if cycle == 0 else base + self.recycle_h(previous_h)
            g = base_g if cycle == 0 else base_g + self.recycle_g(previous_g)
            h = h * batch.valid[..., None]
            for layer in self.local:
                h = layer(h, batch)
            for layer in self.global_blocks:
                h, g = layer(h, g, batch.valid)
            h = self.output_norm(h) * batch.valid[..., None]
            states.append(h)
            previous_h, previous_g = h, g
        return {'tokens': states[-1], 'cycles': states, 'global': previous_g}


class MaskDecoder(nn.Module):
    """Training-only teacher decoder; shared across object/terminal roles."""
    def __init__(self, width=256):
        super().__init__()
        self.slot = nn.Embedding(3, 16)
        self.kind = nn.Embedding(3, 16)
        self.net = nn.Sequential(nn.LayerNorm(width + 32), nn.Linear(width + 32, 512),
                                 nn.GELU(), nn.Linear(512, 1152))

    def forward(self, h, kind):
        b, n, _ = h.shape
        roles = self.slot(torch.arange(3, device=h.device))[None, None].expand(b, n, -1, -1)
        kinds = self.kind(kind)[..., None, :].expand(-1, -1, 3, -1)
        return self.net(torch.cat([h[..., None, :].expand(-1, -1, 3, -1), roles, kinds], -1))


def masked_teacher_loss(pred, teacher, mean, target, kind, residual_weight=0.0):
    """FP32. Equal D/covalent/break group means within each assembly, then assemblies.

    target marks primary supervised slots ONLY, not every slot removed by closure.
    mean is fixed, training-only type-conditioned cache mean, broadcastable to teacher.
    """
    pred, teacher, mean = pred.float(), teacher.detach().float(), mean.detach().float()
    # Missing/non-target teacher slots may be NaN in a production loader.
    teacher = torch.where(target[..., None], teacher, 0)
    mean = torch.where(target[..., None], mean, 0)
    raw = 1 - F.cosine_similarity(pred, teacher, dim=-1, eps=1e-6)
    residual = 1 - F.cosine_similarity(pred - mean, teacher - mean, dim=-1, eps=1e-6)
    use_residual = (teacher - mean).norm(dim=-1) > 1e-6
    loss = torch.where(use_residual, (1 - residual_weight) * raw + residual_weight * residual, raw)
    per_assembly = []
    for i in range(pred.shape[0]):
        groups = []
        for group in (DOMAIN, COVALENT, BREAK):
            eligible = target[i] & (kind[i, :, None] == group)
            if eligible.any():
                # Equal primary relations; a break with two terminal targets is
                # not counted twice relative to one with only one available target.
                per_node = torch.where(eligible,loss[i],0).sum(-1)/eligible.sum(-1).clamp_min(1)
                groups.append(per_node[eligible.any(-1)].mean())
        if groups:
            per_assembly.append(torch.stack(groups).mean())
    if not per_assembly:
        raise ValueError('Batch has no eligible primary SSL targets.')
    return torch.stack(per_assembly).mean()


def activity_features(h, batch):
    d = (batch.kind == DOMAIN) & batch.valid
    e = (batch.kind != DOMAIN) & batch.valid
    def mean(mask):
        return (h * mask[..., None]).sum(1) / mask.sum(1).clamp_min(1)[:, None]
    maximum = h.masked_fill(~e[..., None], float('-inf')).max(1).values
    maximum = torch.where(e.any(1)[:, None], maximum, torch.zeros_like(maximum))
    counts = torch.stack([d.sum(1), e.sum(1)], -1).to(h.dtype).log1p()
    return torch.cat([mean(d), mean(e), maximum, counts], -1)


class ActivityHead(nn.Module):
    def __init__(self, width=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(3 * width + 2), nn.Linear(3 * width + 2, 128),
                                 nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, h, batch):
        return self.net(activity_features(h, batch)).squeeze(-1)  # logits for BCEWithLogitsLoss


def corrupt(batch, mask):
    """mask [b,n,3] from closure. EMPTY/NA are structural states, never teacher targets."""
    actual = mask & (batch.state == OBSERVED)
    state = batch.state.clone().masked_fill(actual, MASKED)
    values = batch.x.clone().masked_fill(actual[..., None], 0)
    meta = batch.meta.clone().masked_fill(actual.any(-1)[..., None], 0)
    return Batch(values, state, batch.kind, batch.domain_type, batch.pos, batch.chain, meta, batch.valid)


def purged_part_split(parts: Sequence[set[str]], heldout: set[str], eligible=None):
    """A construct is test if ANY part is held out. Its full set is excluded from train."""
    eligible = list(range(len(parts))) if eligible is None else list(eligible)
    test = [i for i in eligible if parts[i] & heldout]
    train = [i for i in eligible if not parts[i] & heldout]
    assert all(not (parts[i] & heldout) for i in train)
    return train, test
