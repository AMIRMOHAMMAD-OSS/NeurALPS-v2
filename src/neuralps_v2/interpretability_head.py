"""Optional additive activity readout, compared separately with the common MLP.

Factors explain this model's logit. They are not identified biological seam
probabilities. All factors can change when contextual encoder states change.
"""
import torch
from torch import nn
from neuralps_v2.neuralps_reference import DOMAIN


class AdditiveActivityHead(nn.Module):
    def __init__(self, width=256):
        super().__init__()
        self.domain = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 1))
        self.connection = nn.Sequential(nn.LayerNorm(3 * width), nn.Linear(3 * width, 128), nn.GELU(), nn.Linear(128, 1))
        self.counts = nn.Linear(2, 1)

    def components(self, h, batch):
        d = batch.valid & (batch.kind == DOMAIN)
        e = batch.valid & ~d
        left = torch.cat([torch.zeros_like(h[:, :1]), h[:, :-1]], 1)
        right = torch.cat([h[:, 1:], torch.zeros_like(h[:, :1])], 1)
        # The canonical graph validator guarantees a domain on each edge side.
        domain = self.domain(h).squeeze(-1).masked_fill(~d, 0)
        edge = self.connection(torch.cat([left, h, right], -1)).squeeze(-1).masked_fill(~e, 0)
        domain = domain / d.sum(1).clamp_min(1)[:, None]
        edge = edge / e.sum(1).clamp_min(1)[:, None]
        count = torch.stack([d.sum(1), e.sum(1)], -1).to(h.dtype).log1p()
        background = self.counts(count).squeeze(-1)
        logit = domain.sum(1) + edge.sum(1) + background
        return dict(logit=logit, domain_contribution=domain, connection_contribution=edge,
                    background_contribution=background)

    def forward(self, h, batch):
        return self.components(h, batch)['logit']
