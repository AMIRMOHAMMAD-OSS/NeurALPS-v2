"""Optional A-R/C-R extension. Requires raw window FASTAs + residue/region masks.

This is a raw-AA CNN, NOT recovered ESM-C residue embeddings. Corrupt raw inputs
and close pooled-cache dependencies before calling. Integrate pooled vectors at
the sequence-slot level as specified in the report; independent termini stay
independent CNN examples. Never concatenate across a physical protein break.
"""
import torch
from torch import nn
from torch.nn import functional as F

ALPHABET = 'ACDEFGHIKLMNPQRSTVWYBXZOU'  # 25 symbols
PAD, MASK, UNK = 0, 1, 2
AA_TO_ID = {aa: i + 3 for i, aa in enumerate(ALPHABET)}


class ResidueBlock(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.norm = nn.LayerNorm(128)
        self.conv1 = nn.Conv1d(128,128,5,padding=2*dilation,dilation=dilation)
        self.conv2 = nn.Conv1d(128,128,5,padding=2*dilation,dilation=dilation)
        self.dropout = nn.Dropout(.1)

    def forward(self, x, valid):
        z = (self.norm(x)*valid[...,None]).transpose(1,2)
        z = self.conv1(z).transpose(1,2)
        z = self.dropout(F.gelu(z))*valid[...,None]
        z = self.dropout(self.conv2(z.transpose(1,2)).transpose(1,2))
        return (x+z)*valid[...,None]


class ResidueWindowEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.aa = nn.Embedding(28,32,padding_idx=PAD)
        self.region = nn.Embedding(3,32)  # left flank, connection/core, right flank
        self.input = nn.Linear(32,128)
        self.blocks = nn.ModuleList([ResidueBlock(d) for d in (1,2,4,8)])
        self.norm = nn.LayerNorm(128)
        self.attention = nn.Sequential(nn.Linear(128,64), nn.Tanh(), nn.Linear(64,1))
        self.pool = nn.Sequential(nn.Linear(3*128+3,256), nn.GELU(), nn.LayerNorm(256))

    def forward(self, ids, region, valid):
        # ids/region/valid: [unique windows in microbatch, padded residues]
        assert ids.shape == region.shape == valid.shape
        assert valid.any(-1).all(), 'Empty biological termini use EMPTY; do not run CNN on them.'
        h = self.input(self.aa(ids)+self.region(region))*valid[...,None]
        for block in self.blocks:
            h = block(h,valid)
        h = self.norm(h)*valid[...,None]
        score = self.attention(h).squeeze(-1).float()
        vectors, present = [], []
        for i in range(3):
            mask = valid & (region == i)
            weight = score.masked_fill(~mask,-1e4).softmax(-1)*mask
            weight = weight/weight.sum(-1,keepdim=True).clamp_min(1e-6)
            vectors.append((h*weight.to(h.dtype)[...,None]).sum(1))
            present.append(mask.any(-1).to(h.dtype))
        pooled = self.pool(torch.cat(vectors+[torch.stack(present,-1)],-1))
        return h, pooled


class SpanDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.context = nn.Linear(256,256)  # gamma/beta, each 128
        self.norm = nn.LayerNorm(128)
        self.out = nn.Linear(128,25)

    def forward(self, residue_states, object_context):
        gamma,beta = self.context(object_context).chunk(2,-1)
        h = (1+.1*gamma.tanh()[:,None])*residue_states+beta[:,None]
        return self.out(self.norm(h))  # CE class is original AA token ID minus 3
