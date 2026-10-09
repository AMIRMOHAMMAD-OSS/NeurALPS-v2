"""Pooled-feature prototype for the proposed module-independent assembly encoder.

This is a downstream encoder, not an ESM-C replacement. The residue branch,
pretraining sampler and training loop are not implemented in this preparation release.
No module identities, B/J tags, dataset names, labels or donor IDs are model inputs.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F

NA, OBSERVED, EMPTY, MISSING, MASKED = range(5)


@dataclass
class Config:
    feature_dim: int = 1152
    width: int = 256
    heads: int = 8
    local_layers: int = 4
    radius: int = 6
    global_layers: int = 2
    global_slots: int = 4
    vocabulary_size: int = 32
    dropout: float = 0.1


def zero_padding(x, valid):
    return torch.where(valid.unsqueeze(-1), x, torch.zeros_like(x))


def feedforward(width, dropout):
    return nn.Sequential(nn.Linear(width, width*4), nn.GELU(), nn.Dropout(dropout),
                         nn.Linear(width*4, width), nn.Dropout(dropout))


class InputEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.width
        self.project = nn.Sequential(nn.LayerNorm(cfg.feature_dim), nn.Linear(cfg.feature_dim, d), nn.LayerNorm(d))
        self.role = nn.Embedding(3, d)
        self.state = nn.Embedding(5, d)
        self.kind = nn.Embedding(3, d)
        self.domain = nn.Embedding(cfg.vocabulary_size, d)
        self.fuse = nn.Sequential(nn.Linear(3*d, d), nn.GELU(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, batch):
        x, state, valid = batch['x'], batch['state'], batch['valid']
        observed = (state == OBSERVED) & valid.unsqueeze(-1)
        # Clear NaNs and hidden clean values before the first trainable operation.
        clean = torch.where(observed.unsqueeze(-1), x, torch.zeros_like(x))
        slot = self.project(clean)
        slot = torch.where(observed.unsqueeze(-1), slot, torch.zeros_like(slot))
        slot = slot + self.state(state) + self.role(torch.arange(3, device=x.device))
        slot = torch.where(((state != NA) & valid.unsqueeze(-1)).unsqueeze(-1), slot, torch.zeros_like(slot))
        connection = self.fuse(slot.flatten(-2))
        domain = slot[:, :, 0] + self.domain(batch['domain_type'])
        h = torch.where((batch['kind'] == 0).unsqueeze(-1), domain, connection)
        return zero_padding(self.norm(h + self.kind(batch['kind'])), valid)


class LocalBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads, self.radius, self.dropout = cfg.heads, cfg.radius, cfg.dropout
        self.norm1, self.norm2 = nn.LayerNorm(cfg.width), nn.LayerNorm(cfg.width)
        self.qkv = nn.Linear(cfg.width, cfg.width*3)
        self.output = nn.Linear(cfg.width, cfg.width)
        self.relative = nn.Embedding(2*cfg.radius+1, cfg.heads)
        self.relation = nn.Embedding(3, cfg.heads)
        self.ff = feedforward(cfg.width, cfg.dropout)

    def forward(self, h, batch):
        b, t, d = h.shape
        valid = batch['valid']
        q, k, v = self.qkv(self.norm1(h)).reshape(b,t,3,self.heads,d//self.heads).permute(2,0,3,1,4).unbind(0)
        pos = torch.arange(t, device=h.device)
        offset = pos[None, :] - pos[:, None]
        rel = self.relative(offset.clamp(-self.radius,self.radius)+self.radius).permute(2,0,1)
        chain = batch['chain']
        physical = (chain[:, :, None] != chain[:, None, :]).long()
        physical = torch.where((batch['kind'][:, :, None] == 2) | (batch['kind'][:, None, :] == 2), 2, physical)
        bias = self.relation(physical).permute(0,3,1,2) + rel.unsqueeze(0)
        allowed = (offset.abs() <= self.radius).unsqueeze(0) & valid[:,None,:]
        # Padding queries get a safe diagonal row and are removed after the update.
        allowed = allowed | ((~valid)[:, :, None] & torch.eye(t,device=h.device,dtype=torch.bool)[None])
        logits = q.float() @ k.float().transpose(-2,-1) / math.sqrt(d//self.heads) + bias.float()
        logits = logits.masked_fill(~allowed[:,None], float('-inf'))
        attention = F.dropout(logits.softmax(-1), self.dropout, training=self.training)
        update = (attention.to(v.dtype) @ v).transpose(1,2).reshape(b,t,d)
        h = zero_padding(h + F.dropout(self.output(update), self.dropout, training=self.training), valid)
        return zero_padding(h + self.ff(self.norm2(h)), valid)


class GlobalBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.width
        self.h_read, self.g_read = nn.LayerNorm(d), nn.LayerNorm(d)
        self.h_write, self.g_write = nn.LayerNorm(d), nn.LayerNorm(d)
        self.g_ff_norm, self.h_ff_norm = nn.LayerNorm(d), nn.LayerNorm(d)
        self.read = nn.MultiheadAttention(d, cfg.heads, dropout=cfg.dropout, batch_first=True)
        self.write = nn.MultiheadAttention(d, cfg.heads, dropout=cfg.dropout, batch_first=True)
        self.g_ff, self.h_ff = feedforward(d, cfg.dropout), feedforward(d, cfg.dropout)
        self.gate = nn.Parameter(torch.tensor(-2.0))
        # Allows initialization from the local-only encoder without changing H.
        nn.init.zeros_(self.write.out_proj.weight); nn.init.zeros_(self.write.out_proj.bias)
        nn.init.zeros_(self.h_ff[3].weight); nn.init.zeros_(self.h_ff[3].bias)

    def forward(self, h, g, valid):
        hn = self.h_read(h)
        g = g + self.read(self.g_read(g), hn, hn, key_padding_mask=~valid, need_weights=False)[0]
        g = g + self.g_ff(self.g_ff_norm(g))
        gn = self.g_write(g)
        candidate = h + self.write(self.h_write(h), gn, gn, need_weights=False)[0]
        candidate = candidate + self.h_ff(self.h_ff_norm(candidate))
        return zero_padding(h + 0.5*self.gate.sigmoid()*(candidate-h), valid), g


class AssemblyEncoder(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or Config()
        cfg = self.cfg
        if cfg.width % cfg.heads:
            raise ValueError('Width must be divisible by attention heads')
        self.input = InputEncoder(cfg)
        self.local = nn.ModuleList(LocalBlock(cfg) for _ in range(cfg.local_layers))
        self.global_blocks = nn.ModuleList(GlobalBlock(cfg) for _ in range(cfg.global_layers))
        self.global_tokens = nn.Parameter(torch.randn(cfg.global_slots,cfg.width)*0.02)
        self.norm = nn.LayerNorm(cfg.width)
        self.decoder_role, self.decoder_kind = nn.Embedding(3,16), nn.Embedding(3,16)
        self.decoder = nn.Sequential(nn.LayerNorm(cfg.width+32), nn.Linear(cfg.width+32,512), nn.GELU(), nn.Linear(512,cfg.feature_dim))
        self.activity_head = nn.Sequential(nn.LayerNorm(3*cfg.width+2), nn.Linear(3*cfg.width+2,128),
            nn.GELU(), nn.Dropout(0.3), nn.Linear(128,1))

    def features(self, batch, context=True):
        if not bool(batch['valid'].any(dim=1).all()):
            raise ValueError('Empty assembly is invalid')
        h = self.input(batch)
        for block in self.local:
            h = block(h,batch)
        g = self.global_tokens.unsqueeze(0).expand(h.shape[0],-1,-1)
        if context:
            for block in self.global_blocks:
                h,g = block(h,g,batch['valid'])
        return zero_padding(self.norm(h),batch['valid'])

    def predict_masked(self, corrupted_batch):
        if not bool(((corrupted_batch['state'] == MASKED) & corrupted_batch['valid'].unsqueeze(-1)).any()):
            raise ValueError('Natural prediction requires an explicitly corrupted input')
        h = self.features(corrupted_batch)
        b,t,d = h.shape
        roles = self.decoder_role(torch.arange(3,device=h.device))[None,None].expand(b,t,-1,-1)
        kinds = self.decoder_kind(corrupted_batch['kind']).unsqueeze(2).expand(-1,-1,3,-1)
        pred = self.decoder(torch.cat([h.unsqueeze(2).expand(-1,-1,3,-1),roles,kinds],dim=-1))
        return torch.where(corrupted_batch['valid'][:,:,None,None],pred,torch.zeros_like(pred))

    def activity(self, clean_batch):
        if bool((clean_batch['state'] == MASKED).any()):
            raise ValueError('Activity prediction expects the clean complete assembly view')
        h = self.features(clean_batch)
        valid, kind = clean_batch['valid'], clean_batch['kind']
        dm, em = valid & (kind == 0), valid & (kind != 0)
        nd, ne = dm.sum(1,keepdim=True), em.sum(1,keepdim=True)
        dmean = (h*dm.unsqueeze(-1)).sum(1)/nd.clamp_min(1)
        emean = (h*em.unsqueeze(-1)).sum(1)/ne.clamp_min(1)
        emax = h.masked_fill(~em.unsqueeze(-1),float('-inf')).amax(1)
        emax = torch.where(ne > 0,emax,torch.zeros_like(emax))
        pooled = torch.cat([dmean,emean,emax,nd.float().log1p(),ne.float().log1p()],dim=-1)
        return self.activity_head(pooled).squeeze(-1)


def collate_records(records, lookup, feature_dim=1152):
    """Strict observed-cache lookups. Missing slots remain explicit states."""
    b,t = len(records),max(len(r['route']) for r in records)
    out = dict(x=torch.zeros(b,t,3,feature_dim), state=torch.zeros(b,t,3,dtype=torch.long),
        kind=torch.zeros(b,t,dtype=torch.long),domain_type=torch.zeros(b,t,dtype=torch.long),
        chain=torch.full((b,t),-1,dtype=torch.long),valid=torch.zeros(b,t,dtype=torch.bool))
    states = {'NA':NA,'OBSERVED':OBSERVED,'EMPTY':EMPTY,'MISSING':MISSING,'MASKED':MASKED}
    for i,rec in enumerate(records):
        n=len(rec['route']);out['valid'][i,:n]=True
        for j,obj in enumerate(rec['route']):
            out['kind'][i,j]=obj['kind'];out['domain_type'][i,j]=obj['domain_type'] if obj['kind']==0 else 0
            out['chain'][i,j]=obj['chain'] if obj['kind']!=2 else -1
        for slot in rec['model_slots']:
            j,k=slot['object_index'],slot['slot'];out['state'][i,j,k]=states[slot['state']]
            if slot['state']=='OBSERVED':
                vector=torch.as_tensor(lookup(slot['sequence_hash']),dtype=torch.float32)
                if vector.shape!=(feature_dim,) or not bool(torch.isfinite(vector).all()) or not bool(vector.abs().sum()>0):
                    raise ValueError('Invalid observed cached vector')
                out['x'][i,j,k]=vector
    return out
