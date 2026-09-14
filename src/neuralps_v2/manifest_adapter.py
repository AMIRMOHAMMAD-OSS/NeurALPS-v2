"""Canonical JSONL graph adapter. No inference of legacy provenance or splits.

NumPy preparation is independently testable; torch is imported only for collation.
See docs/data-interface.md for the schema and integration boundary.
"""
import json
import re
from pathlib import Path
import numpy as np
from neuralps_v2.masking_contract import Span, SlotProvenance, CopyAlias, mask_slots

KINDS = {'domain': 0, 'covalent': 1, 'break': 2}
STATES = {'na': 0, 'observed': 1, 'empty': 2, 'missing': 3}


def provenance(slot):
    return SlotProvenance(tuple(Span(*s) for s in slot['owned']),
                          tuple(Span(*s) for s in slot['dependencies']), slot['hash'])


def validate_record(record, domain_types=32):
    if not isinstance(record.get('id'), str) or not record['id']:
        raise ValueError('Record needs a nonempty id')
    tokens = record['tokens']
    if not tokens or len(tokens) % 2 != 1:
        raise ValueError('Graph must start and end with domains')
    for i, token in enumerate(tokens):
        kind = token['kind']
        if kind not in KINDS or (kind == 'domain') != (i % 2 == 0):
            raise ValueError('Expected alternating domains and physical connections')
        if type(token['chain']) is not int or (token['chain'] < 0 and kind != 'break'):
            raise ValueError('Physical tokens require nonnegative chain ids')
        if kind == 'break':
            if token['chain'] != -1 or tokens[i-1]['chain'] == tokens[i+1]['chain']:
                raise ValueError('Break must join different chains and carry chain=-1')
        elif kind == 'covalent':
            if not token['chain'] == tokens[i-1]['chain'] == tokens[i+1]['chain']:
                raise ValueError('Covalent connection crosses chains')
        dt = token.get('domain_type', 0)
        if type(dt) is not int or not 0 <= dt < domain_types or (kind != 'domain' and dt != 0):
            raise ValueError('Invalid canonical domain type')
        if len(token['slots']) != 3:
            raise ValueError('Exactly three slots required')
        for s, slot in enumerate(token['slots']):
            state = slot['state']
            applicable = s > 0 if kind == 'break' else s == 0
            if state not in STATES or (state != 'na') != applicable:
                raise ValueError('Invalid slot state/layout')
            if state == 'observed':
                if not re.fullmatch('[0-9a-f]{64}', slot.get('hash', '')):
                    raise ValueError('Expected lowercase sequence sha256')
                if not isinstance(slot.get('feature_key'), str) or not slot['feature_key']:
                    raise ValueError('Observed slots need a context-aware feature_key separate from sequence hash')
                p = provenance(slot)
                if not p.owned or not p.dependencies:
                    raise ValueError('Observed features need ownership and PLM dependencies')
                for owned in p.owned:
                    if not any(d.source == owned.source and d.start <= owned.start and d.stop >= owned.stop for d in p.dependencies):
                        raise ValueError('Dependencies must cover owned residues')
        meta = np.asarray(token.get('meta', [0.] * 6))
        if meta.shape != (6,) or not np.isfinite(meta).all():
            raise ValueError('Expected six finite metadata values')
    if 'label' in record and (type(record['label']) not in (int, float) or record['label'] not in (0, 1)):
        raise ValueError('Activity label must be binary')
    return record


class NpyStore:
    """Index maps feature_key to {file: relative .npy path, row: integer}.

    Each array must be float32 [rows,1152]. Feature keys encode extraction
    context and recipe, while slot hashes identify sequence for copy masking.
    Same-space/cache provenance must be audited upstream.
    """
    def __init__(self, index_path):
        self.root = Path(index_path).resolve().parent
        self.index = json.loads(Path(index_path).read_text())
        self.arrays = {}

    def __getitem__(self, key):
        item = self.index[key]
        path = (self.root / item['file']).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Shard path escapes index directory')
        if path not in self.arrays:
            a = np.load(path, mmap_mode='r', allow_pickle=False)
            if a.dtype != np.float32 or a.ndim != 2 or a.shape[1] != 1152:
                raise ValueError('Expected float32 [rows,1152] shard')
            self.arrays[path] = a
        a = self.arrays[path]
        row = item['row']
        if type(row) is not int or not 0 <= row < len(a):
            raise ValueError('Invalid shard row')
        value = np.array(a[row], copy=True)
        if not np.isfinite(value).all():
            raise ValueError('Nonfinite cache vector')
        return value


def prepare_record(record, store, primary=(), domain_types=32):
    """primary contains (token, slot) pairs. Teacher stays clean; input is zeroed."""
    validate_record(record, domain_types)
    tokens = record['tokens']; n = len(tokens)
    x = np.zeros((n, 3, 1152), dtype=np.float32)
    state = np.array([[STATES[s['state']] for s in t['slots']] for t in tokens], dtype=np.int64)
    locations, slots = [], []
    for i, t in enumerate(tokens):
        for j, s in enumerate(t['slots']):
            if s['state'] == 'observed':
                x[i, j] = store[s['feature_key']]
                locations.append((i, j)); slots.append(provenance(s))
    teacher = x.copy()
    target = np.zeros((n, 3), dtype=bool)
    meta = np.array([t.get('meta', [0.] * 6) for t in tokens], dtype=np.float32)
    forbidden = ()
    if primary:
        primary = [tuple(p) for p in primary]
        if any(p not in locations for p in primary):
            raise ValueError('Primary targets must be observed slots')
        aliases = [CopyAlias(Span(*a['a']), Span(*a['b'])) for a in record.get('aliases', [])]
        hidden, forbidden = mask_slots(slots, [locations.index(p) for p in primary], aliases)
        for loc, hide in zip(locations, hidden):
            if hide:
                state[loc] = 4; x[loc] = 0; meta[loc[0]] = 0
        for loc in primary:
            target[loc] = True
    return dict(x=x, state=state, kind=np.array([KINDS[t['kind']] for t in tokens]),
                domain_type=np.array([t.get('domain_type', 0) for t in tokens]),
                pos=np.arange(n), chain=np.array([t['chain'] for t in tokens]), meta=meta,
                valid=np.ones(n, dtype=bool), teacher=teacher, primary_target=target,
                forbidden=forbidden)


def collate(prepared, device='cpu'):
    """Return batch/teacher/primary_target; training-only teacher means supplied separately."""
    import torch
    from neuralps_v2.neuralps_reference import Batch
    if not prepared:
        raise ValueError('Cannot collate empty batch')
    longest = max(len(p['x']) for p in prepared)
    fields = {}
    for key in ('x', 'state', 'kind', 'domain_type', 'pos', 'chain', 'meta', 'valid', 'teacher', 'primary_target'):
        sample = prepared[0][key]
        array = np.zeros((len(prepared), longest, *sample.shape[1:]), dtype=sample.dtype)
        for i, p in enumerate(prepared):
            array[i, :len(p[key])] = p[key]
        fields[key] = torch.as_tensor(array, device=device)
    teacher = fields.pop('teacher'); primary = fields.pop('primary_target')
    batch = Batch(**fields); batch.validate()
    return {'batch': batch, 'teacher': teacher, 'primary_target': primary}


def read_records(path, domain_types=32):
    seen = set()
    with open(path) as stream:
        for line in stream:
            r = validate_record(json.loads(line), domain_types)
            if r['id'] in seen:
                raise ValueError('Duplicate assembly id: ' + r['id'])
            seen.add(r['id'])
            yield r


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Validate canonical graph JSONL and referenced cache rows')
    parser.add_argument('manifest'); parser.add_argument('--index'); parser.add_argument('--domain-types', type=int, default=32)
    args = parser.parse_args()
    store = NpyStore(args.index) if args.index else None
    count = 0
    for r in read_records(args.manifest, args.domain_types):
        if store is not None:
            prepare_record(r, store, domain_types=args.domain_types)
        count += 1
    print(json.dumps({'validated_records': count, 'cache_checked': store is not None}))


if __name__ == '__main__':
    main()
