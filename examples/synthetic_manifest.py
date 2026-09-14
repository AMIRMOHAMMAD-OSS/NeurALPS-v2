"""Prepare an artificial D–covalent–D–break–D route and demonstrate masking."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from neuralps_v2.manifest_adapter import prepare_record, validate_record


def make_example():
    proteins = {
        'toy_protein_0': 'ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY',
        'toy_protein_1': 'YWVTSRQPNMLKIHGFEDCAYWVTSRQPNMLKIHGFEDCAYWVTSRQPNMLKIHGFEDCA',
    }
    store = {}
    rng = np.random.default_rng(20260914)

    def observed(protein, start, stop):
        sequence = proteins[protein][start:stop]
        sequence_hash = hashlib.sha256(sequence.encode()).hexdigest()
        context_hash = hashlib.sha256(proteins[protein].encode()).hexdigest()
        key = f'synthetic:{context_hash}:{start}:{stop}'
        store[key] = rng.standard_normal(1152).astype(np.float32)
        return dict(state='observed', hash=sequence_hash, feature_key=key,
                    owned=[[protein, start, stop]],
                    dependencies=[[protein, 0, len(proteins[protein])]])

    def single(kind, chain, start, stop):
        return dict(kind=kind, chain=chain,
                    slots=[observed(f'toy_protein_{chain}', start, stop),
                           dict(state='na'), dict(state='na')])

    record = dict(id='synthetic_two_proteins', tokens=[
        single('domain', 0, 0, 20),
        single('covalent', 0, 10, 30),
        single('domain', 0, 20, 40),
        dict(kind='break', chain=-1, slots=[
            dict(state='na'), observed('toy_protein_0', 40, 60),
            observed('toy_protein_1', 0, 20)]),
        single('domain', 1, 20, 40),
    ])
    validate_record(record)
    return record, store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', type=Path, help='Optional directory for synthetic JSONL/index/shard files.')
    args = parser.parse_args()
    record, store = make_example()
    clean = prepare_record(record, store)
    masked = prepare_record(record, store, primary=[(1, 0)])
    np.testing.assert_array_equal(clean['teacher'], masked['teacher'])
    assert np.all(masked['x'][masked['state'] == 4] == 0)
    assert masked['primary_target'].sum() == 1
    assert np.array_equal(masked['state'][3], [0, 4, 1])
    if args.write:
        args.write.mkdir(parents=True, exist_ok=True)
        np.save(args.write / 'features.npy', np.stack(list(store.values())))
        index = {key: dict(file='features.npy', row=i) for i, key in enumerate(store)}
        (args.write / 'feature_index.json').write_text(json.dumps(index, indent=2)+'\n')
        (args.write / 'graphs.jsonl').write_text(json.dumps(record)+'\n')
    print(json.dumps(dict(
        example='synthetic; random vectors, not ESM-C outputs',
        tokens=len(record['tokens']), clean_shape=list(clean['x'].shape),
        primary_targets=int(masked['primary_target'].sum()),
        masked_slots=int((masked['state'] == 4).sum()),
        break_states_after_mask=masked['state'][3].tolist(),
        clean_teacher_preserved=True,
    ), indent=2))


if __name__ == '__main__':
    main()
