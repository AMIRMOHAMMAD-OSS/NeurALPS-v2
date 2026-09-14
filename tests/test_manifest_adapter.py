import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from neuralps_v2.manifest_adapter import prepare_record, validate_record, NpyStore


def fixture():
    def token(kind, start, stop):
        slot = dict(state='observed', hash=str(start//10+1)*64,
                    feature_key=str(start//10+1)*64,
                    owned=[['protein',start,stop]], dependencies=[['protein',0,40]])
        return dict(kind=kind, chain=0, slots=[slot,dict(state='na'),dict(state='na')])
    return dict(id='synthetic', tokens=[token('domain',0,10),token('covalent',10,20),token('domain',20,30)])


class AdapterTests(unittest.TestCase):
    def test_dependency_closure_preserves_teacher(self):
        r=fixture(); store={t['slots'][0]['hash']:np.ones(1152,np.float32) for t in r['tokens']}
        p=prepare_record(r,store,[(1,0)])
        self.assertTrue((p['state'][:,0]==4).all())
        self.assertEqual(p['primary_target'].sum(),1)
        self.assertEqual(p['x'].sum(),0)
        self.assertEqual(p['teacher'].sum(),3456)

    def test_module_annotation_invariance(self):
        r=fixture(); other=copy.deepcopy(r)
        for i,t in enumerate(other['tokens']): t['module_id']=i; t['annotation']='J'
        store={t['slots'][0]['hash']:np.ones(1152,np.float32) for t in r['tokens']}
        a,b=prepare_record(r,store),prepare_record(other,store)
        for key in a:
            if isinstance(a[key],np.ndarray): np.testing.assert_array_equal(a[key],b[key])

    def test_reject_cross_chain_covalent(self):
        r=fixture(); r['tokens'][2]['chain']=1
        with self.assertRaises(ValueError): validate_record(r)

    def test_break_has_two_termini(self):
        r=fixture(); t=r['tokens'][1]; s=t['slots'][0]
        t.update(kind='break',chain=-1,slots=[dict(state='na'),s,copy.deepcopy(s)])
        r['tokens'][2]['chain']=1
        validate_record(r)
        t['slots'][2]={'state':'na'}
        with self.assertRaises(ValueError): validate_record(r)

    def test_missing_provenance_rejected(self):
        r=fixture(); r['tokens'][0]['slots'][0]['dependencies']=[]
        with self.assertRaises(ValueError): validate_record(r)

    def test_shard_contract(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); np.save(p/'a.npy',np.ones((1,1152),np.float32))
            (p/'index.json').write_text(json.dumps({'key':{'file':'a.npy','row':0}}))
            store=NpyStore(p/'index.json'); self.assertEqual(store['key'].shape,(1152,))
            store.index['key']['row']=-1
            with self.assertRaises(ValueError): store['key']

if __name__=='__main__': unittest.main()
