"""Run: python -m unittest -v test_contracts.py test_torch_reference.py

These tests exercise new table/provenance math; they do not validate biology.
"""
import unittest
import numpy as np
from neuralps_v2.masking_contract import Span, CopyAlias, SlotProvenance, mask_slots, purged_split
from neuralps_v2.table_math import conditional_scores, reference_relative, candidate_rank_loss


class ContractTests(unittest.TestCase):
    def test_perfect_candidate_and_slot_orientation(self):
        pred = np.array([[[1.,0.],[0.,1.]]])
        cand = np.array([[[[1.,0.],[0.,1.]],[[0.,1.],[1.,0.]]]])
        scores = conditional_scores(pred,cand,np.zeros_like(pred),np.ones((1,2),bool))
        np.testing.assert_allclose(scores,[[1.,0.]])

    def test_invalid_nan_candidate_never_contaminates(self):
        p = np.array([[[1.,2.]]])
        c = np.array([[[[1.,2.]],[[np.nan,np.nan]],[[2.,-1.]]]])
        v = np.array([[True,False,True]])
        s = conditional_scores(p,c,np.zeros_like(p),v)
        self.assertTrue(np.isneginf(s[0,1]))
        loss = candidate_rank_loss(s,[[True,False,False]],v)
        self.assertTrue(np.isfinite(loss).all())

    def test_zero_centered_target_uses_raw_loss(self):
        p = np.array([[[1.,2.]]])
        c = p[:,None]
        s = conditional_scores(p,c,p,np.array([[True]]))
        np.testing.assert_allclose(s,[[1.]])

    def test_reference_offset_invariance(self):
        s = np.array([[1.,2.],[4.,6.]])
        ref = np.array([[0.,1.,np.nan],[2.,5.,np.nan]])
        m = np.array([[True,True,False],[True,True,False]])
        shift = np.array([[100.],[-77.]])
        np.testing.assert_allclose(reference_relative(s,ref,m),
                                   reference_relative(s+shift,ref+shift,m))

    def test_candidate_permutation_and_multiple_positives(self):
        s = np.array([[.2,.8,.5,-.3]])
        pos = np.array([[False,True,True,False]])
        valid = np.ones_like(pos)
        order = [3,1,0,2]
        np.testing.assert_allclose(candidate_rank_loss(s,pos,valid),
                                   candidate_rank_loss(s[:,order],pos[:,order],valid[:,order]))
        self.assertLess(candidate_rank_loss(s,pos,valid)[0],
                        candidate_rank_loss(s,[[False,True,False,False]],valid)[0])

    def test_rank_training_rejects_no_alternatives(self):
        with self.assertRaises(ValueError):
            candidate_rank_loss([[.2]],[[True]],[[True]])

    def test_owned_target_not_full_parent_target(self):
        slots = [
            SlotProvenance((Span('p',40,50),),(Span('p',0,100),),'target'),
            SlotProvenance((Span('p',70,80),),(Span('p',0,100),),'contextual'),
            SlotProvenance((Span('p',70,80),),(Span('p',70,80),),'independent'),
        ]
        hidden,forbidden = mask_slots(slots,[0])
        self.assertEqual(hidden,[True,True,False])
        self.assertEqual(forbidden,(Span('p',40,50),))

    def test_aliases_map_only_target_slice_and_do_not_flood(self):
        slots = [
            SlotProvenance((Span('p',40,50),),(Span('p',40,50),),'target'),
            SlotProvenance((Span('q',0,100),),(Span('q',0,100),),'whole_copy'),
            SlotProvenance((Span('q',90,100),),(Span('q',90,100),),'other_region'),
        ]
        hidden,forbidden = mask_slots(slots,[0],[CopyAlias(Span('p',0,100),Span('q',0,100))])
        self.assertEqual(hidden,[True,True,False])
        self.assertEqual(set(forbidden),{Span('p',40,50),Span('q',40,50)})

    def test_exact_whole_sequence_copy_masks_its_dependents(self):
        slots = [
            SlotProvenance((Span('p',0,20),),(Span('p',0,20),),'same'),
            SlotProvenance((Span('q',0,20),),(Span('q',0,20),),'same'),
            SlotProvenance((Span('q',15,30),),(Span('q',15,30),),'flank'),
        ]
        hidden,_ = mask_slots(slots,[0])
        self.assertEqual(hidden,[True,True,True])

    def test_multi_donor_purge_and_inner_purge(self):
        parts = [{'a','b'},{'b','c'},{'a','d'},{'c','d'}]
        train,test = purged_split(parts,{'a'})
        self.assertEqual(train,[1,3]); self.assertEqual(test,[0,2])
        fit,val = purged_split(parts,{'b'},train)
        self.assertEqual(fit,[3]); self.assertEqual(val,[1])
        self.assertFalse(any(parts[i]&{'a','b'} for i in fit))


if __name__ == '__main__':
    unittest.main()
