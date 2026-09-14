import copy
import importlib.util
import json
from pathlib import Path
import unittest
import numpy as np
from neuralps_v2.evaluation_protocol import auroc, spearman, donor_fold, igem_reported_threshold
from neuralps_v2.manifest_adapter import prepare_record, validate_record
from tests.test_manifest_adapter import fixture
from neuralps_v2.phase2_preflight import validate_config, validate_registry


class ProtocolTests(unittest.TestCase):
    def test_igem_precision_is_not_accuracy(self):
        m = igem_reported_threshold()['metrics']
        self.assertAlmostEqual(m['precision'],43/51)
        self.assertAlmostEqual(m['accuracy'],77/105)
        self.assertAlmostEqual(m['balanced_accuracy'],.5*(43/63+34/42))

    def test_auc_ties_and_missing_class(self):
        self.assertEqual(auroc([0,1],[2,2]),.5)
        self.assertEqual(auroc([0,1,0,1],[0,3,1,2]),1.)
        self.assertIsNone(auroc([1,1],[1,2]))

    def test_quantitative_rank_ties(self):
        self.assertAlmostEqual(spearman([0,0,1,2],[3,3,2,1]),-1.)
        self.assertIsNone(spearman([1,1],[2,3]))

    def test_multidonor_purge_and_missing_membership(self):
        records=[dict(engineered_donors=['a','b']),dict(engineered_donors=['b']),dict(engineered_donors=['c'])]
        self.assertEqual(donor_fold(records,{'a'}),([1,2],[0]))
        with self.assertRaises(ValueError): donor_fold([{},records[0]],{'a'})

    def test_context_key_is_not_sequence_hash(self):
        r=fixture()
        for i,t in enumerate(r['tokens']): t['slots'][0]['feature_key']=f'context-{i}'
        store={f'context-{i}':np.full(1152,i+1,np.float32) for i in range(3)}
        p=prepare_record(r,store)
        np.testing.assert_array_equal(p['x'][:,0,0],[1,2,3])
        del r['tokens'][0]['slots'][0]['feature_key']
        with self.assertRaises(ValueError): validate_record(r)

    def test_configuration_and_stage_graph(self):
        root=Path(__file__).resolve().parents[1]/'configs'
        c=json.loads((root/'phase2_final_config.json').read_text())
        self.assertTrue(validate_config(c))
        self.assertTrue(validate_registry(json.loads((root/'experiment_registry.json').read_text())))
        c['local']['blocks']=2
        with self.assertRaises(ValueError): validate_config(c)


HAS_TORCH=importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from neuralps_v2.conditional_heads import FoundationBundle
    from neuralps_v2.phase2_training import build_bundle, build_optimizer, copy_variant
    from neuralps_v2.training_objectives import current_rank_loss, activity_step, activity_loss
    from neuralps_v2.neuralps_reference import Batch
    torch.set_num_threads(2)


@unittest.skipUnless(HAS_TORCH,'PyTorch is not installed in this environment.')
class NeuralIntegrationTests(unittest.TestCase):
    def batch(self):
        state=torch.tensor([[[1,0,0],[4,0,0],[1,0,0]]])
        return Batch(torch.randn(1,3,3,1152),state,torch.tensor([[0,1,0]]),torch.zeros(1,3,dtype=torch.long),
                     torch.arange(3)[None],torch.zeros(1,3,dtype=torch.long),torch.zeros(1,3,6),torch.ones(1,3,dtype=torch.bool))

    def test_rank_uses_current_prediction_gradient(self):
        b=self.batch(); p=torch.randn(1,3,3,1152,requires_grad=True)
        primary=torch.zeros(1,3,3,dtype=torch.bool);primary[0,1,0]=True
        rank=dict(query_rows=torch.tensor([0]),token_indices=torch.tensor([[1]]),slot_indices=torch.tensor([[0]]),
                  candidates=torch.randn(1,2,1,1152),mean=torch.zeros(1,1,1152),
                  positive=torch.tensor([[True,False]]),valid=torch.tensor([[True,True]]))
        loss=current_rank_loss(p,b,primary,rank,0.0);loss.backward()
        self.assertGreater(p.grad[0,1,0].abs().sum().item(),0)
        rank['query_pred']=p[0,1,0]
        with self.assertRaises(ValueError): current_rank_loss(p,b,primary,rank,0.0)

    def test_warm_residue_gate_and_rate_groups(self):
        c=json.loads((Path(__file__).resolve().parents[1]/'configs'/'phase2_final_config.json').read_text())
        b=build_bundle(c,'A','residue')
        opt,names=build_optimizer(b,'A_residue_warm',lr=1e-4,lr_by_prefix={'residue_encoder':3e-4})
        self.assertIn('encoder.input.raw_gate',names)
        self.assertEqual(set(g['lr'] for g in opt.param_groups),{1e-4,3e-4})
        self.assertFalse(b.encoder.input.project[1].weight.requires_grad)

    def test_complete_bundle_inheritance(self):
        a,c=FoundationBundle('A'),FoundationBundle('C')
        with torch.no_grad(): a.decoder.net[-1].bias.fill_(.7)
        copy_variant(a,c)
        torch.testing.assert_close(a.decoder.net[-1].bias,c.decoder.net[-1].bias)

    def test_additive_components_sum(self):
        from neuralps_v2.interpretability_head import AdditiveActivityHead
        b=self.batch(); h=torch.randn(1,3,256); parts=AdditiveActivityHead().components(h,b)
        torch.testing.assert_close(parts['logit'],parts['domain_contribution'].sum(1)+parts['connection_contribution'].sum(1)+parts['background_contribution'])

    def test_residue_padding_nan_invariance(self):
        from neuralps_v2.pretrained_residue import PretrainedResidueEncoder
        pool=PretrainedResidueEncoder().eval();x=torch.randn(1,5,1152)
        valid=torch.tensor([[True,True,True,False,False]]);region=torch.tensor([[0,1,2,0,0]])
        y=x.clone();y[:,3:]=float('nan')
        torch.testing.assert_close(pool(x,region,valid)[1],pool(y,region,valid)[1])

    def test_activity_supervises_both_passes(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.a=torch.nn.Parameter(torch.tensor([1.]));self.b=torch.nn.Parameter(torch.tensor([-1.]))
            def forward(self,batch,**kwargs):
                return {'cycle_activity_logits':[self.a,self.b]}
        m=Model(); labels=torch.ones(1)
        expected=.25*activity_loss(m.a,labels)+.75*activity_loss(m.b,labels)
        result=activity_step(m,self.batch(),labels,optimizer=torch.optim.SGD(m.parameters(),lr=0.0))
        torch.testing.assert_close(result['loss'],expected.detach())
        self.assertIsNotNone(m.a.grad);self.assertIsNotNone(m.b.grad)

if __name__=='__main__': unittest.main()
