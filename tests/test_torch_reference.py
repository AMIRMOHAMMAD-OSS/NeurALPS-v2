"""Target-environment smoke checks; explicitly skipped when PyTorch is absent."""
import unittest
import importlib.util
HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from dataclasses import replace
    from neuralps_v2.neuralps_reference import Batch, NeurALPS, NA, OBSERVED, MASKED
    from neuralps_v2.conditional_heads import initialize_child, conditional_scores
    import table_math
    torch.set_num_threads(2)


@unittest.skipUnless(HAS_TORCH, 'PyTorch is not installed in this environment.')
class TorchReferenceTests(unittest.TestCase):
    def batch(self):
        torch.manual_seed(123)
        kind = torch.tensor([[0,1,0,1,0,1,0]])
        state = torch.full((1,7,3),NA,dtype=torch.long)
        state[:,:,0] = OBSERVED
        return Batch(torch.randn(1,7,3,1152),state,kind,torch.zeros((1,7),dtype=torch.long),
                     torch.arange(7)[None],torch.zeros((1,7),dtype=torch.long),
                     torch.zeros(1,7,6),torch.ones((1,7),dtype=torch.bool))

    def test_a_c_d_initialization_identity(self):
        b = self.batch()
        a,c,d = [NeurALPS(v).eval() for v in ('A','C','D')]
        initialize_child(a,c); initialize_child(c,d)
        with torch.no_grad():
            ha,hc,hd = [m(b)['tokens'] for m in (a,c,d)]
        torch.testing.assert_close(ha,hc,rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(hc,hd,rtol=1e-5,atol=1e-6)

    def test_masked_cache_values_cannot_change_d(self):
        b = self.batch()
        b.state[:,2,0] = MASKED
        alt = replace(b,x=b.x.clone())
        alt.x[:,2,0] = float('nan')
        model = NeurALPS('D').eval()
        with torch.no_grad():
            torch.testing.assert_close(model(b)['tokens'],model(alt)['tokens'])

    def test_padded_nan_cannot_change_valid_outputs(self):
        b = self.batch()
        b.valid[:,-1] = False
        alt = replace(b,x=b.x.clone(),meta=b.meta.clone())
        alt.x[:,-1] = float('nan'); alt.meta[:,-1] = float('nan')
        m = NeurALPS('C').eval()
        with torch.no_grad():
            torch.testing.assert_close(m(b)['tokens'][:,:-1],m(alt)['tokens'][:,:-1])

    def test_candidate_math_matches_numpy_oracle(self):
        torch.manual_seed(20)
        p,t,mu = torch.randn(2,2,8),torch.randn(2,4,2,8),torch.randn(2,2,8)
        valid = torch.tensor([[True,True,False,True],[True,False,True,True]])
        got = conditional_scores(p,t,mu,valid)
        want = torch.tensor(table_math.conditional_scores(p.numpy(),t.numpy(),mu.numpy(),valid.numpy()))
        torch.testing.assert_close(got.double(),want,atol=1e-6,rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
