import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from neuralps.inputs import build_record, region_targets
from neuralps.features import typed_context, contributions
from neuralps.head import fit, predict, save_model, load_model, objective
from neuralps.training import validate_training
from neuralps.scoring import make_plan, Runtime, reconstruction_agreement, predict_mode
from neuralps.contracts import SSL_SHA, sha, dump

ROOT = Path(__file__).resolve().parents[1]

def fixture():
    rng = np.random.default_rng(11)
    aa = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
    sequence = "".join(rng.choice(aa, 1400))
    domains = [dict(start=40+i*160, end=130+i*160, type="PCP" if i%2 else "AMP-binding") for i in range(7)]
    return dict(assembly_id="fixture", proteins=[dict(id="chain1", sequence=sequence, domains=domains)])

class InputAndMaskTests(unittest.TestCase):
    def test_windows_and_independent_hashes(self):
        record, parents, sequences = build_record(fixture())
        self.assertEqual(len(record["route"]), 13)
        s = record["model_slots"][1]
        self.assertEqual((s["start"], s["end"]), (110, 220))
        for s in record["model_slots"]:
            seq = parents[s["parent_sequence_hash"]][s["start"]:s["end"]]
            self.assertEqual(sequences[s["sequence_hash"]], seq)
            self.assertEqual(s["sequence_hash"], hashlib.sha256(seq.encode()).hexdigest())

    def test_overlap_seam(self):
        spec = fixture()
        spec["proteins"][0]["domains"][1]["start"] = 110
        rec, _, _ = build_record(spec)
        self.assertEqual((rec["model_slots"][1]["start"], rec["model_slots"][1]["end"]), (90, 150))

    def test_break_is_two_physical_tails(self):
        spec = fixture()
        second = copy.deepcopy(spec["proteins"][0]); second["id"] = "chain2"
        spec["proteins"].append(second)
        rec, _, _ = build_record(spec)
        s = [s for s in rec["model_slots"] if s["kind"] == 2]
        self.assertEqual([x["slot"] for x in s], [1, 2])
        self.assertEqual([(x["start"], x["end"]) for x in s], [(1090, 1154), (0, 40)])

    def test_small_tail_stays_missing(self):
        spec = fixture()
        second = copy.deepcopy(spec["proteins"][0]); second["id"] = "chain2"
        second["domains"][0]["start"] = 1
        spec["proteins"].append(second)
        rec, parents, _ = build_record(spec)
        j = next(j for j, o in enumerate(rec["route"]) if o["kind"] == 2)
        self.assertEqual(make_plan(rec, [j], parents)["status"], "MISSING_SEQUENCE_FEATURE")

    def test_reject_unresolved_type_and_coordinates(self):
        for change in [dict(type="Condensation"), dict(start=-1), dict(end=2000)]:
            spec = fixture(); spec["proteins"][0]["domains"][0].update(change)
            with self.assertRaises(ValueError): build_record(spec)

    def test_joint_mask_hides_overlapping_features(self):
        rec, parents, _ = build_record(fixture())
        ds, bs = region_targets(rec, [4, 6])
        plan = make_plan(rec, ds+bs, parents)
        self.assertEqual(plan["status"], "READY")
        hidden = set(map(tuple, plan["hidden"]))
        self.assertTrue({(j,0) for j in [2,3,4,5,6,7,8]} <= hidden)
        self.assertEqual(plan["targets"], [3,4,5,6,7])

    def test_sequence_alias_is_hidden_on_other_chain(self):
        spec = fixture(); other = copy.deepcopy(spec["proteins"][0]); other["id"] = "copy"
        spec["proteins"].append(other)
        rec, parents, _ = build_record(spec)
        plan = make_plan(rec, [4], parents)
        self.assertTrue({(4,0),(18,0)} <= set(map(tuple, plan["hidden"])))

    def test_masking_all_context_remains_unscored(self):
        rec, parents, _ = build_record(fixture())
        self.assertEqual(make_plan(rec, list(range(len(rec["route"]))), parents)["status"], "INSUFFICIENT_VISIBLE_CONTEXT")

    def test_target_region_must_be_contiguous_domains(self):
        rec, _, _ = build_record(fixture())
        for indices in ([1], [0,4], [0,0], []):
            with self.assertRaises(ValueError): region_targets(rec, indices)

class HeadTests(unittest.TestCase):
    def test_exact_494_membership_and_missing_case_rejected(self):
        expected = json.loads((ROOT/"data/expected_labels.json").read_text())
        rows = json.loads((ROOT/"data/training_rows.json").read_text())
        with np.load(ROOT/"data/training_features.npz") as z: x = z["features"]
        _, y = validate_training(rows, x, expected)
        self.assertEqual(int(y.sum()),286)
        with self.assertRaises(ValueError): validate_training(rows[:-1],x[:-1],expected)
        bad=copy.deepcopy(rows);bad[0]["observed_activity"] = 1-bad[0]["observed_activity"]
        with self.assertRaises(ValueError): validate_training(bad,x,expected)
        with self.assertRaises(ValueError): validate_training(rows+[rows[0]],np.vstack([x,x[:1]]),expected)

    def test_additive_evidence_matches_classifier_logit(self):
        rec,_,_=build_record(fixture())
        rng=np.random.default_rng(31);h=rng.normal(size=(len(rec["route"]),256))
        x=typed_context(rec,h)
        m=load_model(ROOT/"models/all494/supervised_head.npz")
        result=contributions(rec,h,m)
        self.assertAlmostEqual(result["logit"],float(predict(m,x[None])[0]),places=8)
        self.assertAlmostEqual(result["assembly_context_logit_term"]+sum(r["logit_contribution"] for r in result["objects"]),result["logit"],places=8)

    def test_logistic_gradient_and_save_reload(self):
        rng=np.random.default_rng(5);x=rng.normal(size=(40,7));y=(x[:,0]>.1).astype(float)
        theta=rng.normal(size=8);value,grad=objective(theta,x,y,10.)
        eps=1e-6
        numerical=np.array([(objective(theta+np.eye(8)[j]*eps,x,y,10.)[0]-objective(theta-np.eye(8)[j]*eps,x,y,10.)[0])/(2*eps) for j in range(8)])
        self.assertTrue(np.allclose(grad,numerical,atol=1e-7))
        model,diag=fit(x,y,10.)
        self.assertLess(diag["attempts"][-1]["gradient_inf"],2e-5)
        with tempfile.TemporaryDirectory() as d:
            save_model(Path(d)/"m.npz",model)
            self.assertTrue(np.array_equal(predict(model,x),predict(load_model(Path(d)/"m.npz"),x)))

class TorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try: import torch
        except ImportError: raise unittest.SkipTest("PyTorch unavailable")
        torch.set_num_threads(1)

    def test_masked_features_cannot_leak(self):
        import torch
        from neuralps.assembly_model import AssemblyEncoder,Config,collate_records
        torch.manual_seed(5);model=AssemblyEncoder(Config(dropout=0)).eval()
        record,parents,sequences=build_record(fixture())
        rng=np.random.default_rng(17);vectors={h:rng.normal(size=1152).astype(np.float32) for h in sequences}
        batch=collate_records([record],vectors.__getitem__)
        plan=make_plan(record,[4],parents)
        for j,k in plan["hidden"]:batch["state"][0,j,k]=4;batch["x"][0,j,k]=float("nan")
        other={k:v.clone() for k,v in batch.items()}
        other["x"][other["state"]==4]=1e20
        for local in (False,True):
            with torch.inference_mode():a=predict_mode(model,batch,local);b=predict_mode(model,other,local)
            self.assertTrue(torch.equal(a,b))

    def test_centered_score_and_degenerate_fallback(self):
        import torch
        p=torch.tensor([[1.,0.]]);t=torch.tensor([[0.,1.]]);m=torch.tensor([[0.,1.]])
        self.assertAlmostEqual(float(reconstruction_agreement(p,t,m)),0.)
        m=torch.tensor([[.2,.1]])
        expected=.25*torch.nn.functional.cosine_similarity(p,t)+.75*torch.nn.functional.cosine_similarity(p-m,t-m)
        self.assertAlmostEqual(float(reconstruction_agreement(p,t,m)),float(expected[0]),places=6)

    def test_npz_runtime_and_joint_scores(self):
        import torch
        from neuralps.assembly_model import AssemblyEncoder,Config
        torch.manual_seed(9);cfg=Config(dropout=0);model=AssemblyEncoder(cfg)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            np.savez_compressed(root/"encoder_weights.npz",**{k:v.detach().numpy() for k,v in model.state_dict().items()})
            np.savez_compressed(root/"target_means.npz",m0=np.zeros(1152,dtype=np.float32))
            dump(root/"encoder_config.json",cfg.__dict__)
            dump(root/"target_mean_map.json",{k:["m0",100] for k in ["D","C","B:cterm","B:nterm"]})
            dump(root/"manifest.json",dict(source_checkpoint_sha256=SSL_SHA,selected_step=26000,
                files_sha256={p.name:sha(p) for p in root.iterdir()}))
            runtime=Runtime(root)
            rec,parents,sequences=build_record(fixture())
            rng=np.random.default_rng(8);vectors={h:rng.normal(size=1152).astype(np.float32) for h in sequences}
            result=runtime.score(rec,parents,vectors,[4])
            self.assertEqual(result["joint_region"]["status"],"SCORED")
            self.assertTrue(all(-1<=x<=1 for mode in result["joint_region"]["aggregate"].values() for x in mode.values()))
            (root/"encoder_config.json").write_text("{}")
            with self.assertRaises(ValueError):Runtime(root)

class ExportTests(unittest.TestCase):
    def test_preserved_fork_wheel_is_installable_and_retains_license(self):
        import importlib.util
        import subprocess
        import sys
        from neuralps.contracts import FORK_COMMIT
        spec=importlib.util.spec_from_file_location("release_export_test",ROOT/"scripts/export_runtime.py")
        exporter=importlib.util.module_from_spec(spec);spec.loader.exec_module(exporter)
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);installed=base/"installed";out=base/"out";out.mkdir()
            source={"transformers/__init__.py":"__version__ = '4.57.6'\n",
                    "transformers/models/esmc/modeling_esmc.py":"# preserved source\n",
                    "transformers-4.57.6.dist-info/METADATA":"Metadata-Version: 2.1\nName: transformers\nVersion: 4.57.6\n",
                    "transformers-4.57.6.dist-info/licenses/LICENSE":"Test license retained unchanged\n"}
            for name,text in source.items():
                p=installed/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
            class Dist:
                version="4.57.6"
                files=list(source)
                def read_text(self,name):return json.dumps({"vcs_info":{"commit_id":FORK_COMMIT}})
                def locate_file(self,name):return installed/name
            with patch.object(exporter.importlib.metadata,"distribution",return_value=Dist()):
                manifest=exporter.export_transformers_wheel(out)
            target=base/"target"
            subprocess.run([sys.executable,"-m","pip","install","--no-deps","--no-compile","--quiet","--disable-pip-version-check","--root-user-action=ignore","--target",str(target),str(out/manifest["wheel"])],check=True,capture_output=True)
            for name,text in source.items():self.assertEqual((target/name).read_text(),text)

if __name__ == "__main__":unittest.main()
