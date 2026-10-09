import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from neuralps.contracts import SSL_SHA, sha, seqsha
from neuralps.inputs import build_record
from neuralps.reference import reference_row, NaturalReference
from neuralps.reference_scale_original import keys, canonical
from neuralps.scoring import make_plan, reconstruction_agreement, slots_by_object
from neuralps.segments import selection, object_weights, candidate_agreements, masked_predictions, score_segment
from neuralps.repertoire import Repertoire, signature
from neuralps.explorer import render_html
from test_release import fixture


class ReferenceTests(unittest.TestCase):
    def test_original_strata_and_midrank_ties(self):
        rec, parents, _ = build_record(fixture())
        row = reference_row(rec, parents, 4, .5)
        self.assertEqual(row["domain_types"], [2])
        self.assertEqual(row["mode"], "object_recovery")
        self.assertEqual(row["slots"][0]["length_bin"], 6)
        protocol = dict(closure_fraction_edges=[.25,.5], minimum_reference_components=100,
            very_low_minimum_components=500, low_tail_threshold=.05, very_low_tail_threshold=.01)
        key = canonical(list(keys(row, protocol)[0]))
        ref = NaturalReference.__new__(NaturalReference)
        ref.protocol, ref.audit = protocol, {}
        ref.reference = {"groups":{key:dict(n=100,candidate_objects=700,
            scores=[.1]*30+[.5]*20+[.9]*50,
            representatives=[dict(native_hashes=[],parent_hashes=[]) for _ in range(100)])}}
        result = ref.apply(rec, parents, 4, .5)
        self.assertEqual(result["natural_percentile_0_to_100"], 40)
        self.assertEqual((result["reference_lower"],result["reference_equal"],result["reference_higher"]),(30,20,50))
        self.assertAlmostEqual(result["lower_tail_rank"],51/101)
        self.assertEqual(ref.apply(rec,parents,4,.5,mode="local")["scale_status"],"UNSUPPORTED_CONTEXT_MODE")
        ref.reference["groups"][key]["n"]=99
        self.assertIsNone(ref.apply(rec,parents,4,.5)["natural_percentile_0_to_100"])

    def test_break_uses_two_terminal_lengths_and_flank_types(self):
        spec = fixture(); p=copy.deepcopy(spec["proteins"][0]);p["id"]="other";spec["proteins"].append(p)
        rec,parents,_=build_record(spec)
        j=next(j for j,o in enumerate(rec["route"]) if o["kind"]==2)
        row=reference_row(rec,parents,j,.4)
        self.assertEqual(row["mode"],"joint_terminal_recovery")
        self.assertEqual(len(row["slots"]),2)
        self.assertEqual(row["domain_types"],[rec["route"][j-1]["domain_type"],rec["route"][j+1]["domain_type"]])

    def test_arbitrary_selection_and_explicit_touching_boundaries(self):
        rec,parents,_=build_record(fixture())
        self.assertEqual(selection(rec,[1]),[1])
        self.assertEqual(selection(rec,[4,6],True),[3,4,5,6,7])
        self.assertEqual(selection(rec,[0,5]),[0,5])
        self.assertEqual(make_plan(rec,list(range(len(rec["route"]))),parents)["status"],"INSUFFICIENT_VISIBLE_CONTEXT")
        for indices in ([],[True],[-1],[999],[1.0]):
            with self.assertRaises(ValueError):selection(rec,indices)

    def test_embedded_html_escapes_script_termination(self):
        html=render_html({"assembly_id":"</script><script>alert(1)</script>","objects":[]})
        self.assertNotIn('"assembly_id": "</script>',html)
        self.assertIn('\\u003c/script>',html)


class CandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def test_vectorized_ranking_formula_matches_original_torch_loss(self):
        import torch
        rng=np.random.default_rng(72)
        p=rng.normal(size=(3,1152)).astype(np.float32)
        m=rng.normal(size=p.shape).astype(np.float32)
        values=rng.normal(size=(7,3,1152)).astype(np.float32)
        values[0,1]=m[1]  # exact near-zero centered-target fallback
        weights=np.array([.5,.25,.25])  # domain once, two break termini once
        got=candidate_agreements(p,m,values,weights)
        expected=[]
        for v in values:
            scores=[float(reconstruction_agreement(torch.from_numpy(p[i:i+1]),torch.from_numpy(v[i:i+1]),torch.from_numpy(m[i:i+1]))) for i in range(3)]
            expected.append(np.dot(weights,scores))
        self.assertTrue(np.allclose(got,expected,atol=2e-7,rtol=0))

    def runtime_fixture(self):
        import torch
        from neuralps.assembly_model import AssemblyEncoder,Config,collate_records
        from neuralps.scoring import Runtime
        torch.manual_seed(38)
        runtime=Runtime.__new__(Runtime)
        runtime.model=AssemblyEncoder(Config(dropout=0)).eval().requires_grad_(False)
        runtime.device="cpu";runtime.collate=collate_records
        runtime.means={"m":np.zeros(1152,dtype=np.float32)}
        runtime.mean_map={k:["m",100] for k in ("D","C","B:cterm","B:nterm")}
        rec,parents,seqs=build_record(fixture())
        rng=np.random.default_rng(69)
        vectors={h:rng.normal(size=1152).astype(np.float32) for h in seqs}
        return runtime,rec,parents,vectors

    def test_selection_incumbent_matches_joint_model_output(self):
        runtime,rec,parents,vectors=self.runtime_fixture()
        for mode in ("full","local"):
            for targets in ([1],[4],[3,4,5],[4,5,6]):
                scored=score_segment(runtime,rec,parents,vectors,targets,mode)
                p,m,v,w,_=masked_predictions(runtime,rec,parents,vectors,targets,mode)
                self.assertAlmostEqual(scored["score"],float(candidate_agreements(p,m,v[None],w)[0]),places=6)
        unscored=score_segment(runtime,rec,parents,vectors,list(range(len(rec["route"]))))
        self.assertEqual(unscored["status"],"INSUFFICIENT_VISIBLE_CONTEXT")
        self.assertIsNone(unscored["score"])

    def bank(self, directory, record, values, entries):
        root=Path(directory)
        db=sqlite3.connect(root/"repertoire.sqlite3")
        db.executescript('CREATE TABLE features(hash TEXT PRIMARY KEY,shard INTEGER,row_index INTEGER,sequence TEXT);CREATE TABLE parents(hash TEXT PRIMARY KEY,sequence TEXT);CREATE TABLE objects(assembly_id TEXT,j INTEGER,signature TEXT,object_json TEXT,slots_json TEXT,PRIMARY KEY(assembly_id,j));')
        for row,(h,(seq,vec)) in enumerate(values.items()):
            db.execute('INSERT INTO features VALUES(?,?,?,?)',(h,0,row,seq))
        np.stack([v for _,v in values.values()]).astype('<f4').tofile(root/'vectors-000.f32')
        for aid,h,kind in entries:
            seq=values[h][0];ph=seqsha(seq)
            db.execute('INSERT OR IGNORE INTO parents VALUES(?,?)',(ph,seq))
            obj=dict(kind=0,domain_type=kind,object_index=0,protein_uid=aid,canonical_domain_type='AMP-binding',start_aa_0based=0,end_aa_0based_exclusive=len(seq))
            slots=[dict(object_index=0,slot=0,kind=0,state='OBSERVED',sequence_hash=h,parent_sequence_hash=ph,start=0,end=len(seq),length=len(seq),protein_uid=aid,group='D:'+str(kind))]
            sig=json.dumps([0,[kind]],separators=(',',':'))
            db.execute('INSERT INTO objects VALUES(?,?,?,?,?)',(aid,0,sig,json.dumps(obj),json.dumps(slots)))
        db.commit();db.close()
        (root/'manifest.json').write_text(json.dumps(dict(schema='neuralps_natural_repertoire_v1',checkpoint_sha256=SSL_SHA,database_sha256=sha(root/'repertoire.sqlite3'),natural_assemblies_by_split={'train':len(entries)},shards=[dict(path='vectors-000.f32',rows=len(values),sha256=sha(root/'vectors-000.f32'),archive='test.zip')])) )
        return Repertoire(root)

    def test_exact_counts_deduplication_alias_exclusion_and_variant_splicing(self):
        runtime,rec,parents,vectors=self.runtime_fixture()
        p,m,current,w,visible=masked_predictions(runtime,rec,parents,vectors,[4],'full')
        by=slots_by_object(rec);own=by[4][0]['sequence_hash'];vh=next(iter(visible))
        ownseq=parents[by[4][0]['parent_sequence_hash']][by[4][0]['start']:by[4][0]['end']]
        values={own:(ownseq,current[0]),vh:('V'*80,vectors[vh])}
        entries=[('self',own,2),('alias',vh,2)]
        for name,seq,v in [('higher','A'*80,p[0]),('lower','C'*80,-p[0]),('equal','D'*80,current[0])]:
            h=seqsha(seq);values[h]=(seq,v);entries.append((name,h,2))
        entries.append(('higher_duplicate',seqsha('A'*80),2))
        entries.append(('wrong_type',seqsha('C'*80),3))
        with tempfile.TemporaryDirectory() as d:
            bank=self.bank(d,rec,values,entries)
            result=bank.rank(runtime,rec,parents,vectors,[4])
            self.assertEqual({k:result['counts'][k] for k in ('eligible_unique','higher','equal','lower')},dict(eligible_unique=3,higher=1,equal=1,lower=1))
            self.assertEqual(result['counts']['duplicate_fragments'],1)
            self.assertEqual(result['counts']['visible_sequence_alias'],1)
            self.assertEqual(result['repertoire_percentile_0_to_100'],50)
            top=result['top_candidates'][0]
            self.assertEqual(top['sequences'][0]['sequence'],'A'*80)
            variant=bank.variant_spec(fixture(),rec,[4],top['candidate_id'])
            old=fixture()['proteins'][0];new=variant['proteins'][0]
            self.assertEqual(len(new['sequence']),len(old['sequence'])-10)
            self.assertEqual(new['sequence'][360:440],'A'*80)
            self.assertEqual(new['domains'][3]['start'],old['domains'][3]['start']-10)
            rebuilt,_,_=build_record(variant)
            self.assertEqual(len(rebuilt['route']),len(rec['route']))
            for j in (3,5):self.assertNotEqual(rebuilt['route'][j]['seq_hash'],rec['route'][j]['seq_hash'])
            with self.assertRaises(ValueError):bank.rank(runtime,rec,parents,vectors,[2,4])


if __name__ == '__main__':
    unittest.main()
