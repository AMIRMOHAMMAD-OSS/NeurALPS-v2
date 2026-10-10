"""Replacement previews expose rebuilt geometry and only genuine retest scores."""
import copy
import json
import unittest
from unittest.mock import Mock, patch

from neuralps.explorer import ExplorerSession
from neuralps.inputs import build_record
from test_release import fixture


class FixedRuntime:
    """Known values for testing response ownership; no scientific inference."""
    def __init__(self, score):
        self.value = score

    def score(self, record, parents, vectors, joint_domain_indices=None):
        return dict(assembly_id=record['assembly_id'], pretrained_map=[
            dict(object_index=j, kind=obj['kind'], status='SCORED',
                 raw=dict(local=self.value, full=self.value + .1))
            for j, obj in enumerate(record['route'])])


class VariantPreviewTests(unittest.TestCase):
    def setUp(self):
        self.spec = fixture()
        record, parents, _ = build_record(self.spec)
        self.session = ExplorerSession(self.spec, record, parents, {}, FixedRuntime(.2),
                                       model_dir='esmc', assets='assets')
        self.session.result['source_commit'] = 'test-source'
        self.session.result['supervised'] = dict(activity_score=.3, objects=[
            dict(object_index=4, logit_contribution=-.2)])
        self.candidate_id = ['natural_donor', 8, 1]
        self.request = dict(action='variant', indices=[4], mode='local',
                            candidate_id=self.candidate_id)
        self.session.segments = {
            mode + ':4': dict(status='SCORED', mode=mode, targets=[4], score=.2)
            for mode in ('local', 'full')}
        self.session.searches['local:4'] = dict(top_candidates=[
            dict(candidate_id=self.candidate_id)])
        self.variant = copy.deepcopy(self.spec)
        self.variant['assembly_id'] = 'fixture__donor'
        protein = self.variant['proteins'][0]
        start, end = protein['domains'][2]['start'], protein['domains'][2]['end']
        protein['sequence'] = protein['sequence'][:start] + 'W' * (end-start+13) + protein['sequence'][end:]
        protein['domains'][2]['end'] += 13
        for domain in protein['domains'][3:]:
            domain['start'] += 13
            domain['end'] += 13
        self.session.repertoire = Mock()
        self.session.repertoire.variant_spec.return_value = self.variant

    def test_preview_rebuilds_coordinates_without_model_calls_or_old_scores(self):
        original = copy.deepcopy(self.session.result)
        with patch.object(ExplorerSession, 'from_spec') as embed, \
             patch('neuralps.explorer.score_segment_modes') as scorer:
            value = self.session.handle(self.request)
        embed.assert_not_called()
        scorer.assert_not_called()
        self.assertEqual(value['replacement']['replaced_object_indices'], [4])
        self.assertEqual(value['replacement']['affected_object_indices'], [3, 4, 5])
        self.assertEqual(value['replacement']['selected_object_indices'], [4])
        self.assertEqual(value['preview_objects'][4]['spans'][0]['end'],
                         self.spec['proteins'][0]['domains'][2]['end']+13)
        self.assertEqual(value['preview_objects'][6]['spans'][0]['start'],
                         self.spec['proteins'][0]['domains'][3]['start']+13)
        self.assertTrue(all(obj['status'] == 'NOT_RETESTED' and obj['raw'] is None
                            for obj in value['preview_objects']))
        self.assertEqual(self.session.result, original)
        self.assertEqual(self.session.spec, self.spec)
        self.assertEqual(self.session.payload()['variant_preview'], value)
        self.assertFalse(self.session.lock.locked())
        json.dumps(self.session.payload(), allow_nan=False)

    def test_retest_returns_new_maps_and_supervised_results_preserving_original(self):
        record, parents, _ = build_record(self.variant)
        rebuilt = ExplorerSession(self.variant, record, parents, {}, FixedRuntime(.7),
                                   model_dir='esmc', assets='assets')
        rebuilt.result['supervised'] = dict(activity_score=.8, objects=[
            dict(object_index=4, logit_contribution=.9)])
        joint_scores = {mode: dict(status='SCORED', mode=mode, targets=[4], score=.6)
                        for mode in ('local', 'full')}
        original_result = copy.deepcopy(self.session.result)
        with patch.object(ExplorerSession, 'from_spec', return_value=rebuilt) as embed, \
             patch('neuralps.explorer.score_segment_modes', return_value=joint_scores) as scorer:
            value = self.session.handle(dict(self.request, action='test_variant'))
        embed.assert_called_once_with(self.variant, 'assets', 'esmc', None, None, None)
        scorer.assert_called_once_with(rebuilt.runtime, record, parents, {}, [4])
        payload = value['variant_payload']
        self.assertEqual(payload['assembly_id'], self.variant['assembly_id'])
        self.assertEqual(payload['input_spec'], self.variant)
        self.assertEqual(payload['objects'][4]['raw']['local'], .7)
        self.assertEqual(payload['result']['supervised']['objects'][0]['logit_contribution'], .9)
        self.assertEqual(payload['result']['source_commit'], 'test-source')
        self.assertEqual(payload['initial_selection'], [4])
        self.assertEqual(payload['segment_results']['local:4']['score'], .6)
        self.assertEqual(payload['replacement'], value['replacement'])
        self.assertEqual(value['base_activity_score'], .3)
        self.assertEqual(value['variant_activity_score'], .8)
        self.assertEqual(self.session.result, original_result)
        self.assertIs(self.session.spec, self.spec)
        self.assertEqual(self.session.payload()['variant_preview']['variant_payload'], payload)
        self.assertFalse(self.session.lock.locked())
        json.dumps(self.session.payload(), allow_nan=False)

    def test_mismatched_topology_stops_before_reembedding(self):
        self.variant['proteins'][0]['domains'][2]['type'] = 'PCP'
        with patch.object(ExplorerSession, 'from_spec') as embed:
            with self.assertRaisesRegex(ValueError, 'topology'):
                self.session.handle(dict(self.request, action='test_variant'))
        embed.assert_not_called()
        self.assertIsNone(self.session.variant_preview)
        self.assertFalse(self.session.lock.locked())

    def test_unranked_candidate_cannot_be_previewed(self):
        with self.assertRaisesRegex(ValueError, 'Rank candidates'):
            self.session.handle(dict(self.request, candidate_id=['unknown', 0, 1]))
        self.session.repertoire.variant_spec.assert_not_called()
        self.assertIsNone(self.session.variant_preview)

    def test_preview_without_cached_selection_never_runs_the_model(self):
        self.session.segments.clear()
        with patch('neuralps.explorer.score_segment_modes') as scorer:
            with self.assertRaisesRegex(ValueError, 'Rank candidates'):
                self.session.handle(self.request)
        scorer.assert_not_called()
        self.session.repertoire.variant_spec.assert_not_called()

    def test_existing_live_sessions_do_not_need_new_field(self):
        del self.session.variant_preview
        self.assertIsNone(self.session.payload()['variant_preview'])
        self.assertEqual(self.session.handle(self.request)['preview_assembly_id'], 'fixture__donor')


if __name__ == '__main__':
    unittest.main()
