"""Scientific and session regressions for local/global segment workflows."""
import copy
import json
import unittest
from unittest.mock import patch
from neuralps.explorer import ExplorerSession, input_from_saved, render_html
from neuralps.segments import score_segment, score_segment_modes
import test_explorer
from test_release import fixture


class CompatibilityWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def setUp(self):
        self.runtime, self.record, self.parents, self.vectors = test_explorer.CandidateTests().runtime_fixture()
        self.session = ExplorerSession(fixture(), self.record, self.parents,
                                       self.vectors, self.runtime)

    def test_both_contexts_use_same_joint_mask_and_original_scores(self):
        pair = score_segment_modes(self.runtime, self.record, self.parents, self.vectors,
                                   [4, 5, 6], touching_boundaries=True)
        self.assertEqual(pair['local']['targets'], [3, 4, 5, 6, 7])
        self.assertEqual(pair['local']['hidden'], pair['full']['hidden'])
        for mode in ('local', 'full'):
            old = score_segment(self.runtime, self.record, self.parents, self.vectors,
                                [4, 5, 6], mode, True)
            self.assertAlmostEqual(pair[mode]['score'], old['score'], places=7)
            self.assertIsNone(pair[mode]['natural_percentile_0_to_100'])
            mapped = self.session.result['pretrained_map'][4]['raw'][mode]
            self.assertNotAlmostEqual(pair[mode]['objects']['4'][mode], mapped, places=5)

    def test_one_request_computes_and_caches_both_context_modes(self):
        request = dict(action='score', indices=[4, 5, 6], mode='local', touching_boundaries=True)
        with patch.object(self.runtime, 'score_plans', wraps=self.runtime.score_plans) as spy:
            first = self.session.handle(request)
            second = self.session.handle(dict(request, mode='full'))
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(set(first['segments']), {'local:3,4,5,6,7', 'full:3,4,5,6,7'})
        self.assertEqual(first['segments'], second['segments'])
        self.assertEqual(second['segment']['mode'], 'full')

    def test_gapped_search_rejected_before_downloading_bank(self):
        with patch.object(self.session, 'repertoire_factory') as factory:
            with self.assertRaisesRegex(ValueError, 'continuous range'):
                self.session.handle(dict(action='rank', indices=[2, 6], mode='local'))
        factory.assert_not_called()
        self.assertIsNone(self.session.repertoire)

    def test_unscoreable_selection_has_no_invented_values(self):
        result = self.session.handle(dict(action='score', indices=list(range(len(self.record['route'])))))
        for value in result['segments'].values():
            self.assertEqual(value['status'], 'INSUFFICIENT_VISIBLE_CONTEXT')
            self.assertIsNone(value['score'])
            self.assertIsNone(value['natural_percentile_0_to_100'])

    def test_saved_html_and_json_restore_input_without_executing_html(self):
        payload = self.session.payload()
        payload['input_spec']['assembly_id'] = '</script><script>throw Error("should not run")</script>'
        for text in (json.dumps(payload), render_html(payload), json.dumps(payload['input_spec'])):
            restored = input_from_saved(text)
            self.assertEqual(restored, payload['input_spec'])
        self.assertNotIn('"assembly_id": "</script>', render_html(payload))

    def test_old_html_without_input_requests_original_sequence(self):
        old = self.session.payload()
        old.pop('input_spec')
        with self.assertRaisesRegex(ValueError, 'no input sequence'):
            input_from_saved(render_html(old))

    def test_new_assembly_preserves_lock_reference_and_downloaded_bank(self):
        self.session.assets = 'assets'; self.session.model_dir = 'esmc'
        self.session.repertoire = object()
        bank, lock = self.session.repertoire, self.session.lock
        new_spec = fixture(); new_spec['assembly_id'] = 'new engineered assembly'
        new_record = copy.deepcopy(self.record); new_record['assembly_id'] = new_spec['assembly_id']
        replacement = ExplorerSession(new_spec, new_record, self.parents, self.vectors, self.runtime)
        with patch.object(ExplorerSession, 'from_spec', return_value=replacement):
            result = self.session.handle(dict(action='load_assembly', spec=new_spec))
        self.assertEqual(result['payload']['assembly_id'], new_spec['assembly_id'])
        self.assertIs(self.session.repertoire, bank)
        self.assertIs(self.session.lock, lock)
        self.assertFalse(lock.locked())
        self.assertEqual(self.session.handle({'action':'state'})['input_spec'], new_spec)

    def test_failed_input_load_keeps_previous_assembly(self):
        self.session.assets = 'assets'; self.session.model_dir = 'esmc'
        before = self.session.spec
        with patch.object(ExplorerSession, 'from_spec', side_effect=ValueError('invalid sequence')):
            with self.assertRaisesRegex(ValueError, 'invalid sequence'):
                self.session.handle(dict(action='load_assembly', spec=fixture()))
        self.assertIs(self.session.spec, before)
        self.assertFalse(self.session.lock.locked())


if __name__ == '__main__':
    unittest.main()
