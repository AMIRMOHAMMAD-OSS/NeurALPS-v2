"""Frontend regressions for score modes and honest replacement comparison.

Callbacks use deterministic fixture values: no model quality or biology is
inferred from these tests. Production frontend and Colab mount scripts execute
unchanged using the same isolated-document harness as test_colab_embedding.
"""
import copy
import math
import unittest

from test_colab_embedding import NODE_HARNESS, capture_mount, payload, run_javascript


def supervised_payload(assembly_id="original engineered assembly"):
    data = payload(assembly_id)
    contributions = [-.4321, .0234, .2012, -.0654, .0942]
    logit = math.log(.775/.225)
    data["result"]["supervised"] = dict(
        activity_score=.775, logit=logit,
        assembly_context_logit_term=logit-sum(contributions),
        objects=[dict(object_index=i, logit_contribution=value)
                 for i, value in enumerate(contributions)],
        calibrated_probability=False)
    data["can_test_variants"] = True
    return data


def responses(data):
    targets = [0, 1, 2]
    segments = {mode+":0,1,2": dict(status="SCORED", mode=mode, score=score,
                                    targets=targets, additional_hidden_objects=[],
                                    aggregation="fixture joint mask", masked_fraction=.3)
                for mode, score in (("local", .41), ("full", .61))}
    spec = dict(assembly_id="engineered replacement", proteins=[])
    replacement = dict(candidate_id="donor-TC", base_assembly_id=data["assembly_id"],
                       variant_assembly_id=spec["assembly_id"], selected_object_indices=targets,
                       replaced_object_indices=[0, 2], affected_object_indices=[0, 1, 2, 3])
    preview_objects = copy.deepcopy(data["objects"])
    for obj in preview_objects:
        obj.update(raw=None, status="NOT_RETESTED")
    preview = dict(spec=spec, replacement=replacement, preview_objects=preview_objects,
                   preview_assembly_id=spec["assembly_id"])
    candidate = dict(candidate_id="donor-TC", assembly_id="natural donor", score=.83,
                     first_object=0, last_object=2,
                     sequences=[dict(role="domain", object_index=0, label="PCP", sequence="AAAA")])
    ranking = dict(mode="local", incumbent_score=.41, comparison_tolerance=1e-6,
                   targets=targets,
                   repertoire_percentile_0_to_100=25.,
                   counts=dict(eligible_unique=1, higher=1, equal=0, lower=0,
                               identical_to_query=0, duplicate_fragments=0,
                               visible_sequence_alias=0, missing_features=0),
                   corpus={}, top_candidates=[candidate])
    variant = supervised_payload(spec["assembly_id"])
    variant["replacement"] = replacement
    variant["initial_selection"] = targets
    variant["input_spec"] = spec
    variant["result"]["supervised"]["activity_score"] = .882
    variant["result"]["supervised"]["objects"][0]["logit_contribution"] = -.1113
    for i, obj in enumerate(variant["objects"]):
        obj["raw"] = dict(local=.61+i*.03, full=.73+i*.04)
    variant_segments = {mode: dict(segments[mode+":0,1,2"], score=score)
                        for mode, score in (("local", .79), ("full", .87))}
    tested = dict(spec=spec, replacement=replacement, variant_payload=variant,
                  base_segments={mode: segments[mode+":0,1,2"] for mode in ("local", "full")},
                  variant_segments=variant_segments,
                  base_activity_score=.775, variant_activity_score=.882)
    return dict(score=dict(segments=segments),
                rank=dict(segments=segments, key="local:0,1,2", ranking=ranking),
                variant=preview, test_variant=tested)


VIEW_HARNESS = NODE_HARNESS.split("(async()=>{", 1)[0] + r"""
w.google.colab.kernel.invokeFunction = async(name,args)=>{
  const request=JSON.parse(JSON.stringify(args[0])); calls.push({name,request});
  assert(fixture.responses[request.action], 'Unexpected action '+request.action);
  return {data:{'application/json':{ok:true,value:fixture.responses[request.action]}}};
};
const values = (root, selector) => [...root.querySelectorAll(selector)].map(part=>{
  const value=part.getAttribute('data-score');
  return value===null || value==='' ? null : Number(value);
});
const baseValues = root => values(root,'#assemblysvg .part');
const variantValues = root => values(root,'#variantsvg .variant-part');
function setView(root,mode) {
  by(root,'view-'+mode).click();
  assert.equal(by(root,'assemblysvg').dataset.viewmode,mode);
}
function assertModelValues(root,data,mode,variant=false) {
  const actual=variant?variantValues(root):baseValues(root);
  const expected=mode==='supervised'
    ?data.objects.map(o=>data.result.supervised.objects.find(s=>s.object_index===o.object_index)?.logit_contribution??null)
    :data.objects.map(o=>o.raw?.[mode]??null);
  assert.deepEqual(actual,expected, 'Map values must come from '+mode+' data in the '+(variant?'variant':'original')+' payload');
}
async function savedCopy(root) {
  by(root,'savehtml').click();
  const html=await downloads.at(-1).blob.text();
  const savedDom=new JSDOM(html,{runScripts:'dangerously',virtualConsole:vc,beforeParse:instrument});
  const doc=savedDom.window.document;
  const data=JSON.parse(doc.querySelector('#neuralps-data').textContent);
  assert.equal(JSON.parse(doc.querySelector('#neuralps-bridge').textContent),null);
  assert.equal(doc.querySelector('#connection').textContent,'Saved result');
  return {savedDom,doc,data};
}
(async()=>{
  const root=await mount(fixture.mounts[0]), original=fixture.payloads[0];
  assertMounted(root,original);
  assert.equal(calls.length,0);
  if (fixture.scenario==='modes') {
    select(root,0,2);
    const selection=by(root,'selected-route').textContent;
    for (const mode of ['local','full','supervised']) {
      setView(root,mode);
      assertModelValues(root,original,mode);
      assert.equal(root.querySelectorAll('#assemblysvg .part').length,original.objects.length);
      assert.equal(by(root,'selected-route').textContent,selection,'View changes preserve the selected region');
    }
    assert(by(root,'activity-score').textContent.includes('0.775'),'Whole-assembly activity score remains 0.775');
    assert(by(root,'map-summary').textContent.toLowerCase().includes('logit'),'Contributions must be labeled in logit units');
    assert(baseValues(root)[0]<0,'Negative supervised evidence must keep its sign');
    const parts=root.querySelectorAll('#assemblysvg .part');
    assert.notEqual(parts[0].querySelector('.outline').getAttribute('fill'),parts[2].querySelector('.outline').getAttribute('fill'),'Negative and positive contributions use different colors');
    assert.equal(calls.length,0,'Changing score view must not invoke the model');
  } else if (fixture.scenario==='no-head') {
    by(root,'view-supervised').click();
    if (!by(root,'view-supervised').disabled) {
      assert.equal(by(root,'assemblysvg').dataset.viewmode,'supervised');
      assert(baseValues(root).every(value=>value===null),'Missing head must not produce zero or raw compatibility contributions');
    }
    assert(!root.body.textContent.includes('0.775'),'No activity score is invented without a head');
  } else if (fixture.scenario==='backend-restore') {
    assert(!by(root,'variantpanel').hidden,'Backend-persisted preview restores on mount');
    assert(variantValues(root).every(value=>value===null));
    assert.equal(root.querySelectorAll('#variantsvg [data-replaced="true"]').length,2);
    assert(by(root,'variantcaption').textContent.includes('natural donor'),'Restore resolves the stored candidate identity');
    select(root,4,4);
    by(root,'retest-preview').click(); await finish(root);
    assert.equal(calls.length,1);
    assert.equal(calls[0].request.action,'test_variant');
    assert.equal(calls[0].request.candidate_id,'donor-TC');
    assert.deepEqual(calls[0].request.indices,[0,1,2],'Retest uses persisted replacement targets, not the current selection');
    assert.equal(calls[0].request.touching_boundaries,false);
    assert.equal(calls[0].request.mode,'full','Retest retains the stored search context');
    assertModelValues(root,original,'local');
    assertModelValues(root,fixture.responses.test_variant.variant_payload,'local',true);
  } else {
    select(root,0,2);
    by(root,'touching').checked=false;
    by(root,'touching').dispatchEvent(new root.defaultView.Event('change'));
    by(root,'score').click(); await finish(root);
    assert.deepEqual(calls.at(-1).request.indices,[0,1,2]);
    assert.equal(calls.at(-1).request.action,'score');
    assert.equal(calls.at(-1).request.touching_boundaries,false);
    by(root,'rank').click(); await finish(root);
    assert.equal(calls.at(-1).request.action,'rank');
    assert.equal(calls.at(-1).request.mode,'local');
    assert.deepEqual(calls.at(-1).request.indices,[0,1,2]);
    const previewButton=root.querySelector('[data-preview="0"]');
    assert(previewButton,'Each candidate must offer a structural preview');
    previewButton.click(); await finish(root);
    assert.equal(calls.at(-1).request.action,'variant');
    assert.equal(calls.at(-1).request.candidate_id,'donor-TC');
    assert(!by(root,'variantpanel').hidden,'Replacement preview must be visible');
    assert.equal(root.querySelectorAll('#variantsvg .variant-part').length,original.objects.length);
    assert(variantValues(root).every(value=>value===null),'Preview cannot fabricate compatibility before retest');
    assertModelValues(root,original,'local');
    if (fixture.scenario==='preview') {
      const saved=await savedCopy(root);
      assert(saved.data.variant_preview,'Preview persists in offline export');
      assert(!by(saved.doc,'variantpanel').hidden);
      assert(variantValues(saved.doc).every(value=>value===null));
      assertModelValues(saved.doc,original,'local');
      saved.savedDom.window.close();
    } else {
      by(root,'retest-preview').click(); await finish(root);
      assert.equal(calls.at(-1).request.action,'test_variant');
      assert.equal(calls.at(-1).request.candidate_id,'donor-TC');
      const variant=fixture.responses.test_variant.variant_payload;
      for(const mode of ['local','full','supervised']) {
        setView(root,mode);
        assertModelValues(root,original,mode);
        assertModelValues(root,variant,mode,true);
      }
      assert(by(root,'variantpanel').textContent.includes('0.882'),'Comparison shows retested whole-assembly activity');
      const saved=await savedCopy(root);
      assert(saved.data.variant_preview.variant_payload,'Tested replacement payload survives export');
      setView(saved.doc,'supervised');
      assertModelValues(saved.doc,variant,'supervised',true);
      assertModelValues(saved.doc,original,'supervised');
      saved.savedDom.window.close();
    }
  }
  assert.deepEqual(errors,[]);
  dom.window.close(); process.stdout.write(JSON.stringify({status:'PASS'}));
})().catch(e=>{console.error(e.stack);dom.window.close();process.exit(1)});
"""


class ExplorerViewTests(unittest.TestCase):
    def run_view(self, scenario):
        data = supervised_payload()
        if scenario == "no-head":
            data["result"].pop("supervised")
        replies = responses(data)
        if scenario == "backend-restore":
            data["variant_preview"] = copy.deepcopy(replies["variant"])
            data["candidate_results"] = {"full:0,1,2": dict(replies["rank"]["ranking"], mode="full")}
            data["segment_results"] = replies["score"]["segments"]
        fixture = dict(payloads=[data], mounts=[capture_mount(data)[0]],
                       scenario=scenario, responses=replies)
        run_javascript(self, fixture, VIEW_HARNESS)

    def test_local_global_and_signed_supervised_views_keep_distinct_values(self):
        self.run_view("modes")

    def test_missing_supervised_head_does_not_invent_contributions(self):
        self.run_view("no-head")

    def test_unscored_preview_and_offline_export_have_no_fabricated_values(self):
        self.run_view("preview")

    def test_retested_replacement_uses_actual_payload_and_keeps_original_map(self):
        self.run_view("tested")

    def test_backend_persisted_preview_restores_its_original_retest_context(self):
        self.run_view("backend-restore")


if __name__ == "__main__":
    unittest.main()
