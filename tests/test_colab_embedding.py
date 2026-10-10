"""Execute the real frontend under a notebook MIME insertion contract.

The old tests loaded a complete document with every inline script enabled. That
cannot detect an HTML-only notebook output whose data scripts are removed or
whose initialization never executes. These tests intentionally strip scripts
from HTML MIME, then run only the explicitly emitted Javascript MIME/eval_js.

Install jsdom for the optional frontend tests with ``npm install jsdom`` and set
NEURALPS_JSDOM_MODULE to its absolute module directory when it is not on NODE_PATH.
No encoder, GPU, or remote notebook session is needed for this rendering test.
"""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

from neuralps.explorer import render_html, show_explorer


def payload(assembly_id="engineered assembly"):
    parts = [(0, "PCP", "T", "square"),
             (1, "PCP → Condensation", "J", "boundary"),
             (0, "Condensation", "C", "diamond"),
             (1, "Condensation → AMP-binding", "J", "boundary"),
             (0, "AMP-binding", "A", "circle")]
    return dict(schema="neuralps_explorer_v2", assembly_id=assembly_id,
                objects=[dict(object_index=i, kind=kind, name=name, short=short,
                              shape=shape, status="SCORED",
                              raw=dict(local=0.12+i/10, full=0.32+i/10), spans=[])
                         for i, (kind, name, short, shape) in enumerate(parts)],
                result={}, reference_available=True, repertoire_available=True,
                segment_results={}, candidate_results={}, initial_selection=[0],
                input_spec=dict(assembly_id=assembly_id, proteins=[]))


class FixtureSession:
    def __init__(self, data):
        self.data = data

    def payload(self):
        return copy.deepcopy(self.data)

    def handle(self, request):
        return dict(request=request)


def capture_mount(data, ready=True):
    """Capture actual display calls, without importing IPython or Colab."""
    events, callbacks = [], {}
    display_module = types.ModuleType("IPython.display")
    for kind in ("HTML", "Javascript", "JSON"):
        def init(self, data=None, **kwargs):
            self.data = data
        setattr(display_module, kind, type(kind, (), {"__init__": init}))

    def display(*objects, **kwargs):
        for value in objects:
            events.append(dict(kind=type(value).__name__, data=value.data))
    display_module.display = display
    output = types.ModuleType("google.colab.output")
    output.register_callback = lambda name, callback: callbacks.__setitem__(name, callback)
    def eval_js(script, **kwargs):
        events.append(dict(kind="Javascript", data=script))
        return dict(ready=ready, parts=len(data["objects"]), choices=len(data["objects"]))
    output.eval_js = eval_js
    output.no_vertical_scroll = lambda: None
    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    google.colab = colab
    colab.output = output
    ipython = types.ModuleType("IPython")
    ipython.display = display_module
    with patch.dict(sys.modules, {"IPython": ipython, "IPython.display": display_module,
                                  "google": google, "google.colab": colab,
                                  "google.colab.output": output}):
        name = show_explorer(FixtureSession(data))
    return dict(events=events, callback=name), callbacks


NODE_HARNESS = r"""
const fs = require('fs'), assert = require('assert');
const {JSDOM, VirtualConsole} = require(process.env.NEURALPS_JSDOM_MODULE || 'jsdom');
const fixture = JSON.parse(fs.readFileSync(0, 'utf8'));
const errors = [], calls = [], downloads = [];
const vc = new VirtualConsole();
vc.on('jsdomError', e => { if (e.type !== 'css-parsing') errors.push(e.message); });
const dom = new JSDOM('<!doctype html><html><head></head><body><button id="outside">Notebook button</button><svg id="outside-map"></svg></body></html>', {
  runScripts: 'outside-only', pretendToBeVisual: true,
  url: 'https://colab.research.google.com/', virtualConsole: vc,
});
const w = dom.window, d = w.document;
if (fixture.failure === 'timeout') {
  const setTimeout = w.setTimeout.bind(w);
  w.setTimeout = (fn, delay, ...args) => setTimeout(fn, delay >= 15000 ? 20 : delay, ...args);
}
const blobs = new Map();
function instrument(win) {
  win.Blob = Blob;
  win.URL.createObjectURL = blob => {const url = 'blob:download/' + blobs.size; blobs.set(url, blob); return url;};
  win.URL.revokeObjectURL = () => {};
  win.HTMLAnchorElement.prototype.click = function(){ downloads.push({name:this.download, blob:blobs.get(this.href)}); };
  win.HTMLDialogElement.prototype.showModal = function(){ this.setAttribute('open', ''); };
  win.HTMLDialogElement.prototype.close = function(){ this.removeAttribute('open'); };
}
instrument(w);
w.google = {colab:{output:{setIframeHeight:()=>{}}, kernel:{invokeFunction:async (name,args)=>{
  const req = JSON.parse(JSON.stringify(args[0])); calls.push({name, request:req});
  const indices = req.indices;
  const segments = Object.fromEntries(['local','full'].map((mode,i)=>[mode+':'+indices.join(','), {
    status:'SCORED', mode, score:.41+i*.2, targets:indices,
    additional_hidden_objects:[], aggregation:'test joint mask', masked_fraction:.3,
  }]));
  return {data:{'application/json':{ok:true,value:{segments}}}};
}}}};
// jsdom has no srcdoc implementation. This shim only supplies its browser
// document-loading semantics; the production mount, readiness check, child
// script and parent-to-kernel callback run unmodified.
const pending = new Map(), loaded = new WeakSet();
Object.defineProperty(w.HTMLIFrameElement.prototype, 'srcdoc', {
  set(value) { pending.set(this,String(value)); }, get() { return pending.get(this)||''; },
});
const appendChild = w.Node.prototype.appendChild;
w.Node.prototype.appendChild = function(child) {
  const result = appendChild.call(this,child);
  for (const [frame,html] of pending) {
    if (!frame.isConnected || loaded.has(frame)) continue;
    loaded.add(frame);
    const doc=frame.contentDocument, win=frame.contentWindow;
    instrument(win);
    doc.open(); doc.write(html);
    if (fixture.failure === 'startup') doc.querySelector('#range-start').remove();
    if (fixture.failure === 'timeout') frame.onload = null;
    for (const script of doc.querySelectorAll('script')) {
      if (!script.type || script.type==='text/javascript' || script.type==='application/javascript') win.eval(script.textContent);
    }
    doc.close();
  }
  return result;
};
async function mount(capture) {
  const existing = new Set(d.querySelectorAll('iframe'));
  const wrapper = d.createElement('div'); wrapper.className = 'notebook-output'; d.body.append(wrapper);
  for (const event of capture.events) {
    if (event.kind === 'HTML') {
      const fragment = d.createElement('div'); fragment.innerHTML = event.data;
      fragment.querySelectorAll('script').forEach(s=>s.remove());
      wrapper.append(...fragment.childNodes);
    } else if (event.kind === 'Javascript') {
      const element = d.createElement('div'); wrapper.append(element);
      // IPython's Javascript MIME exposes an output element. Support both
      // DOM and jQuery-style indexing without emulating its execution policy.
      element[0] = element; element.append = element.append.bind(element);
      w.element = element;
      const ready = await w.eval(event.data);
      assert.equal(ready.ready,true,'Mount returns readiness only after initialization');
    }
  }
  const frame = [...d.querySelectorAll('iframe')].find(frame=>!existing.has(frame));
  assert(frame, 'Notebook output needs its own isolated document');
  assert(!frame.contentWindow.google,'Child document does not inherit parent Colab globals');
  return frame.contentDocument;
}
function by(root, id) { const found = root.querySelector('#'+id); assert(found, 'Missing '+id); return found; }
function assertMounted(root, expected) {
  assert.equal(by(root,'assembly').textContent, expected.assembly_id);
  assert.equal(root.querySelectorAll('#assemblysvg .part').length, expected.objects.length, 'The selected map must contain every assembly part');
  assert.equal(by(root,'range-start').options.length, expected.objects.length, 'Start selector populated');
  assert.equal(by(root,'range-end').options.length, expected.objects.length, 'End selector populated');
  assert.match(by(root,'connection').textContent, /^Live\b/);
  assert(!by(root,'score').disabled, 'Score control initialized');
  assert(!by(root,'rank').disabled, 'Search control initialized');
}
function select(root,a,b) {
  by(root,'range-start').value=String(a); by(root,'range-end').value=String(b);
  by(root,'selectrange').click();
}
async function finish(root) {
  for (let i=0;i<20 && by(root,'score').disabled;i++) await new Promise(resolve=>setTimeout(resolve,0));
  assert(!by(root,'score').disabled, 'Callback finished');
}
(async()=>{
  if (fixture.failure) {
    await assert.rejects(()=>mount(fixture.mounts[0]), /Explorer could not start/);
    const notice=d.querySelector('[role="status"]');
    assert(notice && !notice.hidden, 'Initialization failure must stay visible');
    assert(notice.textContent.includes('loaded model and results are retained'));
    if (fixture.failure === 'timeout') assert(notice.textContent.includes('did not finish initializing'));
    if (fixture.failure === 'startup') {
      const doc=d.querySelector('iframe').contentDocument;
      assert(doc.documentElement.dataset.neuralpsError);
      assert(doc.querySelector('[role="alert"]'), 'Child startup error must explain failed initialization');
    }
    assert.equal(calls.length,0);
    dom.window.close();
    process.stdout.write(JSON.stringify({status:'PASS', checks:1}));
    return;
  }
  const first = await mount(fixture.mounts[0]);
  assertMounted(first, fixture.payloads[0]);
  const firstSelection = by(first,'selected-route').textContent;
  const second = await mount(fixture.mounts[1]);
  assertMounted(second, fixture.payloads[1]);
  assertMounted(first, fixture.payloads[0]);
  assert.equal(calls.length,0,'Mount must not start a model request');
  select(second,0,2);
  assert.equal(by(first,'selected-route').textContent, firstSelection, 'Second output must not change the first selection');
  by(second,'touching').checked=false;
  by(second,'touching').dispatchEvent(new second.defaultView.Event('change'));
  by(second,'score').click(); await finish(second);
  assert.equal(calls.length,1);
  assert.equal(calls[0].name,fixture.mounts[1].callback, 'Each output owns its callback');
  assert.deepEqual(calls[0].request.indices,[0,1,2]);
  assert.equal(calls[0].request.touching_boundaries,false);
  assert(by(second,'segmentresult').textContent.includes('0.410'));
  assert(by(second,'segmentresult').textContent.includes('0.610'));
  assert(!by(first,'segmentresult').textContent.includes('0.410'), 'Scores stay in their own output');
  assert.equal(d.querySelector('#outside').textContent,'Notebook button');
  assert.equal(d.querySelector('#outside-map').getAttribute('viewBox'),null);
  assert.equal(d.querySelectorAll('style').length,0, 'Explorer styles must stay inside its scoped root');
  assert.equal(d.querySelectorAll('script[type="application/json"]').length,0, 'Mount cannot depend on JSON script nodes in notebook DOM');
  by(second,'savehtml').click();
  assert.equal(downloads.at(-1).name,'neuralps_explorer.html');
  const saved = await downloads.at(-1).blob.text();
  const savedDom = new JSDOM(saved,{runScripts:'dangerously', virtualConsole:vc});
  const savedRoot = savedDom.window.document;
  assert.equal(savedRoot.querySelector('#connection').textContent,'Saved result');
  assert.equal(savedRoot.querySelector('#assembly').textContent,fixture.payloads[1].assembly_id);
  assert.equal(savedRoot.querySelectorAll('#assemblysvg .part').length,fixture.payloads[1].objects.length);
  assert(!saved.includes('Notebook button'), 'Save must export only this explorer, not notebook DOM');
  assert(!saved.includes(fixture.mounts[0].callback));
  assert(!saved.includes(fixture.mounts[1].callback));
  assert(!saved.includes(fixture.payloads[0].assembly_id), 'Save must not include the other live session');
  assert.deepEqual(errors,[]);
  savedDom.window.close(); dom.window.close();
  process.stdout.write(JSON.stringify({status:'PASS', checks:10}));
})().catch(e=>{console.error(e.stack); dom.window.close(); process.exit(1);});
"""


def run_javascript(testcase, fixture, script):
    """Run the real browser scripts when the optional jsdom dependency exists."""
    node = shutil.which("node")
    if not node:
        testcase.skipTest("Node.js is required for frontend execution regression")
    env = os.environ.copy()
    if not env.get("NEURALPS_JSDOM_MODULE"):
        sibling = Path(__file__).resolve().parents[2]/"browser-tools/node_modules/jsdom"
        if sibling.is_dir():
            env["NEURALPS_JSDOM_MODULE"] = str(sibling)
    probe = subprocess.run([node, "-e", "require(process.env.NEURALPS_JSDOM_MODULE || 'jsdom')"],
                           env=env, capture_output=True, text=True)
    if probe.returncode:
        testcase.skipTest("Install jsdom and set NEURALPS_JSDOM_MODULE to run frontend execution regression")
    result = subprocess.run([node, "-e", script], input=json.dumps(fixture),
                            env=env, capture_output=True, text=True, timeout=30)
    testcase.assertEqual(result.returncode, 0, result.stdout+result.stderr)
    testcase.assertEqual(json.loads(result.stdout)["status"], "PASS")


class ColabEmbeddingTests(unittest.TestCase):
    def test_live_output_has_an_explicit_javascript_execution_path(self):
        capture, callbacks = capture_mount(payload())
        self.assertTrue(any(event["kind"] == "Javascript" for event in capture["events"]),
                        "A full document in HTML MIME does not initialize reliably in Colab")
        self.assertIn(capture["callback"], callbacks)
        response = callbacks[capture["callback"]]({"action": "state"})
        self.assertTrue(response.data["ok"])

    def test_distinct_outputs_register_distinct_callbacks(self):
        first, _ = capture_mount(payload("first"))
        second, _ = capture_mount(payload("second"))
        self.assertNotEqual(first["callback"], second["callback"])

    def test_failed_mount_is_not_reported_as_a_working_explorer(self):
        with self.assertRaisesRegex(ValueError, "did not initialize"):
            capture_mount(payload(), ready=False)

    def test_standalone_retains_data_and_has_no_live_callback(self):
        data = payload('example </script><script>throw Error("injection")</script>')
        saved = render_html(data)
        self.assertIn('id="neuralps-data"', saved)
        self.assertNotIn(data["assembly_id"], saved)
        self.assertIn('id="neuralps-bridge" type="application/json">null</script>', saved)

    def run_frontend(self, failure=None):
        data = [payload("assembly in first cell"), payload("assembly in second cell")]
        mounts = [capture_mount(p)[0] for p in data]
        run_javascript(self, dict(payloads=data, mounts=mounts, failure=failure), NODE_HARNESS)

    def test_real_frontend_mounts_after_notebook_strips_html_scripts(self):
        self.run_frontend()

    def test_child_startup_failure_rejects_mount_with_visible_error(self):
        self.run_frontend("startup")

    def test_child_load_timeout_rejects_mount_with_visible_error(self):
        self.run_frontend("timeout")


if __name__ == "__main__":
    unittest.main()
