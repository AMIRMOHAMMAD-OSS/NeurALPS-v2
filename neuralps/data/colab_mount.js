(async () => {
  'use strict';
  const config = __NEURALPS_MOUNT_CONFIG__;
  const colab = globalThis.google?.colab;
  if (!colab?.kernel?.invokeFunction) {
    throw new Error('Open this explorer in a live Colab output cell.');
  }
  const host = document.createElement('div');
  host.id = config.mount_id;
  host.style.cssText = 'width:100%;margin:0;background:#f4f7f3;color:#183d3c;';
  const notice = document.createElement('div');
  notice.setAttribute('role', 'status');
  notice.style.cssText = 'padding:14px;font:14px/1.5 system-ui;color:#183d3c;background:#f4f7f3;';
  notice.textContent = 'Opening the NeurALPS compatibility explorer…';
  host.appendChild(notice);
  const frame = document.createElement('iframe');
  frame.title = 'NeurALPS live compatibility explorer';
  frame.style.cssText = 'display:block;width:100%;height:950px;border:0;background:#f4f7f3;color-scheme:light;';
  let observer;
  let resizePending = false;
  const resize = () => {
    if (resizePending || !frame.isConnected) return;
    resizePending = true;
    requestAnimationFrame(() => {
      resizePending = false;
      if (!frame.isConnected) { observer?.disconnect(); return; }
      const doc = frame.contentDocument;
      if (!doc?.body) return;
      const bottom = Math.max(...Array.from(doc.body.children, child => child.getBoundingClientRect().bottom));
      const height = Math.max(480, Math.ceil(bottom + frame.contentWindow.scrollY + 16));
      if (Math.abs(frame.getBoundingClientRect().height - height) > 2) frame.style.height = height + 'px';
      // The Colab output frame must grow as well as the nested app frame.
      colab.output?.setIframeHeight?.(document.documentElement.scrollHeight, true);
    });
  };
  const ready = new Promise((resolve, reject) => {
    let settled = false;
    const fail = message => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      observer?.disconnect();
      notice.hidden = false;
      notice.style.color = '#a43f35';
      notice.textContent = 'Explorer could not start: ' + message + ' Your loaded model and results are retained.';
      reject(new Error(notice.textContent));
    };
    const timer = setTimeout(() => fail('the interface did not finish initializing'), 15000);
    frame.onload = () => {
      if (settled) return;
      try {
        const win = frame.contentWindow;
        const doc = frame.contentDocument;
        // Bind one session-specific callback; srcdoc has no Colab globals.
        win.__neuralpsInvoke = request => colab.kernel.invokeFunction(config.callback, [request], {});
        const parts = doc.querySelectorAll('#assemblysvg .part').length;
        const choices = doc.querySelectorAll('#range-start option').length;
        if (doc.documentElement.dataset.neuralpsReady !== 'true' ||
            parts !== 2 * config.object_count || choices !== config.object_count) {
          throw new Error(doc.documentElement.dataset.neuralpsError || 'the compatibility maps did not initialize');
        }
        notice.hidden = true;
        if (typeof ResizeObserver !== 'undefined') {
          observer = new ResizeObserver(resize);
          observer.observe(doc.body);
        }
        win.addEventListener('resize', resize);
        win.addEventListener('click', resize);
        doc.addEventListener('toggle', resize, true);
        resize();
        settled = true;
        clearTimeout(timer);
        resolve({ready: true, parts, choices, mount_id: config.mount_id});
      } catch (error) {
        fail(error.message || String(error));
      }
    };
    frame.onerror = () => fail('the embedded document could not load');
  });
  // Native srcdoc preserves JSON/script nodes and separates notebook CSS.
  frame.srcdoc = config.html;
  host.appendChild(frame);
  document.body.appendChild(host);
  return await ready;
})()
