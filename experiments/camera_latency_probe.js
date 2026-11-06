// Injected into every page by camera_latency_driver.cjs (addInitScript). Measurement only; it reads pixels of
// the displayed <img>, counts DOM mutations and React commits, and never alters what the page shows.
(() => {
  if (window.__probe) return;
  const P = (window.__probe = { paints: [], mut: {}, commits: 0, renders: {}, mutCount: 0, events: [], recv: [] });
  P.now = () => performance.timeOrigin + performance.now();
  const BLOCKS = 19;
  let canvas = null, ctx = null;
  P.decode = (img) => {
    const w = img.naturalWidth, h = img.naturalHeight;
    if (!w || !h) return null;
    const bs = Math.round(w / 20);
    if (!canvas) { canvas = new OffscreenCanvas(w, bs); ctx = canvas.getContext('2d', { willReadFrequently: true }); }
    if (canvas.width !== w || canvas.height !== bs) { canvas.width = w; canvas.height = bs; }
    ctx.drawImage(img, 0, 0, w, bs, 0, 0, w, bs);
    const d = ctx.getImageData(0, 0, w, bs).data;
    const bits = [];
    for (let k = 0; k < BLOCKS; k++) {
      let sum = 0, n = 0;
      for (let y = Math.floor(bs / 4); y < Math.floor((3 * bs) / 4); y++)
        for (let x = k * bs + Math.floor(bs / 4); x < k * bs + Math.floor((3 * bs) / 4); x++) {
          const i = (y * w + x) * 4; sum += d[i] + d[i + 1] + d[i + 2]; n += 3;
        }
      bits.push(sum / n > 127.5 ? 1 : 0);
    }
    if (bits[0] !== 1 || bits[1] !== 0 || bits[2] !== 1) return { seq: -1, ok: false };
    let seq = 0, chk = 0;
    for (let i = 0; i < 12; i++) seq = (seq << 1) | bits[3 + i];
    for (let i = 0; i < 4; i++) chk = (chk << 1) | bits[15 + i];
    const want = (seq ^ (seq >> 4) ^ (seq >> 8)) & 15;
    return { seq, ok: chk === want };
  };
  // Poll the displayed element every animation frame; a new src that has finished loading is a painted frame.
  P.watchImg = (selector) => {
    let last = '';
    const tick = () => {
      const img = document.querySelector(selector);
      if (img && img.src && img.src !== last && img.complete && img.naturalWidth) {
        last = img.src;
        const t = P.now(), dec = P.decode(img);
        if (dec) P.paints.push({ t, seq: dec.seq, ok: dec.ok });
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  };
  P.startMutations = (cameraSelector) => {
    P.mut = {}; P.mutCount = 0;
    const key = (n) => {
      const el = n.nodeType === 1 ? n : n.parentElement;
      if (!el) return 'detached';
      const inside = !!el.closest(cameraSelector);
      return (inside ? 'INSIDE ' : 'OUTSIDE ') + el.tagName.toLowerCase() + (el.className && typeof el.className === 'string' ? '.' + el.className.split(' ')[0] : '');
    };
    P.observer = new MutationObserver((records) => {
      for (const r of records) { const k = key(r.target) + ' [' + r.type + (r.attributeName ? ':' + r.attributeName : '') + ']'; P.mut[k] = (P.mut[k] || 0) + 1; P.mutCount++; }
    });
    P.observer.observe(document.documentElement, { subtree: true, childList: true, attributes: true, characterData: true });
  };
  P.stopMutations = () => { if (P.observer) { P.observer.takeRecords(); P.observer.disconnect(); } };
  // Minimal React DevTools hook: counts commits and, per commit, which function components performed work.
  P.countRenders = false; P.cameraSelector = '.camera-frame';
  const firstHost = (f) => { for (let c = f.child; c; c = c.child) if (c.tag === 5) return c.stateNode; return null; };
  window.__REACT_DEVTOOLS_GLOBAL_HOOK__ = {
    supportsFiber: true, isDisabled: false, renderers: new Map(),
    inject(r) { this.renderers.set(1, r); return 1; },
    onCommitFiberUnmount() {}, onPostCommitFiberRoot() {}, checkDCE() {},
    onCommitFiberRoot(_id, root) {
      if (!P.countRenders) return;
      P.commits++;
      const stack = [root.current];
      while (stack.length) {
        const f = stack.pop();
        if (f.sibling) stack.push(f.sibling);
        if (f.child) stack.push(f.child);
        if ((f.flags & 1) && typeof f.type === 'function') {
          const host = firstHost(f);
          const where = host && host.closest ? (host.closest(P.cameraSelector) ? 'INSIDE camera' : 'OUTSIDE ' + host.tagName.toLowerCase() + (typeof host.className === 'string' && host.className ? '.' + host.className.split(' ')[0] : '')) : 'no-host';
          P.renders[where] = (P.renders[where] || 0) + 1;
        }
      }
    },
  };
  P.sync = async (ctlBase, n = 40) => {
    const out = [];
    for (let i = 0; i < n; i++) {
      const t0 = P.now();
      const j = await (await fetch(ctlBase + '/ctl/time', { cache: 'no-store' })).json();
      const t1 = P.now();
      out.push({ t0, t1, ts: j.t * 1000 });
      await new Promise((r) => setTimeout(r, 15));
    }
    return out;
  };
})();
