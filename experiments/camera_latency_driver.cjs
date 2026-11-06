// Headless Chrome driver for experiments/camera_latency_measure.py (playwright-core).
// Usage: node camera_latency_driver.cjs <config.json>   (writes cfg.out as JSON)
// All timestamps are page-clock epoch milliseconds (performance.timeOrigin + performance.now()); the Python side
// converts them to the server clock with the round-trip offsets recorded by __probe.sync.
const { chromium } = require(process.env.DREAM_PLAYWRIGHT_CORE || 'playwright-core');
const fs = require('fs');
const path = require('path');

const cfg = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const APP_IMG = '.camera-frame img';

(async () => {
  const out = { cfg: { ...cfg, token: '<redacted>' }, actions: [], dom: [], urls_with_token: 0, request_count: 0, console_errors: [] };
  const browser = await chromium.launch({
    executablePath: cfg.chrome,
    headless: true,
    args: ['--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows', '--autoplay-policy=no-user-gesture-required'],
  });
  out.chrome_version = browser.version();
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 800 } });
  await ctx.addInitScript({ content: fs.readFileSync(path.join(__dirname, 'camera_latency_probe.js'), 'utf8') });
  await ctx.addInitScript((t) => { try { sessionStorage.setItem('dream-token', t); } catch {} }, cfg.token);
  const ctl = await ctx.newPage();
  await ctl.goto(cfg.ctl + '/ctl/time');
  const page = await ctx.newPage();
  page.on('request', (r) => { out.request_count++; if (r.url().includes(cfg.token)) out.urls_with_token++; });
  out.responses = {};
  page.on('response', (r) => { const k = r.request().method() + ' ' + new URL(r.url()).pathname + ' ' + r.status(); out.responses[k] = (out.responses[k] || 0) + 1; });
  page.on('requestfailed', (r) => { const k = 'FAILED ' + new URL(r.url()).pathname + ' ' + (r.failure() || {}).errorText; out.responses[k] = (out.responses[k] || 0) + 1; });
  page.on('console', (m) => { if (m.type() === 'error') out.console_errors.push(m.text().slice(0, 200)); });
  page.on('pageerror', (e) => out.console_errors.push(String(e).slice(0, 200)));
  const cdp = await browser.newBrowserCDPSession();

  const sync = (n = 40) => ctl.evaluate((n) => window.__probe.sync(location.origin, n), n);
  const act = async (name, query = '') => {
    const r = await ctl.evaluate(async ([name, query]) => {
      const t0 = window.__probe.now();
      const resp = await (await fetch('/ctl/' + name + query, { cache: 'no-store' })).json();
      return { name, query, t0, t1: window.__probe.now(), resp };
    }, [name, query]);
    out.actions.push(r);
    return r;
  };
  const cpu = async () => {
    const server = await ctl.evaluate(async () => (await (await fetch('/ctl/cpu')).json()));
    const info = (await cdp.send('SystemInfo.getProcessInfo')).processInfo;
    const byType = {};
    for (const p of info) byType[p.type] = (byType[p.type] || 0) + p.cpuTime;
    return { wall_ms: await ctl.evaluate(() => window.__probe.now()), server_process_s: server.process_s, browser_by_type_s: byType };
  };
  const paints = () => page.evaluate(() => window.__probe.paints.length);
  const waitPaints = async (n, timeoutMs) => {
    const end = Date.now() + timeoutMs;
    while (Date.now() < end) { if ((await paints()) >= n) return true; await sleep(100); }
    return false;
  };

  const target = cfg.client === 'raw' ? `${cfg.base}/?mode=${cfg.mode}` : `${cfg.base}/`;
  await page.goto(target, { waitUntil: 'load' });
  await page.bringToFront();
  if (cfg.client === 'app') {
    await page.getByRole('navigation', { name: 'Interface mode' }).getByRole('button', { name: 'Live' }).click({ timeout: 20000 });
    await page.waitForSelector(APP_IMG, { state: 'attached', timeout: cfg.connect_timeout_ms || 30000 }).catch(() => { out.app_img_missing = true; });
    await page.evaluate((sel) => {
      const P = window.__probe;
      P.watchImg(sel);
      const labels = ['MJPEG push', 'JPEG polling', 'Simulation paused or not sensing', 'Last observed', 'Disconnected', 'Waiting for frames', 'No video', 'video disconnected'];
      let last = null;
      P.age = [];
      setInterval(() => { const t = document.body.innerText; P.age.push({ t: P.now(), age: (t.match(/([\d.]+ ?(?:s|min)) old/) || [])[1] || null, fps: (t.match(/(\d+) fps/) || [])[0] || null }); }, 500);
      setInterval(() => {
        const t = document.body.innerText;
        const found = labels.filter((l) => t.includes(l));
        const fps = (t.match(/(\d+) fps/) || [])[0] || null;
        const key = found.join('|') + '#' + (fps ? 'fps' : '');
        if (key !== last) { last = key; (P.dom = P.dom || []).push({ t: P.now(), found, fps }); }
      }, 40);
    }, APP_IMG);
  }
  out.first_paint = await waitPaints(3, cfg.first_paint_timeout_ms || 30000);
  if (!out.first_paint) {
    out.screenshot_note = 'no frames painted';
    if (cfg.client === 'app') out.body_text = (await page.evaluate(() => document.body.innerText)).slice(0, 600);
  }
  out.sync_start = await sync();

  const scenario = cfg.scenario;
  if (scenario === 'latency' || scenario === 'rerender') {
    if (scenario === 'rerender') {
      await page.evaluate(() => { const P = window.__probe; P.countRenders = true; P.commits = 0; P.renders = {}; P.startMutations('.camera-frame'); });
    }
    out.cpu_start = await cpu();
    const p0 = await paints();
    await sleep(cfg.duration_s * 1000);
    out.cpu_end = await cpu();
    if (scenario === 'rerender') {
      out.rerender = await page.evaluate(() => { const P = window.__probe; P.stopMutations(); P.countRenders = false; return { commits: P.commits, renders: P.renders, mutations: P.mut, mutation_total: P.mutCount, paints: P.paints.length }; });
    }
    out.paints_in_window = (await paints()) - p0;
  } else if (scenario === 'block_push') {
    await sleep(cfg.warm_s * 1000);
    await act('block', `?seconds=${cfg.block_s}`);
    await sleep((cfg.block_s + cfg.after_s) * 1000);
  } else if (['drop', 'swap', 'restart'].includes(scenario)) {
    await sleep(cfg.warm_s * 1000);
    for (let i = 0; i < cfg.repeats; i++) {
      await act(scenario === 'restart' ? 'restart' : scenario === 'swap' ? 'swap' : 'drop', scenario === 'restart' ? `?down_s=${cfg.down_s}` : '');
      await sleep(cfg.gap_s * 1000);
    }
  } else if (scenario === 'stale') {
    await sleep(cfg.warm_s * 1000);
    await act('source', '?run=0');
    await sleep(cfg.stale_s * 1000);
    await act('source', '?run=1');
    await sleep(cfg.after_s * 1000);
  } else if (scenario === 'crash') {
    await sleep(cfg.warm_s * 1000);
    await act('mark', '?name=chrome_kill');
    out.t_kill_node_ms = performance.timeOrigin + performance.now();
    fs.writeFileSync(cfg.out, JSON.stringify(out));
    browser.process().kill('SIGKILL');
    process.exit(0);
  }

  out.sync_end = await sync();
  out.paints = await page.evaluate(() => window.__probe.paints);
  out.recv = await page.evaluate(() => window.__probe.recv);
  out.page_events = await page.evaluate(() => window.__probe.events);
  out.dom = await page.evaluate(() => window.__probe.dom || []);
  out.age = await page.evaluate(() => window.__probe.age || []);
  if (cfg.client === 'app' && cfg.screenshot) await page.screenshot({ path: cfg.screenshot });
  fs.writeFileSync(cfg.out, JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error('DRIVER ERROR', e); process.exit(2); });
