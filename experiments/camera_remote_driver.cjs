// Headless Chrome driver for experiments/camera_remote_measure.py (playwright-core).
// Usage: node camera_remote_driver.cjs <config.json>; the bearer token arrives in the DREAM_MEASURE_TOKEN
// environment variable (this job's own pairing token) and is never written. One page per run, one mode per page.
// Web security is disabled only so a local file page can call the loopback IAP tunnel without CORS headers; the
// transports, requests and WHEP signalling are the web client's.
const { chromium } = require(process.env.DREAM_PLAYWRIGHT_CORE || 'playwright-core');
const fs = require('fs');
const os = require('os');
const path = require('path');

const cfg = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
(async () => {
  const out = { cfg: { ...cfg }, console: [], failed_requests: [] };
  const browser = await chromium.launch({
    executablePath: cfg.chrome,
    headless: true,
    args: ['--disable-web-security', '--autoplay-policy=no-user-gesture-required',
      '--disable-background-timer-throttling', '--disable-renderer-backgrounding',
      '--disable-backgrounding-occluded-windows', '--enable-features=WebRtcHideLocalIpsWithMdns:false',
      '--disable-features=WebRtcHideLocalIpsWithMdns'],
  });
  out.chrome_version = browser.version();
  const ctx = await browser.newContext({ viewport: { width: 640, height: 480 } });
  const page = await ctx.newPage();
  page.on('console', (m) => { if (['error', 'warning'].includes(m.type())) out.console.push(m.type() + ': ' + m.text().slice(0, 200)); });
  page.on('pageerror', (e) => out.console.push('pageerror: ' + String(e).slice(0, 200)));
  page.on('requestfailed', (r) => out.failed_requests.push({ url: r.url().replace(/\?.*/, ''), error: (r.failure() || {}).errorText, t: Date.now() }));
  await page.goto('file:///' + cfg.page.replace(/\\/g, '/'));
  await page.bringToFront();
  out.start_ms = Date.now();
  try {
    out.result = await page.evaluate(([mode, base, token, seconds]) => window.__m.run(mode, base, token, seconds),
      [cfg.mode, cfg.base, process.env.DREAM_MEASURE_TOKEN, cfg.seconds]);
  } catch (e) { out.run_error = String(e).slice(0, 400); }
  out.end_ms = Date.now();
  fs.writeFileSync(cfg.out, JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error('DRIVER ERROR', e); process.exit(2); });
