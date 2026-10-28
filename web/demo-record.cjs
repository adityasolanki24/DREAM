// Records one take of the DREAM web interface demo (storyboard in docs/phase1/INTERFACE_DEMO.md).
//
// Usage (Git Bash, from the repo root, after `cd web && npm run build:export`):
//   DEMO_VIEWPORT=desktop DEMO_OUT=<outdir> DEMO_STORE=<new empty dir> node web/demo-record.cjs
//   DEMO_VIEWPORT=phone   DEMO_OUT=<outdir> DEMO_STORE=<new empty dir> node web/demo-record.cjs
//
// The script starts its own backend (experiments.phase1_interface, seed 4100, DEMO_PORT default 8765, the given fresh store),
// stops and restarts only that process for the offline step, and stops it at the end. It refuses to run
// if port 8765 already answers, so it never attaches to or kills someone else's backend.
// The token is read from %LOCALAPPDATA%\DREAM\phone_token and is never printed or shown on screen.
// Only the structured goal parser is used (no language model, no ChatGPT or Qwen calls).

const fs = require('fs');
const path = require('path');
const { spawn, execFileSync } = require('child_process');
const { chromium } = require(process.env.DREAM_PLAYWRIGHT_CORE || 'C:/Users/24adi/.cache/dream-phase1/shots/node_modules/playwright-core');

const REPO = path.resolve(__dirname, '..');
const PORT = Number(process.env.DEMO_PORT || 8765);
const BASE = `http://127.0.0.1:${PORT}`;
const PY = path.join(REPO, '.venv', 'Scripts', 'python.exe');
const CHROME = process.env.DREAM_BROWSER || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const FFMPEG = process.env.DREAM_FFMPEG || 'ffmpeg';
const sleep = ms => new Promise(r => setTimeout(r, ms));

const VIEWPORTS = {
  desktop: { width: 1280, height: 800, phone: false, theme: 'light' },
  phone: { width: 390, height: 844, phone: true, theme: 'dark' },
};
const which = process.env.DEMO_VIEWPORT || 'desktop';
const vp = VIEWPORTS[which];
if (!vp) throw new Error(`DEMO_VIEWPORT must be one of ${Object.keys(VIEWPORTS).join(', ')}`);
const OUT = process.env.DEMO_OUT;
const STORE = process.env.DEMO_STORE;
if (!OUT || !STORE) throw new Error('DEMO_OUT and DEMO_STORE are required');
if (fs.existsSync(STORE) && fs.readdirSync(STORE).length) throw new Error('DEMO_STORE must be a new empty directory');
fs.mkdirSync(STORE, { recursive: true });
const KEYS = path.join(OUT, 'keyframes');
const RAW = path.join(OUT, 'raw');
fs.mkdirSync(KEYS, { recursive: true });
fs.mkdirSync(RAW, { recursive: true });
const LOG = path.join(OUT, `backend-${which}.log`);
const name = `dream-interface-${which}-${vp.theme}`;

const token = fs.readFileSync(path.join(process.env.LOCALAPPDATA, 'DREAM', 'phone_token'), 'utf8').trim();
const auth = { Authorization: `Bearer ${token}` };
const TERMINAL = ['succeeded', 'failed', 'cancelled'];
const COPY_AUTO = 'Paused for interrupt · resumes automatically when it is next in priority';

// ---- timeline -----------------------------------------------------------------------------
let t0 = Date.now();
const timeline = [];
let frameNo = 0;
const mark = (step, note = '') => {
  const entry = { t: +((Date.now() - t0) / 1000).toFixed(1), step, note };
  timeline.push(entry);
  console.log(`[${entry.t}s] ${step}${note ? ' · ' + note : ''}`);
};

// ---- backend (only the process this script started) ---------------------------------------
let backend = null;
async function serving() {
  try { await fetch(BASE + '/status', { signal: AbortSignal.timeout(1500) }); return true; } catch { return false; }
}
function startBackend() {
  stopping = false;
  const fd = fs.openSync(LOG, 'a');
  backend = spawn(PY, ['-m', 'experiments.phase1_interface', '--seed', '4100', '--port', String(PORT), '--store', STORE], {
    cwd: REPO, stdio: ['ignore', fd, fd], windowsHide: true,
  });
  mark('backend', 'started (pid ' + backend.pid + ')');
  const self = backend;
  backend.on('exit', (code, signal) => {
    if (backend === self && !stopping) mark('backend', `exited unexpectedly (code ${code}, signal ${signal})`);
  });
}
async function waitReady(ms) {
  const end = Date.now() + ms;
  while (Date.now() < end) { if (await serving()) return; await sleep(1000); }
  throw new Error('backend did not start');
}
let stopping = false;
async function stopBackend() {
  if (!backend) return;
  stopping = true;
  const exited = new Promise(r => backend.once('exit', r));
  execFileSync('taskkill', ['/PID', String(backend.pid), '/T', '/F'], { stdio: 'ignore' });
  await exited;
  backend = null;
  mark('backend', 'stopped (only this take\'s process tree)');
}

// ---- overlay caption (fixed, pointer-events none, facts only) ------------------------------
async function caption(page, title, fact) {
  await page.evaluate(({ title, fact, bottom }) => {
    let el = document.getElementById('demo-caption');
    if (!el) {
      el = document.createElement('div');
      el.id = 'demo-caption';
      el.setAttribute('aria-hidden', 'true');
      el.style.cssText = 'position:fixed;left:50%;transform:translateX(-50%);z-index:2147483647;pointer-events:none;'
        + 'max-width:min(92vw,640px);padding:8px 14px;border-radius:14px;background:rgba(18,18,20,.84);color:#fff;'
        + 'font:400 13px/1.4 Inter,system-ui,sans-serif;text-align:center;box-shadow:0 4px 16px rgba(0,0,0,.25);';
      document.body.appendChild(el);
    }
    if (bottom) { el.style.top = ''; el.style.bottom = bottom; } else { el.style.bottom = ''; el.style.top = '10px'; }
    el.replaceChildren();
    const b = document.createElement('b');
    b.textContent = title;
    const s = document.createElement('span');
    s.style.display = 'block';
    s.textContent = fact;
    el.append(b, s);
  }, { title, fact, bottom: vp.phone ? '78px' : null });
  mark('caption', title);
}

async function key(page, label) {
  frameNo += 1;
  const file = path.join(KEYS, `${which}-${String(frameNo).padStart(2, '0')}-${label}.png`);
  await page.screenshot({ path: file });
  mark('keyframe', path.basename(file));
}

// ---- navigation helpers --------------------------------------------------------------------
// Desktop has a sidebar (Overview / Activity / World / Memory); phone has the bottom tab bar (Chat / Robot / Task / Memory).
async function goView(page, desk, phone) {
  const label = vp.phone ? phone : desk;
  if (!label) return;
  await page.getByRole('button', { name: label, exact: true }).first().click();
}

async function setTheme(page, choice) {
  await page.getByRole('button', { name: 'Session settings' }).click();
  await page.getByRole('menuitemradio', { name: choice, exact: true }).click();
}

// ---- the take ------------------------------------------------------------------------------
async function run() {
  if (await serving()) throw new Error(`port ${PORT} already answers; choose another DEMO_PORT (this script never attaches to a backend it did not start)`);
  startBackend();
  await waitReady(180000);
  mark('backend', 'ready');

  const browser = await chromium.launch({ executablePath: CHROME, headless: true });
  const ctx = await browser.newContext({
    viewport: { width: vp.width, height: vp.height },
    recordVideo: { dir: RAW, size: { width: vp.width, height: vp.height } },
  });
  await ctx.addInitScript(theme => {
    try { localStorage.setItem('dream-theme', theme); } catch (e) { /* private window: keep the default */ }
  }, vp.theme);
  const page = await ctx.newPage();
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  t0 = Date.now();
  mark('take', `${which} ${vp.width}x${vp.height} ${vp.theme}`);
  const tasksOf = async () => (await (await ctx.request.get(`${BASE}/tasks`, { headers: auth })).json());
  const taskNow = async id => (await tasksOf()).find(t => t.id === id);
  const waitTask = async (id, pred, ms = 240000) => {
    const end = Date.now() + ms;
    while (Date.now() < end) { const t = await taskNow(id); if (t && pred(t)) return t; await sleep(700); }
    throw new Error('timeout waiting on task ' + id);
  };
  const waitForBoundary = async ms => {
    const end = Date.now() + ms;
    while (Date.now() < end) {
      const body = await (await ctx.request.get(`${BASE}/events?after=0`, { headers: auth })).json();
      const last = (body.events || body).slice(-1)[0];
      if (last && (last.kind === 'result' || last.kind === 'suspended')) return last.kind;
      await sleep(200);
    }
    return 'timeout';
  };
  const submit = async text => {
    await goView(page, null, 'Chat');
    await page.getByLabel('Task', { exact: true }).fill(text);
    const posted = page.waitForResponse(r => r.url().endsWith('/tasks') && r.request().method() === 'POST');
    await page.getByRole('button', { name: 'Send task' }).click();
    const id = (await (await posted).json()).id;
    mark('submitted', text);
    return id;
  };

  // 1. Connect
  await caption(page, 'Step 1 · Connect', 'A wrong token is rejected. The token is kept in this tab only (sessionStorage).');
  await page.goto(BASE + '/');
  await page.getByLabel('Access token').fill('wrong');
  await page.getByRole('button', { name: 'Connect', exact: true }).click();
  // The narrow header shows short labels (Rejected / Offline), so wait on the visible status element, not on text.
  await page.locator('.conn-unauthorised:visible').first().waitFor();
  await key(page, 'token-rejected');
  await page.getByLabel('Access token').fill(token);
  await page.getByRole('button', { name: 'Connect', exact: true }).click();
  await page.locator('.conn-online:visible').first().waitFor();
  await sleep(1200);
  await key(page, 'connected');

  // 2. Chat: structured goal, live tool rows, camera age, map, observed graph
  await caption(page, 'Step 2 · Submit a structured goal', 'Structured goal parser (no language model) · MuJoCo simulation · oracle-assisted odometry');
  const first = await submit('inspect cabinet');
  await page.locator(`[data-task-id="${first}"] .tool`).first().waitFor({ state: 'attached', timeout: 180000 });
  await key(page, 'chat-running-tool-rows');
  await goView(page, 'Overview', 'Robot');
  await caption(page, 'Step 2 · Camera and map', 'The camera updates only when a tool observes; its age is shown on the frame. Map odometry is oracle-assisted.');
  await page.locator('img[alt="Robot head camera frame"]').first().waitFor({ state: 'attached', timeout: 180000 });
  await sleep(2500);
  await key(page, 'camera-map-live');
  await goView(page, 'Activity', 'Task');
  await page.getByRole('tab', { name: 'Graph', exact: true }).click();
  await page.locator('.react-flow__node').first().waitFor({ state: 'attached', timeout: 60000 });
  await caption(page, 'Step 2 · Observed tool sequence', 'This graph is the observed tool sequence, not a plan.');
  await key(page, 'observed-graph');
  await goView(page, 'Overview', 'Task');

  // 3. Interrupt: raise the queued goto entrance above the running inspect
  await caption(page, 'Step 3 · Interrupt by priority', 'The running task is paused at a safe boundary and resumes by itself once it is next in priority (resume=auto).');
  const hi = await submit('goto entrance');
  // The phone renders only the active tab, so open the queue's view before looking for its priority field.
  await goView(page, 'Overview', 'Task');
  const prio = page.getByLabel('Priority for goto entrance');
  await prio.waitFor({ state: 'visible', timeout: 60000 });
  await prio.fill('5');
  await prio.press('Enter');
  mark('action', 'priority of goto entrance set to 5 in the queue list');
  // The pause takes effect only at the next action boundary, so this can wait for an in-flight tool (minutes here).
  await page.getByText(COPY_AUTO).first().waitFor({ state: 'attached', timeout: 900000 });
  await key(page, 'paused-for-interrupt');
  const hiEnd = await waitTask(hi, t => TERMINAL.includes(t.state), 900000);
  mark('interrupt', `goto entrance finished: ${hiEnd.state} (${hiEnd.result})`);
  await sleep(1000);
  await key(page, 'interrupt-finished');
  const resumed = await waitTask(first, t => t.state === 'running', 900000);
  mark('interrupt', `inspect cabinet state after interrupt: ${resumed.state}`);
  await sleep(1500);
  await key(page, 'auto-resumed');

  // 4. Cancel the resumed task (acts at the next safe point)
  const beforeCancel = await taskNow(first);
  if (TERMINAL.includes(beforeCancel.state)) {
    mark('cancel', `skipped: inspect cabinet already ${beforeCancel.state}`);
  } else {
    await caption(page, 'Step 4 · Cancel', 'Cancel takes effect at the next safe point. The robot is never commanded from the interface.');
    await goView(page, 'Overview', 'Task');
    // Cancel is honoured at the next safe point. A tool can run for minutes, so the click waits until the last
    // event is a tool result (the agent is between actions); the runtime then stops before dispatching another tool.
    const boundary = await waitForBoundary(120000);
    mark('boundary', `last event before cancel: ${boundary}`);
    await page.getByRole('button', { name: 'Cancel', exact: true }).first().click();
    mark('action', 'cancel requested');
    await page.getByText('Waiting for next safe point').first().waitFor({ state: 'attached', timeout: 10000 }).catch(() => {});
    await key(page, 'cancelling');
    // The runtime stops only after the in-flight tool returns; in rehearsal that took several minutes.
    await page.locator(`[data-task-id="${first}"]`).getByText('Task cancelled.', { exact: true }).waitFor({ state: 'attached', timeout: 900000 });
    await sleep(1000);
    await key(page, 'task-cancelled');
  }

  // 5. Views, theme switch, model picker if present
  await caption(page, 'Step 5 · World and Memory', 'Read-only views of the robot map, camera and session candidates. Candidates are session observations, not persistent memory.');
  await goView(page, 'World', 'Robot');
  await sleep(800);
  await key(page, 'world-view');
  await goView(page, 'Memory', 'Memory');
  await sleep(800);
  await key(page, 'memory-view');
  await caption(page, 'Step 5 · Theme', 'System, Light or Dark, chosen in the settings menu and kept in this browser.');
  const themeSequence = vp.phone ? ['System', 'Dark'] : ['Dark', 'Light'];
  for (const choice of themeSequence) {
    await setTheme(page, choice);
    await sleep(400);
    await key(page, `theme-${choice.toLowerCase()}`);
    await page.keyboard.press('Escape');
  }
  const picker = page.getByRole('button', { name: /model|provider/i });
  if (await picker.count()) {
    await picker.first().click();
    await key(page, 'model-picker');
    await page.keyboard.press('Escape');
  } else {
    mark('model-picker', 'not present in this build; step skipped');
  }
  await goView(page, 'Overview', 'Robot');

  // 6. Offline: stop only this take's backend, show the banner, restart, recover
  await caption(page, 'Step 6 · Backend stopped', 'The page keeps the last data received, says it is not live, and the composer is disabled until the backend answers.');
  await stopBackend();
  await page.locator('.conn-offline:visible').first().waitFor({ timeout: 60000 });
  await page.locator('.banner[role="alert"]').waitFor({ state: 'attached' });
  await sleep(4000);
  await key(page, 'offline-banner');
  await caption(page, 'Step 6 · Restart', 'Starting the backend again. The page reconnects by itself when it answers; the new session starts with no camera frame.');
  startBackend();
  await waitReady(180000);
  await page.locator('.conn-online:visible').first().waitFor({ timeout: 180000 });
  await sleep(1500);
  await key(page, 'recovered');

  await sleep(1500);
  if (errors.length) mark('page errors', errors.join(' | '));
  await ctx.close();
  const raw = await page.video().path();
  const webm = path.join(OUT, `${name}.webm`);
  fs.copyFileSync(raw, webm);
  await browser.close();
  mark('video', `saved ${path.basename(webm)}`);
  const mp4 = path.join(OUT, `${name}.mp4`);
  try {
    execFileSync(FFMPEG, ['-y', '-loglevel', 'error', '-i', webm, '-c:v', 'libx264', '-preset', 'slow', '-crf', '30',
      '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-an', mp4], { stdio: 'inherit' });
    mark('video', `converted ${path.basename(mp4)}`);
  } catch (error) {
    mark('video', 'mp4 conversion failed; the webm remains: ' + error.message);
  }
  fs.writeFileSync(path.join(OUT, `timeline-${which}-${vp.theme}.json`), JSON.stringify({ viewport: vp, timeline }, null, 2));
}

(async () => {
  try {
    await run();
  } catch (error) {
    console.error('TAKE FAILED:', error.message);
    process.exitCode = 1;
  } finally {
    await stopBackend().catch(() => {});
  }
})();
