const { chromium } = require(process.env.DREAM_PLAYWRIGHT_CORE || 'playwright-core');
const fs = require('fs');
const assert = require('assert');

const BASE = 'http://127.0.0.1:8765';
const OFFLINE_BANNER = 'Connection lost. Showing the last data received; it is not live. Retrying…';
const COPY_ASK = 'Paused by you · waiting for Resume';
const COPY_AUTO = 'Paused for interrupt · resumes automatically when it is next in priority';
const TERMINAL = ['succeeded', 'failed', 'cancelled'];
const sleep = ms => new Promise(r => setTimeout(r, ms));

// The original v2 checks, unchanged except for the scratch screenshot directory.
async function legacy(browser, token, dir) {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(BASE + '/');
  await page.getByLabel('Access token').fill('wrong');
  await page.getByRole('button', { name: 'Connect', exact: true }).click();
  await page.getByText('Token rejected', { exact: true }).first().waitFor();
  await page.getByLabel('Access token').fill(token);
  await page.getByRole('button', { name: 'Connect', exact: true }).click();
  await page.getByText('Connected', { exact: true }).waitFor();
  console.log('authentication verified');

  await page.getByLabel('Task', { exact: true }).fill('inspect cabinet');
  const submitted = page.waitForResponse(r => r.url().endsWith('/tasks') && r.request().method() === 'POST');
  await page.getByRole('button', { name: 'Send task' }).click();
  const taskId = (await (await submitted).json()).id;
  const exchange = page.locator(`[data-task-id="${taskId}"]`);
  await page.locator('img[alt="Robot head camera frame"]').waitFor({ timeout: 120000 });
  await exchange.locator('.tool').first().waitFor({ timeout: 120000 });
  await page.screenshot({ path: dir + 'v2-live-camera.png' });
  const pause = page.getByRole('button', { name: 'Pause', exact: true });
  await pause.waitFor({ state: 'visible' });
  await page.waitForFunction(() => [...document.querySelectorAll('button')].some(x => x.textContent.trim() === 'Pause' && !x.disabled), undefined, { timeout: 90000 });
  await pause.click();
  await page.waitForFunction(() => [...document.querySelectorAll('button')].some(x => x.textContent.trim() === 'Resume' && !x.disabled), undefined, { timeout: 120000 });
  console.log('suspended');
  await page.screenshot({ path: dir + 'v2-live-paused.png' });
  await page.getByRole('button', { name: 'Resume', exact: true }).click();
  await page.waitForFunction(() => [...document.querySelectorAll('button')].some(x => x.textContent.trim() === 'Pause' && !x.disabled), undefined, { timeout: 90000 });
  console.log('resumed');
  await page.getByRole('button', { name: 'Cancel', exact: true }).click();
  await exchange.getByText('Task cancelled.', { exact: true }).waitFor({ timeout: 180000 });
  const deadline = Date.now() + 180000;
  let ownTask;
  while (Date.now() < deadline) {
    const response = await page.request.get(BASE + '/tasks', { headers: { Authorization: `Bearer ${token}` } });
    if (!response.ok()) throw new Error('Task status request failed');
    ownTask = (await response.json()).find(t => t.id === taskId);
    if (ownTask?.state === 'cancelled' && ownTask.steps >= 1) break;
    await new Promise(resolve => setTimeout(resolve, 500));
  }
  assert(ownTask?.state === 'cancelled' && ownTask.steps >= 1, 'submitted task must be cancelled after at least one step');
  console.log('submitted task cancelled after at least one step');
  await page.screenshot({ path: dir + 'v2-live-cancelled.png' });

  await page.getByRole('tab', { name: 'Map', exact: true }).click();
  await page.locator('canvas').waitFor({ timeout: 30000 });
  await page.getByRole('button', { name: 'Activity', exact: true }).click();
  await page.getByRole('tab', { name: 'Graph', exact: true }).click();
  await page.locator('.react-flow__node').first().waitFor({ timeout: 30000 });
  await page.screenshot({ path: dir + 'v2-final-graph.png' });
  await page.getByRole('tab', { name: 'Graph', exact: true }).focus();
  await page.keyboard.press('ArrowLeft');
  assert.equal(await page.getByRole('tab', { name: 'Execution', exact: true }).getAttribute('data-state'), 'active');
  await page.getByRole('button', { name: 'Memory', exact: true }).click();
  await page.getByRole('tab', { name: /History/ }).click();
  assert(await page.getByText('inspect cabinet').count());
  await page.getByRole('button', { name: 'Session settings' }).click();
  await page.getByRole('menuitem', { name: 'Change access token' }).click();
  await page.getByRole('dialog').waitFor();
  await page.keyboard.press('Escape');
  assert.equal(await page.getByRole('dialog').count(), 0);
  console.log('navigation, graph, tabs, dialog verified');

  const phone = await browser.newPage({ viewport: { width: 390, height: 844 } });
  await phone.goto(BASE + '/');
  await phone.getByLabel('Access token').fill(token);
  await phone.getByRole('button', { name: 'Connect', exact: true }).click();
  await phone.getByText('Connected', { exact: true }).waitFor();
  await phone.getByRole('button', { name: 'Chat', exact: true }).click();
  await phone.getByLabel('Task', { exact: true }).fill('draft survives');
  await phone.getByRole('button', { name: 'Robot', exact: true }).click();
  await phone.screenshot({ path: dir + 'v2-final-phone-robot.png' });
  await phone.getByRole('button', { name: 'Chat', exact: true }).click();
  assert.equal(await phone.getByLabel('Task', { exact: true }).inputValue(), 'draft survives');
  assert(await phone.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  await phone.screenshot({ path: dir + 'v2-final-phone-chat.png' });
  console.log('phone draft and width verified');

  await page.route('**/snapshot', route => route.abort());
  await page.getByText('Backend unreachable', { exact: true }).waitFor({ timeout: 30000 });
  await page.unroute('**/snapshot');
  await page.getByText('Connected', { exact: true }).waitFor({ timeout: 30000 });
  assert.deepEqual(errors, []);
  console.log('offline and reconnect verified; no page errors');
  await page.close();
  await phone.close();
}

async function v3(browser, token, shotDir) {
  const auth = { Authorization: `Bearer ${token}` };
  let anySuccess = false;
  const tasksOf = async ctx => {
    const r = await ctx.request.get(`${BASE}/tasks`, { headers: auth });
    assert(r.ok(), 'tasks request failed');
    return r.json();
  };
  const quiesce = async ctx => {
    for (const t of await tasksOf(ctx)) {
      if (!TERMINAL.includes(t.state)) await ctx.request.post(`${BASE}/tasks/${t.id}/cancel`, { headers: auth, data: {} });
    }
    const end = Date.now() + 240000;
    while (Date.now() < end) {
      if ((await tasksOf(ctx)).every(t => TERMINAL.includes(t.state))) return;
      await sleep(1000);
    }
    throw new Error('tasks did not settle');
  };
  const waitTask = async (ctx, id, pred, timeout = 180000) => {
    const end = Date.now() + timeout;
    while (Date.now() < end) {
      const t = (await tasksOf(ctx)).find(x => x.id === id);
      if (t && pred(t)) return t;
      await sleep(700);
    }
    throw new Error('timeout waiting on task ' + id);
  };
  const open = async (theme, vp) => {
    const ctx = await browser.newContext({ viewport: vp.size });
    await ctx.addInitScript(t => {
      try { localStorage.setItem('dream-theme', t); } catch (e) {}
      document.addEventListener('DOMContentLoaded', () => { window.__themeAtDCL = document.documentElement.dataset.theme; });
    }, theme);
    const page = await ctx.newPage();
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    return { ctx, page, errors };
  };
  const connect = async (page, t) => {
    await page.goto(BASE + '/');
    assert.equal(await page.evaluate(() => window.__themeAtDCL), t, 'theme must be applied before first paint');
    await page.getByLabel('Access token').fill(token);
    await page.getByRole('button', { name: 'Connect', exact: true }).click();
    await page.getByText('Connected', { exact: true }).waitFor();
  };
  const tab = async (page, vp, name) => { if (vp.phone) await page.getByRole('button', { name, exact: true }).click(); };
  // Toasts are transient (7 s); wait for them so they do not cover the state being captured.
  const shot = async (page, theme, vp, state, fast = false) => {
    if (fast) {
      // The interrupt window is short (the interrupting task finishes in seconds), so hide toasts instead of waiting for them.
      await page.addStyleTag({ content: '.toasts { display: none !important; }' });
    } else {
      await page.waitForFunction(() => !document.querySelector('.toast'), undefined, { timeout: 20000 }).catch(() => {});
      await page.waitForTimeout(400);
    }
    return page.screenshot({ path: `${shotDir}v3-${theme}-${vp.name}-${state}.png` });
  };

  const vps = [
    { name: 'desktop', size: { width: 1440, height: 900 }, phone: false },
    { name: 'phone', size: { width: 390, height: 844 }, phone: true },
  ];
  const themes = ['light', 'dark'];
  // DREAM_COMBO=light-desktop limits a run to one theme/viewport pair (the robot memory makes a repeated
  // "inspect cabinet" finish instantly, so the driver restarts the backend with a fresh store per pair).
  const wanted = (theme, vp) => !process.env.DREAM_COMBO || process.env.DREAM_COMBO === `${theme}-${vp.name}`;

  // Empty and unauthorised first, while the store holds no tasks.
  for (const theme of themes) for (const vp of vps) {
    if (!wanted(theme, vp)) continue;
    const { ctx, page, errors } = await open(theme, vp);
    await connect(page, theme);
    assert.equal((await tasksOf(ctx)).length, 0, 'empty state needs a fresh store');
    await page.waitForTimeout(800);
    await shot(page, theme, vp, 'empty');
    assert.deepEqual(errors, []);
    await ctx.close();
    const u = await open(theme, vp);
    await u.page.goto(BASE + '/');
    await u.page.getByLabel('Access token').fill('wrong');
    await u.page.getByRole('button', { name: 'Connect', exact: true }).click();
    await u.page.getByText('Token rejected', { exact: true }).first().waitFor();
    await u.page.waitForTimeout(500);
    await shot(u.page, theme, vp, 'unauthorised');
    assert.deepEqual(u.errors, []);
    await u.ctx.close();
    console.log('empty + unauthorised', theme, vp.name);
  }

  // Live scenes.
  for (const theme of themes) for (const vp of vps) {
    if (!wanted(theme, vp)) continue;
    const { ctx, page, errors } = await open(theme, vp);
    await connect(page, theme);
    const first = await (await ctx.request.post(`${BASE}/tasks`, { headers: auth, data: { text: 'inspect cabinet', priority: 0 } })).json();
    await waitTask(ctx, first.id, t => t.state === 'running');
    await tab(page, vp, 'Robot');
    await page.locator('img[alt="Robot head camera frame"]').first().waitFor({ timeout: 120000 });
    await page.waitForTimeout(800);
    await shot(page, theme, vp, 'online');

    // user pause: resume='ask'
    await ctx.request.post(`${BASE}/tasks/${first.id}/pause`, { headers: auth, data: {} });
    const paused = await waitTask(ctx, first.id, t => t.state === 'suspended');
    assert.equal(paused.resume, 'ask');
    await tab(page, vp, 'Task');
    await page.getByText(COPY_ASK).first().waitFor({ timeout: 30000 });
    await page.waitForTimeout(500);
    await shot(page, theme, vp, 'suspended');
    await ctx.request.post(`${BASE}/tasks/${first.id}/resume`, { headers: auth, data: {} });
    await waitTask(ctx, first.id, t => t.state === 'running');

    // higher-priority interrupt: resume='auto'
    const hi = await (await ctx.request.post(`${BASE}/tasks`, { headers: auth, data: { text: 'goto entrance', priority: 5 } })).json();
    const interrupted = await waitTask(ctx, first.id, t => t.state === 'suspended', 240000);
    assert.equal(interrupted.resume, 'auto');
    await tab(page, vp, 'Task');
    await page.getByText(COPY_AUTO).first().waitFor({ timeout: 30000 });
    await shot(page, theme, vp, 'suspended-interrupt', true);

    // offline banner after having connected
    await tab(page, vp, 'Robot');
    await page.route('**/snapshot', route => route.abort());
    await page.getByText('Backend unreachable', { exact: true }).waitFor({ timeout: 30000 });
    const banner = page.locator('.banner[role="alert"]');
    await banner.waitFor();
    assert.equal((await banner.innerText()).trim(), OFFLINE_BANNER);
    await page.waitForTimeout(500);
    await shot(page, theme, vp, 'offline');
    await page.unroute('**/snapshot');
    await page.getByText('Connected', { exact: true }).waitFor({ timeout: 30000 });

    await waitTask(ctx, hi.id, t => t.state !== 'queued', 240000).catch(() => null);
    if ((await tasksOf(ctx)).some(t => t.state === 'succeeded')) {
      anySuccess = true;
      await tab(page, vp, 'Task');
      await page.getByText('Runtime reports success').first().waitFor({ state: 'attached', timeout: 30000 });
    }
    assert.deepEqual(errors, []);
    await quiesce(ctx);
    await ctx.close();
    console.log('live scenes', theme, vp.name);
  }
  console.log('a task succeeded and showed "Runtime reports success":', anySuccess);
}

(async () => {
  const browser = await chromium.launch({
    executablePath: process.env.DREAM_BROWSER || 'C:/Program Files/Google/Chrome/Application/chrome.exe',
    headless: true,
  });
  const token = fs.readFileSync(process.env.LOCALAPPDATA + '/DREAM/phone_token', 'utf8').trim();
  const dir = process.env.DREAM_SCREENSHOTS || '../results/phase1_interface/screenshots/';
  const legacyDir = process.env.DREAM_LEGACY_SCREENSHOTS || dir;
  try {
    if (process.env.DREAM_LEGACY !== '1') await v3(browser, token, dir);
    if (!process.env.DREAM_COMBO || process.env.DREAM_LEGACY === '1') await legacy(browser, token, legacyDir);
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exit(1); });
