const { chromium } = require(process.env.DREAM_PLAYWRIGHT_CORE || 'playwright-core');
const fs = require('fs');
const assert = require('assert');

(async () => {
  const browser = await chromium.launch({
    executablePath: process.env.DREAM_BROWSER || 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
    headless: true,
  });
  const token = fs.readFileSync(process.env.LOCALAPPDATA + '/DREAM/phone_token', 'utf8').trim();
  const dir = process.env.DREAM_SCREENSHOTS || '../results/phase1_interface/screenshots/';
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  try {
    await page.goto('http://127.0.0.1:8765/');
    await page.getByLabel('Access token').fill('wrong');
    await page.getByRole('button', { name: 'Connect', exact: true }).click();
    await page.getByText('Token rejected', { exact: true }).waitFor();
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
      const response = await page.request.get('http://127.0.0.1:8765/tasks', { headers: { Authorization: `Bearer ${token}` } });
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
    await phone.goto('http://127.0.0.1:8765/');
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
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exit(1); });
