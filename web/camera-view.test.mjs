import test from 'node:test';
import assert from 'node:assert/strict';
import { createFrameStore, elapsedAgeS, STALL_GRACE_MS } from './app/camera-view.mjs';

test('the age readout grows with real elapsed time once frames stop, and never before', () => {
  assert.equal(elapsedAgeS(0.2, 1000, 1050), 0.2);                          // between frames: no manufactured motion
  assert.equal(elapsedAgeS(0.2, 1000, 1000 + STALL_GRACE_MS), 0.2);        // exactly at the grace boundary
  assert.ok(Math.abs(elapsedAgeS(0.2, 1000, 1000 + 5000) - 5.2) < 1e-9);
  assert.ok(Math.abs(elapsedAgeS(0.2, 1000, 1000 + 14000) - 14.2) < 1e-9);
  assert.equal(elapsedAgeS(0.2, 1000, 900), 0.2);                           // clock going backwards never lowers it
  assert.equal(elapsedAgeS(null, 1000, 9000), null);                        // unknown age stays unknown
  assert.equal(elapsedAgeS(0.2, null, 9000), 0.2);                          // no frame time known: keep the estimate
});

test('a new frame resets the readout to the fresh estimate', () => {
  assert.ok(elapsedAgeS(0.1, 20000, 20030) < 0.2);
});

test('the frame store notifies subscribers on each set and stops after unsubscribe', () => {
  const store = createFrameStore({ source: 'jpeg', url: 'a', fps: 10 });
  let sourceCalls = 0, urlCalls = 0;
  const offSource = store.subscribe(() => { sourceCalls++; });
  const offUrl = store.subscribe(() => { urlCalls++; });
  store.set({ source: 'jpeg', url: 'b', fps: 10 });
  store.set({ source: 'jpeg', url: 'c', fps: 11 });
  assert.equal(urlCalls, 2);
  assert.equal(store.get().url, 'c');
  offSource(); offUrl();
  store.set({ source: 'none', url: '', fps: 0 });
  assert.equal(sourceCalls, 2); // unsubscribed listeners are not called
});

test('select returns a stable primitive so per-frame fields do not disturb coarse readers', () => {
  const store = createFrameStore({ source: 'jpeg', url: 'a', fps: 10 });
  const snap = () => store.select((s) => s.source);
  const first = snap();
  store.set({ source: 'jpeg', url: 'b', fps: 12 });
  assert.equal(snap(), first);
});
