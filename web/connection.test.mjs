import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createPollGate, fetchWithTimeout } from './app/connection.mjs';

test('concurrent polls share one request and invalidated work cannot commit', async () => {
  const gate = createPollGate();
  let release; let calls = 0; let committed = 0;
  const run = () => gate.run(async (current) => { calls++; await new Promise(r => { release = r; }); if (current()) committed++; });
  const a = run(); const b = run();
  await Promise.resolve();
  assert.equal(calls, 1);
  gate.invalidate(); release(); await Promise.all([a, b]);
  assert.equal(committed, 0);
});

test('a stalled fetch is aborted within the deadline', async () => {
  const fetcher = (_url, { signal }) => new Promise((_resolve, reject) => signal.addEventListener('abort', () => reject(new Error('aborted'))));
  await assert.rejects(fetchWithTimeout(fetcher, '/', {}, 10), /aborted/);
});

test('a stalled response body is aborted within the deadline', async () => {
  const fetcher = async (_url, { signal }) => ({ json: () => new Promise((_resolve, reject) => signal.addEventListener('abort', () => reject(new Error('body aborted')))) });
  await assert.rejects(fetchWithTimeout(fetcher, '/', {}, 10), /body aborted/);
});
