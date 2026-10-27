import test from 'node:test';
import assert from 'node:assert/strict';
import { applyEvents, emptyFeed } from './app/feed.mjs';

const run = (extra = []) => [
  { type: 'CUSTOM', name: 'dream.scheduler', value: { kind: 'accepted', id: 't', text: 'inspect cabinet' } },
  { type: 'RUN_STARTED', threadId: 't', runId: 't-2' },
  { type: 'TOOL_CALL_START', toolCallId: 't-3', toolCallName: 'look_around' },
  { type: 'TOOL_CALL_ARGS', toolCallId: 't-3', delta: '{"turns":2}' },
  { type: 'TOOL_CALL_END', toolCallId: 't-3' },
  { type: 'TOOL_CALL_RESULT', toolCallId: 't-3', content: '{"status":"succeeded","message":""}' },
  ...extra,
];

test('tool calls fold into their task thread with arguments and result', () => {
  const { feed } = applyEvents(emptyFeed(), run());
  const t = feed.threads.t;
  assert.equal(t.text, 'inspect cabinet');
  assert.equal(t.state, 'running');
  assert.deepEqual(t.tools[0], { id: 't-3', name: 'look_around', args: { turns: 2 }, status: 'succeeded', message: '' });
});

test('a cancelled run is never presented as a success', () => {
  const end = { type: 'RUN_FINISHED', threadId: 't', runId: 't-2', result: { state: 'cancelled', reason: 'cancelled by user' } };
  const { feed, notices } = applyEvents(emptyFeed(), run([end]));
  assert.equal(feed.threads.t.end.state, 'cancelled');
  assert.equal(notices.at(-1).tone, 'warn');
  assert.equal(notices.at(-1).title, 'Task cancelled');
});

test('failures come from RUN_ERROR and stopped tools raise a warning', () => {
  const events = [
    { type: 'RUN_STARTED', threadId: 'u', runId: 'u-1' },
    { type: 'TOOL_CALL_START', toolCallId: 'u-2', toolCallName: 'go_to' },
    { type: 'TOOL_CALL_RESULT', toolCallId: 'u-2', content: '{"status":"stopped","message":"blocked"}' },
    { type: 'RUN_ERROR', runId: 'u-1', message: 'blocked' },
  ];
  const { feed, notices } = applyEvents(emptyFeed(), events);
  assert.equal(feed.threads.u.end.state, 'failed');
  assert.deepEqual(notices.map((n) => n.tone), ['info', 'warn', 'bad']);
});

test('applying events does not mutate the previous feed', () => {
  const first = applyEvents(emptyFeed(), run()).feed;
  const before = JSON.stringify(first);
  applyEvents(first, [{ type: 'RUN_ERROR', runId: 'x', message: 'm' }, { type: 'CUSTOM', value: { kind: 'finish', id: 't', success: true } }]);
  assert.equal(JSON.stringify(first), before);
});
