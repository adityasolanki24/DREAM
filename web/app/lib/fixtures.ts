/**
 * Interface-preview fixtures for `next dev` only (`?fixture=<name>`). They exist so the layout can be reviewed in
 * idle, running, failed and safety-stopped states without a robot. They are never robot data: the app shows a
 * permanent "Interface preview fixture" label while one is active, nothing is sent anywhere, and production builds
 * ignore the parameter.
 */
import type { Feed, Thread, Tool } from '../feed.mjs';
import type { Link, Snapshot, Status, Task } from '../types';

export type Fixture = {
  name: string;
  link: Link;
  snapshot: Snapshot | null;
  status: Status | null;
  tasks: Task[];
  feed: Feed;
  /** Shows a labelled test pattern instead of a camera frame so the stage layout can be reviewed. */
  testPattern: boolean;
};

export const FIXTURE_NAMES = ['empty', 'idle', 'running', 'paused', 'completed', 'failed', 'safety', 'disconnected'] as const;

const NOW = Math.floor(Date.now() / 1000); // relative to load time so "recent" states read as recent

const providers = [
  { id: 'structured', label: 'Structured goal parser (no language model)', available: true, reason: null, models: [], default_model: null, supports: ['inspect cabinet', 'goto <place>'] },
  { id: 'qwen', label: 'Qwen VL', available: true, reason: null, models: ['qwen2.5-vl-72b-instruct'], default_model: 'qwen2.5-vl-72b-instruct', supports: [] },
  { id: 'openai_chatgpt_plan', label: 'ChatGPT plan', available: false, reason: 'start the backend with --openai-token-file', models: [], default_model: null, supports: [] },
];

function map(): Snapshot['map'] {
  const W = 72, H = 52;
  const cells: number[][] = [];
  for (let x = 0; x < W; x++) {
    const col: number[] = [];
    for (let y = 0; y < H; y++) {
      const wall = x === 6 || x === 65 || y === 5 || y === 46 || (x === 36 && (y < 20 || y > 28)) || (y === 26 && x > 46 && x < 58);
      const seen = (x - 22) ** 2 / 900 + (y - 26) ** 2 / 500 < 1 || (x > 30 && x < 62 && y > 8 && y < 40 && (x + y) % 23 !== 0);
      col.push(!seen && !(x > 4 && x < 67 && (y === 5 || y === 46)) ? 0 : wall ? 2 : 1);
    }
    cells.push(col);
  }
  return { cells, resolution_m: 0.1, origin_m: [-3.6, -2.6], axes: 'x right, y up', odometry: 'oracle-assisted (fixture)' };
}

function testPattern(): string {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">
<defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#2a3038"/><stop offset="0.55" stop-color="#1a1d22"/><stop offset="1" stop-color="#121417"/></linearGradient>
<pattern id="p" width="64" height="64" patternUnits="userSpaceOnUse"><path d="M64 0H0V64" fill="none" stroke="#ffffff" stroke-opacity="0.05"/></pattern></defs>
<rect width="1280" height="720" fill="url(#g)"/><rect width="1280" height="720" fill="url(#p)"/>
<g fill="none" stroke="#ece8e0" stroke-opacity="0.18"><circle cx="640" cy="360" r="120"/><path d="M600 360h80M640 320v80"/></g>
<text x="640" y="540" fill="#ece8e0" fill-opacity="0.5" font-family="monospace" font-size="22" text-anchor="middle" letter-spacing="4">INTERFACE PREVIEW TEST PATTERN · NOT A CAMERA FRAME</text></svg>`;
  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
}

function task(id: string, text: string, state: string, opts: Partial<Task> = {}): Task {
  return { id, text, state, priority: 0, created: NOW - 600, updated: NOW - 20, result: null, steps: 0, suspensions: 0, resume: 'auto', meta: { provider: 'structured', model: null }, ...opts };
}

const tool = (id: string, name: string, args: Record<string, unknown>, status: string, message = ''): Tool => ({ id, name, args, status, message });

function thread(t: Task, tools: Tool[], end: Thread['end'] = null, claim: Thread['claim'] = null): Thread {
  return { id: t.id, text: t.text, state: t.state, tools, end, claim };
}

function feedOf(threads: Thread[]): Feed {
  return { threads: Object.fromEntries(threads.map((t) => [t.id, t])), calls: {}, runs: {} };
}

function snapshot(over: Partial<Snapshot> = {}): Snapshot {
  return {
    provider: 'structured', provider_label: providers[0].label, provider_choices: providers, environment: 'Interface preview fixture',
    robot_time_s: 184.6, odometry: [1.42, 0.37, 0.6], stop_reason: null, camera: null, map: map(),
    objects: [
      { id: 1, color: 'red', position: [2.1, 1.0, 0.4], extent: [0.3, 0.3, 0.6], views: 4 },
      { id: 2, color: 'blue', position: [-0.8, -1.2, 0.4], extent: [0.4, 0.3, 0.8], views: 2 },
    ],
    persistent_memory: { available: false, error: 'Fixture: home memory is not connected.' },
    task_graph: null, outcome: null, history: [], ...over,
  };
}

const done1 = task('t-101', 'goto kitchen', 'succeeded', { created: NOW - 2400, updated: NOW - 2100, steps: 3, result: 'reached the remembered place' });
const done1Tools = [
  tool('a1', 'get_robot_state', {}, 'succeeded', 'pose (0.00, 0.00, 0.00)'),
  tool('a2', 'go_to', { x: 1.2, y: 0.4 }, 'succeeded', 'arrived within 0.15 m'),
  tool('a3', 'verify_goal', { goal: 'kitchen' }, 'succeeded', 'within the remembered region'),
];

function running(): Fixture {
  const t = task('t-102', 'inspect cabinet', 'running', { steps: 4 });
  const tools = [
    tool('b1', 'get_robot_state', {}, 'succeeded', 'pose (1.10, 0.20, 0.40)'),
    tool('b2', 'look_around', {}, 'succeeded', '2 candidates in view'),
    tool('b3', 'approach', { candidate: 1, side: 0 }, 'succeeded', 'at standoff 0.8 m'),
    tool('b4', 'observe_scene', { candidate: 1 }, 'running'),
  ];
  return {
    name: 'running', link: 'online', status: { running: t.id }, tasks: [done1, t], testPattern: true,
    feed: feedOf([thread(done1, done1Tools, { state: 'succeeded', reason: 'reached the remembered place' }), thread(t, tools)]),
    snapshot: snapshot({
      camera: { data_url: testPattern(), time_s: 183.9, transport: 'fixture' },
      task_graph: { kind: 'observed', task_id: t.id, nodes: tools.map((x) => ({ id: x.id, label: x.name, args: x.args })), edges: tools.slice(1).map((x, i) => ({ source: tools[i].id, target: x.id })) },
    }),
  };
}

export function loadFixture(name: string | null): Fixture | null {
  if (!name || process.env.NODE_ENV !== 'development') return null;
  switch (name) {
    case 'empty':
      return { name, link: 'online', status: { running: null }, tasks: [], feed: feedOf([]), testPattern: false, snapshot: snapshot({ map: null, objects: [] }) };
    case 'idle':
      return { name, link: 'online', status: { running: null }, tasks: [done1], testPattern: false, feed: feedOf([thread(done1, done1Tools, { state: 'succeeded', reason: 'reached the remembered place' }, { success: true, reason: 'within the remembered region' })]), snapshot: snapshot({ outcome: { state: 'succeeded', reason: 'reached the remembered place' } }) };
    case 'running':
      return running();
    case 'paused': {
      const r = running();
      const t = { ...r.tasks[1], state: 'suspended', resume: 'ask' as const };
      return { ...r, name, status: { running: null }, tasks: [done1, t], feed: feedOf([...Object.values(r.feed.threads).slice(0, 1), { ...r.feed.threads['t-102'], state: 'suspended', tools: r.feed.threads['t-102'].tools.map((x) => (x.status === 'running' ? { ...x, status: 'interrupted', message: 'paused at a safe boundary' } : x)) }]) };
    }
    case 'completed': {
      const r = running();
      const t = { ...r.tasks[1], state: 'succeeded', steps: 5, result: 'handle visible from the inspection pose' };
      const th = r.feed.threads['t-102'];
      const tools = [...th.tools.map((x) => (x.status === 'running' ? { ...x, status: 'succeeded', message: 'handle visible' } : x)), tool('b5', 'verify_goal', { goal: 'inspect cabinet' }, 'succeeded', 'inspection pose reached')];
      return { ...r, name, status: { running: null }, tasks: [done1, t], feed: feedOf([r.feed.threads['t-101'], { ...th, state: 'succeeded', tools, end: { state: 'succeeded', reason: t.result }, claim: { success: true, reason: 'inspection pose reached' } }]), snapshot: { ...r.snapshot!, outcome: { state: 'succeeded', reason: t.result } } };
    }
    case 'failed': {
      const r = running();
      const t = { ...r.tasks[1], state: 'failed', steps: 6, result: 'no handle visible after two approaches' };
      const th = r.feed.threads['t-102'];
      const tools = [...th.tools.map((x) => (x.status === 'running' ? { ...x, status: 'failed', message: 'handle not visible from this side' } : x)), tool('b5', 'approach', { candidate: 1, side: 1 }, 'succeeded', 'at standoff 0.8 m'), tool('b6', 'observe_scene', { candidate: 1 }, 'failed', 'handle not visible')];
      return { ...r, name, status: { running: null }, tasks: [done1, t], feed: feedOf([r.feed.threads['t-101'], { ...th, state: 'failed', tools, end: { state: 'failed', reason: t.result }, claim: { success: false, reason: t.result } }]), snapshot: { ...r.snapshot!, outcome: { state: 'failed', reason: t.result } } };
    }
    case 'safety': {
      const r = running();
      return { ...r, name, snapshot: { ...r.snapshot!, stop_reason: 'base tilt exceeded 0.35 rad' } };
    }
    case 'disconnected': {
      const r = running();
      return { ...r, name, link: 'offline' };
    }
    default:
      return null;
  }
}
