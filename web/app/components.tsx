'use client';
import * as Tabs from '@radix-ui/react-tabs';
import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { Background, Handle, Position, ReactFlow, type Edge, type Node, type NodeProps } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import {
  AlertTriangle, Ban, Bell, BellOff, Box, Camera, CheckCircle2, CircleDot, Clock, Compass, History, KeyRound, Loader2,
  Mic, MicOff, Pause, Play, Radio, Send, ShieldAlert, Square, WifiOff, XCircle,
} from 'lucide-react';
import { argSummary, endTitle, type Notice, type Thread, type Tool } from './feed.mjs';
import { mapPixels, worldToPixel } from './map.mjs';
import { describeMeta, type ModelChoice, type ProviderEntry } from '../modelChoice.mjs';
import { ModelPicker } from './ModelPicker';
import type { Link, Snapshot, Task } from './types';

export const text = (v: unknown): string => (v == null ? '' : typeof v === 'string' ? v : JSON.stringify(v));
const clock = (seconds: number) => `${seconds.toFixed(1)} s`;

/* ---------- small building blocks ---------------------------------------------------------- */

export function Panel({ title, meta, className = '', children }: { title: string; meta?: ReactNode; className?: string; children: ReactNode }) {
  return (
    <section className={`panel ${className}`}>
      <header className="panel-head">
        <h2>{title}</h2>
        {meta && <div className="panel-meta">{meta}</div>}
      </header>
      {children}
    </section>
  );
}

export function Empty({ icon, title, children }: { icon: ReactNode; title: string; children?: ReactNode }) {
  return (
    <div className="empty">
      <div className="empty-icon">{icon}</div>
      <strong>{title}</strong>
      {children && <span>{children}</span>}
    </div>
  );
}

function ToolIcon({ status }: { status: string }) {
  if (status === 'running') return <Loader2 size={14} className="spin" aria-hidden />;
  if (status === 'succeeded') return <CheckCircle2 size={14} aria-hidden />;
  if (status === 'interrupted') return <Pause size={14} aria-hidden />;
  return <XCircle size={14} aria-hidden />;
}

const toolTone = (status: string) => (status === 'succeeded' ? 'good' : status === 'running' ? 'info' : status === 'interrupted' ? 'warn' : 'bad');

/* ---------- task state vocabulary ----------------------------------------------------------- */

const STATE_META: Record<string, { label: string; tone: string; icon: typeof Clock }> = {
  queued: { label: 'Queued', tone: 'muted', icon: Clock },
  running: { label: 'Running', tone: 'info', icon: Loader2 },
  suspended: { label: 'Suspended', tone: 'warn', icon: Pause },
  succeeded: { label: 'Runtime reports success', tone: 'good', icon: CheckCircle2 },
  failed: { label: 'Failed', tone: 'bad', icon: XCircle },
  cancelled: { label: 'Cancelled', tone: 'muted', icon: Ban },
};

/** State is always an icon plus a text label, never colour alone. */
export function StatePill({ state, label }: { state: string; label?: string }) {
  const meta = STATE_META[state] ?? { label: state, tone: 'muted', icon: CircleDot };
  const Icon = meta.icon;
  return (
    <span className={`pill tone-${meta.tone}`}>
      <Icon size={14} aria-hidden className={state === 'running' ? 'spin' : ''} />
      {label ?? meta.label}
    </span>
  );
}

/**
 * What really happens to a suspended task (dream/runtime/scheduler.py):
 *  - suspended by a higher-priority interrupt: resume='auto', it restarts by itself once it is the
 *    highest-priority runnable task;
 *  - paused by the user: resume='ask', it is held until Resume is pressed.
 */
export function suspendCopy(task: Pick<Task, 'resume'>): string {
  return task.resume === 'ask'
    ? 'Paused by you · waiting for Resume'
    : 'Paused for interrupt · resumes automatically when it is next in priority';
}

function SuspendNote({ task }: { task: Task }) {
  return (
    <p className={`note ${task.resume === 'ask' ? 'tone-muted' : 'tone-warn'}`}>
      <Pause size={14} aria-hidden /> {suspendCopy(task)}
    </p>
  );
}

const RUNNABLE = ['running', 'suspended', 'queued'];

/** The scheduler's order: priority DESC, created ASC. */
export const schedulerOrder = (a: Task, b: Task) => b.priority - a.priority || a.created - b.created;

function relativeAge(epochSeconds: number): string {
  const d = Math.max(0, Date.now() / 1000 - epochSeconds);
  if (d < 5) return 'just now';
  if (d < 90) return `${Math.round(d)} s ago`;
  if (d < 5400) return `${Math.round(d / 60)} min ago`;
  return `${Math.round(d / 3600)} h ago`;
}

/* ---------- connection ---------------------------------------------------------------------- */

const LINK_LABEL: Record<Link, string> = {
  idle: 'Not connected',
  connecting: 'Connecting',
  online: 'Connected',
  offline: 'Backend unreachable',
  unauthorised: 'Token rejected',
};

/** Short labels for the narrow (phone) header; the full label stays in aria-label and title. */
const LINK_SHORT: Record<Link, string> = {
  idle: 'Idle',
  connecting: 'Connecting',
  online: 'Connected',
  offline: 'Offline',
  unauthorised: 'Rejected',
};

export function ConnectionBadge({ link, compact = false }: { link: Link; compact?: boolean }) {
  return (
    <span className={`conn conn-${link}`} role="status" aria-label={LINK_LABEL[link]} title={LINK_LABEL[link]}>
      <i />
      {compact ? LINK_SHORT[link] : LINK_LABEL[link]}
    </span>
  );
}

const CONNECT_COPY: Record<Link, { title: string; body: string }> = {
  idle: { title: 'Connect to DREAM', body: 'Enter the access token for this session. It is kept in this tab only, sent as a request header, and never put in a URL.' },
  connecting: { title: 'Connecting…', body: 'Reaching the robot backend. This usually takes a moment.' },
  online: { title: 'Connected', body: 'You are connected to the robot backend.' },
  offline: { title: 'Cannot reach the robot backend', body: 'Check that the backend is running on this machine, then try again.' },
  unauthorised: { title: 'Token rejected', body: 'The access token was rejected. Check the token and try again.' },
};

export function ConnectCard({ link, error, onConnect }: { link: Link; error: string; onConnect: (token: string) => void }) {
  const [value, setValue] = useState('');
  const copy = CONNECT_COPY[link];
  const Icon = link === 'offline' ? WifiOff : link === 'unauthorised' ? ShieldAlert : link === 'connecting' ? Loader2 : KeyRound;
  return (
    <form
      className={`connect connect-${link}`}
      onSubmit={(e) => {
        e.preventDefault();
        if (value.trim()) onConnect(value.trim());
      }}
    >
      <div className="connect-icon"><Icon size={22} aria-hidden className={link === 'connecting' ? 'spin' : ''} /></div>
      <h2>{link === 'offline' || link === 'connecting' ? copy.title : 'Connect to DREAM'}</h2>
      <p>{error || copy.body}</p>
      <input
        type="password"
        autoComplete="off"
        aria-label="Access token"
        placeholder="Session access token"
        value={value}
        onChange={(e) => setValue(e.target.value)}
      />
      <button className="btn primary" type="submit" disabled={!value.trim() || link === 'connecting'}>
        {link === 'connecting' ? <Loader2 size={15} className="spin" aria-hidden /> : null} Connect
      </button>
      <small>Token location and backend command: see the repository README.</small>
    </form>
  );
}

/* ---------- chat ---------------------------------------------------------------------------- */

function ToolRow({ tool }: { tool: Tool }) {
  const args = argSummary(tool.args);
  return (
    <li className={`tool tone-${toolTone(tool.status)}`}>
      <ToolIcon status={tool.status} />
      <div>
        <b>{tool.name}</b>
        {args && <span className="args">{args}</span>}
        {tool.status !== 'succeeded' && tool.status !== 'running' && (
          <span className="toolnote">
            {tool.status}
            {tool.message ? `: ${tool.message}` : ''}
          </span>
        )}
        {tool.status === 'succeeded' && tool.message && <span className="toolnote">{tool.message}</span>}
      </div>
    </li>
  );
}

function AgentBlock({ task, thread, stopping, modelLabel }: { task: Task; thread?: Thread; stopping: boolean; modelLabel: string | null }) {
  // A live task state beats a stale "suspended" end marker left in the thread by an earlier pause.
  const state = RUNNABLE.includes(task.state) ? task.state : (thread?.end?.state ?? task.state);
  const done = ['succeeded', 'failed', 'cancelled'].includes(state);
  const reason = text(thread?.end?.reason ?? task.result);
  return (
    <div className={`bubble agent state-${state}`}>
      <div className="bubble-head">
        <span className="thread-who"><span className="who">DREAM</span>{modelLabel && <span className="thread-model" title={`Ran with ${modelLabel}`}>{modelLabel}</span>}</span>
        <StatePill state={state} />
      </div>
      {stopping && !done && (
        <p className="note tone-info"><Loader2 size={14} className="spin" aria-hidden /> Waiting for next safe point</p>
      )}
      {thread && thread.tools.length > 0 && (
        <ul className="tools">
          {thread.tools.map((t) => (
            <ToolRow key={t.id} tool={t} />
          ))}
        </ul>
      )}
      {task.state === 'queued' && <p className="muted">Waiting for the robot to be free.</p>}
      {state === 'running' && !stopping && !thread?.tools.length && <p className="muted">Planning the first action…</p>}
      {done && (
        <p className={`outcome tone-${state === 'succeeded' ? 'good' : state === 'cancelled' ? 'warn' : 'bad'}`}>
          <b>{endTitle(state)}.</b> {reason}
          {state === 'succeeded' && <small> This is the runtime&apos;s own report; no independent check is shown here.</small>}
          {state === 'cancelled' && <small> The task was stopped and was not completed.</small>}
        </p>
      )}
      {state === 'suspended' && <SuspendNote task={task} />}
    </div>
  );
}

export function Chat({
  tasks, threads, running, stopping, link, choices, choice, onChoose, onSend, notice,
}: {
  tasks: Task[]; threads: Record<string, Thread>; running: string | null; stopping: Set<string>; link: Link;
  choices: ProviderEntry[]; choice: ModelChoice | null; onChoose: (c: ModelChoice) => void;
  onSend: (text: string) => Promise<boolean>; notice: string;
}) {
  const [draft, setDraft] = useState('');
  const [busy, setBusy] = useState(false);
  const [listening, setListening] = useState(false);
  const [voiceNote, setVoiceNote] = useState('');
  const bottom = useRef<HTMLDivElement>(null);
  const recogniser = useRef<{ stop: () => void } | null>(null);
  const [voiceSupported, setVoiceSupported] = useState(false);
  const online = link === 'online';
  // Goal forms come from the chosen provider's `supports`, so chips and the placeholder never promise a form it rejects.
  const examples = choices.find((c) => c.id === choice?.provider)?.supports ?? [];
  // A chip with a placeholder such as "<place>" fills in the words before it, for the user to finish.
  const fill = (x: string) => (x.includes('<') ? x.slice(0, x.indexOf('<')) : x);
  const placeholder = online
    ? examples.length > 0 ? `Try: ${examples.join(' or ')}` : 'Ask DREAM to do something…'
    : 'Connect to send a task';

  useEffect(() => {
    const w = window as unknown as Record<string, unknown>;
    setVoiceSupported(Boolean(w.SpeechRecognition || w.webkitSpeechRecognition));
  }, []);
  const lastTools = tasks.map((t) => threads[t.id]?.tools.length ?? 0).join(',');
  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [tasks.length, lastTools, running]);

  async function send(value = draft) {
    const message = value.trim();
    if (!message || !online || busy) return;
    setBusy(true);
    const ok = await onSend(message);
    if (ok) setDraft('');
    setBusy(false);
  }

  function toggleVoice() {
    if (listening) return recogniser.current?.stop();
    const w = window as unknown as Record<string, new () => {
      lang: string; interimResults: boolean; start: () => void; stop: () => void;
      onresult: (e: { results: ArrayLike<ArrayLike<{ transcript: string }>> }) => void;
      onerror: (e: { error?: string }) => void; onend: () => void;
    }>;
    const Recognition = w.SpeechRecognition || w.webkitSpeechRecognition;
    if (!Recognition) return;
    const r = new Recognition();
    r.lang = navigator.language || 'en-US';
    r.interimResults = false;
    r.onresult = (e) => setDraft((old) => `${old}${old ? ' ' : ''}${e.results[0][0].transcript}`.trim());
    r.onerror = (e) => setVoiceNote(e.error === 'not-allowed' ? 'Microphone permission was denied.' : 'Voice input failed. You can type instead.');
    r.onend = () => setListening(false);
    recogniser.current = r;
    setVoiceNote('');
    setListening(true);
    r.start();
  }

  return (
    <Panel title="Chat" className="chat">
      <div className="thread" aria-live="polite">
        {tasks.length === 0 && (
          <Empty icon={online ? <Send size={20} /> : <WifiOff size={20} />} title={online ? 'Tell DREAM what to do' : 'Not connected'}>
            {online
              ? (examples.length > 0 ? `Use a supported goal such as ${examples.map((x) => `"${x}"`).join(' or ')}. Progress and outcomes appear here.` : 'Progress and outcomes appear here.')
              : 'Connect to a session to send tasks.'}
          </Empty>
        )}
        {tasks.map((t) => (
          <div key={t.id} className="exchange" data-task-id={t.id}>
            <div className="bubble user">{t.text}</div>
            <AgentBlock task={t} thread={threads[t.id]} stopping={stopping.has(t.id)} modelLabel={describeMeta(choices, t.meta)} />
          </div>
        ))}
        <div ref={bottom} />
      </div>
      {notice && <div className="inline-error" role="alert"><AlertTriangle size={14} /> {notice}</div>}
      <div className="composer-wrap">
        {online && examples.length > 0 && !draft && (
          <div className="examples">
            {examples.map((x) => (
              <button key={x} className="chip-btn" onClick={() => setDraft(fill(x))}>{x}</button>
            ))}
          </div>
        )}
        <div className="composer-tools">
          <ModelPicker choices={choices} value={choice} onChange={onChoose} disabled={!online} />
        </div>
        <div className={`composer ${online ? '' : 'disabled'}`}>
          <textarea
            aria-label="Task"
            rows={1}
            maxLength={500}
            disabled={!online}
            placeholder={placeholder}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                void send();
              }
            }}
          />
          <button
            className={`icon-btn ${listening ? 'rec' : ''}`}
            aria-label={voiceSupported ? (listening ? 'Stop voice input' : 'Start voice input') : 'Voice input is not supported in this browser'}
            title={voiceSupported ? 'Voice input' : 'Voice input is not supported in this browser'}
            disabled={!online || !voiceSupported}
            onClick={toggleVoice}
          >
            {voiceSupported ? <Mic size={18} aria-hidden /> : <MicOff size={18} aria-hidden />}
          </button>
          <button className="send" aria-label="Send task" disabled={!online || !draft.trim() || busy} onClick={() => void send()}>
            {busy ? <Loader2 size={17} className="spin" aria-hidden /> : <Send size={17} aria-hidden />}
          </button>
        </div>
        {voiceNote && <small className="muted">{voiceNote}</small>}
      </div>
    </Panel>
  );
}

/* ---------- robot view ---------------------------------------------------------------------- */

export function RobotView({ snapshot, link, current, thread }: { snapshot: Snapshot | null; link: Link; current?: Task; thread?: Thread }) {
  const stale = link !== 'online';
  const camera = snapshot?.camera;
  const age = camera && snapshot ? Math.max(0, snapshot.robot_time_s - camera.time_s) : null;
  const action = thread?.tools.at(-1);
  // The feed carries no timestamps, so the tool that produced this frame cannot be identified
  // honestly; the overlay therefore states the age only.
  return (
    <Panel title="Robot view" className="robot" meta={<span className="muted small">{camera ? 'Updates when a tool observes' : 'No frame'}</span>}>
      <div className={`viewport ${stale ? 'stale' : ''}`}>
        {camera ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={camera.data_url} alt="Robot head camera frame" />
        ) : (
          <Empty icon={<Camera size={22} />} title="No camera frame yet">
            {link === 'online' ? 'A frame appears once the robot has looked around.' : 'Connect to see what the robot sees.'}
          </Empty>
        )}
        {camera && age != null && (
          <span className="cam-pill">
            <Clock size={13} aria-hidden /> Last observed {age < 10 ? age.toFixed(1) : Math.round(age)} s ago
          </span>
        )}
        {stale && camera && <div className="stale-badge"><WifiOff size={13} aria-hidden /> Disconnected, showing last received frame</div>}
        <div className="view-tags">
          <span>{snapshot?.environment ?? 'Source not labelled'}</span>
        </div>
      </div>
      <dl className="facts">
        <div><dt>Robot clock</dt><dd>{snapshot ? clock(snapshot.robot_time_s) : '—'}</dd></div>
        <div><dt>Frame age</dt><dd>{age == null ? '—' : `${age.toFixed(1)} s`}</dd></div>
        <div>
          <dt>Doing now</dt>
          <dd>{current?.state === 'running' ? (action ? action.name : 'planning') : current?.state === 'suspended' ? 'paused' : 'idle'}</dd>
        </div>
        <div><dt>Safety stop</dt><dd className={snapshot?.stop_reason ? 'bad' : ''}>{snapshot ? (snapshot.stop_reason ?? 'none') : '—'}</dd></div>
      </dl>
    </Panel>
  );
}

/* ---------- spatial map -------------------------------------------------------------------- */

/** Bumps whenever <html data-theme> changes so canvas drawing can re-read the colour tokens. */
function useThemeTick(): number {
  const [tick, setTick] = useState(0);
  useEffect(() => {
    const observer = new MutationObserver(() => setTick((n) => n + 1));
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    return () => observer.disconnect();
  }, []);
  return tick;
}

export function MapPanel({ snapshot }: { snapshot: Snapshot | null }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const map = snapshot?.map;
  const themeTick = useThemeTick();
  useEffect(() => {
    const c = canvas.current;
    if (!c || !map?.cells?.length) return;
    const ctx = c.getContext('2d');
    if (!ctx) return;
    const css = getComputedStyle(document.documentElement);
    const token = (name: string) => css.getPropertyValue(name).trim();
    const cols = map.cells.length;
    const rows = map.cells[0].length;
    const S = Math.max(2, Math.min(10, Math.floor(1600 / Math.max(cols, rows)))); // canvas px per map cell
    c.width = cols * S;
    c.height = rows * S;
    ctx.fillStyle = token('--map-unknown');
    ctx.fillRect(0, 0, c.width, c.height);
    const fills = [token('--map-unknown'), token('--map-free'), token('--map-occupied')];
    for (const { x, y, value } of mapPixels(map.cells)) {
      ctx.fillStyle = fills[value] ?? fills[0];
      ctx.fillRect(x * S, y * S, S, S);
    }
    const unit = Math.max(S * 1.6, Math.max(c.width, c.height) / 48);
    for (const o of snapshot?.objects ?? []) {
      const p = worldToPixel(map, o.position[0], o.position[1]);
      if (!p) continue;
      ctx.strokeStyle = token('--amber');
      ctx.lineWidth = Math.max(1.5, unit / 5);
      ctx.beginPath();
      ctx.arc(p.px * S, p.py * S, unit * 0.55, 0, Math.PI * 2);
      ctx.stroke();
    }
    // No pose history exists in the snapshot, so there is deliberately no trail (--map-trail unused).
    const [x, y, yaw] = snapshot?.odometry ?? [];
    const r = x == null ? null : worldToPixel(map, x, y);
    if (r) {
      ctx.save();
      ctx.translate(r.px * S, r.py * S);
      ctx.fillStyle = token('--map-robot');
      ctx.globalAlpha = 0.22;
      ctx.beginPath();
      ctx.arc(0, 0, unit * 1.1, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.rotate(-(yaw ?? 0)); // canvas y points down, map y points up
      ctx.beginPath();
      ctx.moveTo(unit * 1.0, 0);
      ctx.lineTo(-unit * 0.65, unit * 0.62);
      ctx.lineTo(-unit * 0.3, 0);
      ctx.lineTo(-unit * 0.65, -unit * 0.62);
      ctx.closePath();
      ctx.fill();
      ctx.lineJoin = 'round';
      ctx.lineWidth = Math.max(1.5, unit / 7);
      ctx.strokeStyle = token('--card');
      ctx.stroke();
      ctx.restore();
    }
  }, [map, snapshot?.objects, snapshot?.odometry, themeTick]);

  return (
    <Panel title="Spatial map" className="map" meta={<span className="muted small">Top-down · this session</span>}>
      {map?.cells?.length ? (
        <>
          <div className="mapbox">
            <canvas ref={canvas} role="img" aria-label="Occupancy map with robot position and heading" />
          </div>
          <div className="legend" aria-label="Map legend">
            <span><i className="k-free" /> free</span>
            <span><i className="k-unk" /> unknown</span>
            <span><i className="k-occ" /> occupied</span>
            <span><i className="k-obj" /> object candidate</span>
            <span><i className="k-bot" /> robot (arrow shows heading)</span>
          </div>
        </>
      ) : (
        <Empty icon={<Compass size={22} />} title="No map yet">The map is built from what the robot has observed.</Empty>
      )}
      <p className="footnote">
        {map ? `${map.resolution_m} m per cell · odometry: ${map.odometry}` : 'No spatial data in this session.'}
        {map && snapshot?.odometry ? ` · x ${snapshot.odometry[0].toFixed(2)} m, y ${snapshot.odometry[1].toFixed(2)} m` : ''}
      </p>
    </Panel>
  );
}

/* ---------- current task, queue and activity ------------------------------------------------- */

function PriorityEdit({ task, onPriority }: { task: Task; onPriority: (id: string, priority: number) => void }) {
  const [value, setValue] = useState(String(task.priority));
  useEffect(() => setValue(String(task.priority)), [task.priority]);
  const parsed = Number(value);
  const valid = value.trim() !== '' && Number.isInteger(parsed) && parsed >= -1000 && parsed <= 1000;
  function commit() {
    if (!valid) return setValue(String(task.priority));
    if (parsed !== task.priority) onPriority(task.id, parsed);
  }
  return (
    <label className="prio">
      <span>Priority</span>
      <input
        type="number" inputMode="numeric" step={1} min={-1000} max={1000}
        value={value} aria-invalid={!valid} aria-label={`Priority for ${task.text}`}
        onChange={(e) => setValue(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => { if (e.key === 'Enter') (e.target as HTMLInputElement).blur(); }}
      />
    </label>
  );
}

export type TaskActions = {
  stopping: Set<string>;
  onAction: (id: string, action: 'pause' | 'resume' | 'cancel') => void;
  onPriority: (id: string, priority: number) => void;
};

export function CurrentTaskCard({ current, stopping, onAction, onPriority }: { current?: Task } & TaskActions) {
  if (!current) {
    return (
      <div className="task-card idle">
        <div className="eyebrow">CURRENT TASK</div>
        <Empty icon={<Radio size={20} />} title="Idle">Ready for a supported goal. Submit one in the chat.</Empty>
      </div>
    );
  }
  const runnable = RUNNABLE.includes(current.state);
  const isStopping = stopping.has(current.id) && runnable;
  return (
    <div className={`task-card state-${current.state}`}>
      <div className="task-card-head">
        <span className="eyebrow">{runnable ? 'CURRENT TASK' : 'LAST TASK'}</span>
        <StatePill state={current.state} />
      </div>
      <p className="task-text">{current.text}</p>
      {current.state === 'suspended' && <SuspendNote task={current} />}
      {isStopping && (
        <p className="note tone-info" role="status"><Loader2 size={14} className="spin" aria-hidden /> Waiting for next safe point</p>
      )}
      {runnable && (
        <>
          <div className="controls">
            <button className="btn" disabled={current.state !== 'running' || isStopping} onClick={() => onAction(current.id, 'pause')}>
              <Pause size={15} aria-hidden /> Pause
            </button>
            <button className="btn" disabled={current.state !== 'suspended'} onClick={() => onAction(current.id, 'resume')}>
              <Play size={15} aria-hidden /> Resume
            </button>
            <button className="btn danger" disabled={isStopping} onClick={() => onAction(current.id, 'cancel')}>
              <Square size={14} aria-hidden /> Cancel
            </button>
            {current.state !== 'running' && <PriorityEdit task={current} onPriority={onPriority} />}
          </div>
          <p className="muted small">
            Pause and cancel take effect at the robot&apos;s next safe boundary; the robot is never commanded directly from here.
          </p>
        </>
      )}
    </div>
  );
}

export function TaskQueue({ tasks, shownId, onPriority }: { tasks: Task[]; shownId?: string; onPriority: (id: string, priority: number) => void }) {
  const waiting = tasks.filter((t) => ['queued', 'suspended'].includes(t.state) && t.id !== shownId).sort(schedulerOrder);
  if (!waiting.length) return null;
  return (
    <div className="queue">
      <div className="eyebrow">UP NEXT · PRIORITY, THEN OLDEST FIRST</div>
      <ol>
        {waiting.map((t, i) => (
          <li key={t.id}>
            <span className="rank" aria-hidden>{i + 1}</span>
            <div className="queue-main">
              <b>{t.text}</b>
              <div className="queue-meta">
                <StatePill state={t.state} />
                {t.state === 'suspended' && <span className={t.resume === 'ask' ? 'muted small' : 'amber small'}>{suspendCopy(t)}</span>}
              </div>
            </div>
            <PriorityEdit task={t} onPriority={onPriority} />
          </li>
        ))}
      </ol>
    </div>
  );
}

/** Recent task events, newest first. Task timestamps are wall-clock; the tool feed has no times. */
export function ActivityList({ tasks }: { tasks: Task[] }) {
  const recent = [...tasks].sort((a, b) => b.updated - a.updated).slice(0, 5);
  return (
    <div className="activity">
      <div className="eyebrow">ACTIVITY</div>
      {recent.length ? (
        <ul>
          {recent.map((t) => (
            <li key={t.id}>
              <StatePill state={t.state} />
              <span className="activity-text">{t.text}</span>
              <time className="muted small">{relativeAge(t.updated)}</time>
            </li>
          ))}
        </ul>
      ) : (
        <p className="muted small">Task events appear here.</p>
      )}
    </div>
  );
}

export function TaskOverview({ tasks, shown, onNavigate, ...actions }: { tasks: Task[]; shown?: Task; onNavigate: () => void } & TaskActions) {
  return (
    <section className="overview-row">
      <div className="overview-task">
        <CurrentTaskCard current={shown} {...actions} />
        <TaskQueue tasks={tasks} shownId={shown?.id} onPriority={actions.onPriority} />
      </div>
      <div className="overview-activity">
        <ActivityList tasks={tasks} />
        <button className="text-link" onClick={onNavigate}>View activity →</button>
      </div>
    </section>
  );
}

/* ---------- execution graph and tool outcomes ----------------------------------------------- */

type StepData = { label: string; status: string; current: boolean };

function StepNode({ data }: NodeProps<Node<StepData>>) {
  const failed = !['succeeded', 'running', 'interrupted'].includes(data.status);
  return (
    <div className={`step tone-${toolTone(data.status)} ${data.current ? 'current' : ''}`}>
      <Handle type="target" position={Position.Top} isConnectable={false} />
      <ToolIcon status={data.status} />
      <span className="step-label">{data.label}</span>
      {failed && <span className="step-fail">{data.status}</span>}
      <Handle type="source" position={Position.Bottom} isConnectable={false} />
    </div>
  );
}
const nodeTypes = { step: StepNode };

export function TaskPanel({
  current, graphTask, thread, snapshot, tasks, ...actions
}: {
  current?: Task; graphTask?: Task; thread?: Thread; snapshot: Snapshot | null; tasks: Task[];
} & TaskActions) {
  const graph = snapshot?.task_graph;
  const forCurrent = graph && graphTask && graph.task_id === graphTask.id ? graph : null;
  const lastIndex = (forCurrent?.nodes.length ?? 0) - 1;
  const nodes: Node[] = useMemo(
    () =>
      (forCurrent?.nodes ?? []).map((n, i) => {
        const status = thread?.tools[i]?.status ?? 'running';
        return {
          id: n.id,
          type: 'step',
          position: { x: 0, y: i * 64 },
          data: { label: n.label, status, current: i === lastIndex && graphTask?.state === 'running' },
          draggable: false,
        };
      }),
    [forCurrent, thread, lastIndex, graphTask?.state],
  );
  const edges: Edge[] = useMemo(
    () => (forCurrent?.edges ?? []).map((e, i) => ({ id: `e${i}`, source: e.source, target: e.target, animated: false, style: { stroke: 'var(--graph-edge)', strokeWidth: 1.2 } })),
    [forCurrent],
  );

  return (
    <Panel title="Current task" className="task">
      <div className="task-top">
        <CurrentTaskCard current={current} {...actions} />
        <TaskQueue tasks={tasks} shownId={current?.id} onPriority={actions.onPriority} />
      </div>

      <Tabs.Root defaultValue="execution"><Tabs.List className="tabs" aria-label="Task details"><Tabs.Trigger value="execution">Execution</Tabs.Trigger><Tabs.Trigger value="graph">Graph</Tabs.Trigger></Tabs.List><Tabs.Content value="graph"><div className="section">
        <div className="section-head"><span>Observed tool sequence (not a plan)</span></div>
        {forCurrent ? (
          <div className="flow" key={nodes.length}>
            <ReactFlow
              nodes={nodes} edges={edges} nodeTypes={nodeTypes} fitView fitViewOptions={{ padding: 0.2, maxZoom: 1 }} nodesDraggable={false} nodesConnectable={false}
              elementsSelectable={false} zoomOnScroll={false} proOptions={{ hideAttribution: true }}
            >
              <Background gap={18} size={1.2} color="var(--dot-grid)" />
            </ReactFlow>
          </div>
        ) : (
          <p className="muted small pad">The graph builds as the robot executes tools.</p>
        )}
      </div>

      </Tabs.Content><Tabs.Content value="execution"><div className="section">
        <div className="section-head"><span>Latest execution tools</span><span>{thread?.tools.length ?? 0}</span></div>
        {thread?.tools.length ? (
          <ul className="tools compact">
            {[...thread.tools].reverse().slice(0, 6).map((t) => <ToolRow key={t.id} tool={t} />)}
          </ul>
        ) : (
          <p className="muted small pad">No tools have run for this task.</p>
        )}
      </div>

      </Tabs.Content></Tabs.Root>
      {snapshot?.outcome && (
        <div className="section">
          <div className="section-head"><span>Latest outcome</span><span>Runtime report</span></div>
          <p className={`outcome pad tone-${snapshot.outcome.state === 'succeeded' ? 'good' : snapshot.outcome.state === 'cancelled' ? 'warn' : 'bad'}`}>
            <b>{endTitle(snapshot.outcome.state)}.</b> {text(snapshot.outcome.reason)}
          </p>
        </div>
      )}
    </Panel>
  );
}

/* ---------- memory and history -------------------------------------------------------------- */

export function MemoryPanel({ snapshot, tasks }: { snapshot: Snapshot | null; tasks: Task[] }) {
  const home = snapshot?.persistent_memory;
  const objects = snapshot?.objects ?? [];
  return (
    <Panel title="Memory" className="memory">
      <Tabs.Root defaultValue="objects" className="memory-tabs"><Tabs.List className="tabs" aria-label="Memory views">
        {([['objects', `Seen this session (${objects.length})`], ['home', 'Home memory'], ['history', `History (${tasks.length})`]] as const).map(([id, label]) => (
          <Tabs.Trigger key={id} value={id}>{label}</Tabs.Trigger>
        ))}
      </Tabs.List>
      <div className="memory-body">
        <Tabs.Content value="objects">{
          (objects.length ? (
            <>
              <p className="footnote top">Candidates detected by the robot in this session. Not persistent memory.</p>
              {/* Extension point: a flat candidate list is all the snapshot has. If a backend source for scene
                  relations (on / in / near) is added later, render them here from that data; none are inferred. */}
              <div className="chips">
                {objects.map((o) => (
                  <div key={o.id} className="obj-chip">
                    <i className="swatch" style={o.color ? { background: o.color } : undefined} aria-hidden />
                    <div>
                      <b>Candidate {o.id}{o.color ? ` · ${o.color}` : ''}</b>
                      <span>({o.position[0].toFixed(2)}, {o.position[1].toFixed(2)}) m · {o.views} views</span>
                    </div>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <Empty icon={<Box size={20} />} title="Nothing observed yet">Objects appear after the robot looks around.</Empty>
          ))}</Tabs.Content>
        <Tabs.Content value="home">{
          (home?.available ? (
            home.objects?.length ? (
              <>
                <p className="footnote top">{home.source}</p>
                {home.objects.map((o) => (
                  <div key={o.id} className="row">
                    <Box size={15} />
                    <div>
                      <b>{o.label}</b>
                      <span>last recorded in {o.map_id} at ({o.position.slice(0, 2).map((v) => v.toFixed(2)).join(', ')}) m · {o.observation_source}</span>
                    </div>
                  </div>
                ))}
              </>
            ) : (
              <Empty icon={<Box size={20} />} title="Home memory is empty">{home.source}</Empty>
            )
          ) : (
            <Empty icon={<Box size={20} />} title="Home memory not connected">
              {home?.error ?? 'Start the backend with --memory <home-memory.sqlite> to show stored sightings (read-only).'}
            </Empty>
          ))}</Tabs.Content>
        <Tabs.Content value="history">{
          (tasks.length ? (
            [...tasks].reverse().map((t) => (
              <div key={t.id} className="row">
                <History size={15} />
                <div>
                  <b>{t.text}</b>
                  <span><em className={`state state-${t.state}`}>{t.state}</em> · {t.steps} steps{t.result ? ` · ${t.result}` : ''}</span>
                </div>
              </div>
            ))
          ) : (
            <Empty icon={<History size={20} />} title="No tasks yet" />
          ))}</Tabs.Content>
      </div></Tabs.Root>
    </Panel>
  );
}

/* ---------- toasts and notifications -------------------------------------------------------- */

export type Toast = Notice & { key: number };

export function Toasts({ toasts, onDismiss }: { toasts: Toast[]; onDismiss: (key: number) => void }) {
  return (
    <div className="toasts" aria-live="polite">
      {toasts.map((t) => (
        <div key={t.key} className={`toast tone-${t.tone}`} role="status">
          {t.tone === 'bad' ? <XCircle size={16} /> : t.tone === 'good' ? <CheckCircle2 size={16} /> : t.tone === 'warn' ? <AlertTriangle size={16} /> : <CircleDot size={16} />}
          <div><b>{t.title}</b>{t.body ? <span>{text(t.body)}</span> : null}</div>
          <button aria-label="Dismiss" onClick={() => onDismiss(t.key)}>×</button>
        </div>
      ))}
    </div>
  );
}

export function NotifyButton({ enabled, onToggle }: { enabled: boolean; onToggle: () => void }) {
  return (
    <button className="icon-btn" aria-pressed={enabled} aria-label={enabled ? 'Disable desktop notifications' : 'Enable desktop notifications'} title="Desktop notifications" onClick={onToggle}>
      {enabled ? <Bell size={16} /> : <BellOff size={16} />}
    </button>
  );
}
