'use client';
import * as Tabs from '@radix-ui/react-tabs';
import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { Background, ReactFlow, type Edge, type Node } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import {
  AlertTriangle, Bell, BellOff, Box, Camera, CheckCircle2, CircleDot, Compass, History, KeyRound, Loader2,
  Mic, MicOff, Pause, Play, Send, Square, WifiOff, XCircle,
} from 'lucide-react';
import { argSummary, endTitle, type Notice, type Thread, type Tool } from './feed.mjs';
import { mapPixels, worldToPixel } from './map.mjs';
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
  if (status === 'running') return <Loader2 size={14} className="spin" />;
  if (status === 'succeeded') return <CheckCircle2 size={14} />;
  if (status === 'interrupted') return <Pause size={14} />;
  return <AlertTriangle size={14} />;
}

const toolTone = (status: string) => (status === 'succeeded' ? 'good' : status === 'running' ? 'info' : status === 'interrupted' ? 'warn' : 'bad');

/* ---------- connection ---------------------------------------------------------------------- */

const LINK_LABEL: Record<Link, string> = {
  idle: 'Not connected',
  connecting: 'Connecting',
  online: 'Connected',
  offline: 'Backend unreachable',
  unauthorised: 'Token rejected',
};

export function ConnectionBadge({ link }: { link: Link }) {
  return (
    <span className={`conn conn-${link}`} role="status">
      <i />
      {LINK_LABEL[link]}
    </span>
  );
}

export function ConnectCard({ link, error, onConnect }: { link: Link; error: string; onConnect: (token: string) => void }) {
  const [value, setValue] = useState('');
  return (
    <form
      className="connect"
      onSubmit={(e) => {
        e.preventDefault();
        if (value.trim()) onConnect(value.trim());
      }}
    >
      <div className="connect-icon">{link === 'offline' ? <WifiOff size={20} /> : <KeyRound size={20} />}</div>
      <h2>{link === 'offline' ? 'Cannot reach the robot backend' : 'Connect to DREAM'}</h2>
      <p>
        {error ||
          'Enter the access token for this session. It is kept in this tab only, sent as a request header, and never put in a URL.'}
      </p>
      <input
        type="password"
        autoComplete="off"
        aria-label="Access token"
        placeholder="Session access token"
        value={value}
        onChange={(e) => setValue(e.target.value)}
      />
      <button className="btn primary" type="submit" disabled={!value.trim() || link === 'connecting'}>
        {link === 'connecting' ? <Loader2 size={15} className="spin" /> : null} Connect
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

const STATE_LABEL: Record<string, string> = {
  queued: 'Queued',
  running: 'Working',
  suspended: 'Paused',
  succeeded: 'Runtime reports success',
  failed: 'Failed',
  cancelled: 'Cancelled',
};

function AgentBlock({ task, thread, stopping }: { task: Task; thread?: Thread; stopping: boolean }) {
  const state = thread?.end?.state ?? task.state;
  const done = ['succeeded', 'failed', 'cancelled'].includes(state);
  const reason = text(thread?.end?.reason ?? task.result);
  return (
    <div className={`bubble agent state-${state}`}>
      <div className="bubble-head">
        <span className="who">DREAM</span>
        <span className={`chip chip-${state}`}>{stopping && !done ? 'Stopping at next safe boundary' : (STATE_LABEL[state] ?? state)}</span>
      </div>
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
      {state === 'suspended' && <p className="outcome tone-warn">Paused at a safe boundary. Resume to continue, or cancel.</p>}
    </div>
  );
}

export function Chat({
  tasks, threads, running, stopping, link, examples, providerLabel, onSend, notice,
}: {
  tasks: Task[]; threads: Record<string, Thread>; running: string | null; stopping: Set<string>; link: Link;
  examples: string[]; providerLabel?: string; onSend: (text: string) => Promise<boolean>; notice: string;
}) {
  const [draft, setDraft] = useState('');
  const [busy, setBusy] = useState(false);
  const [listening, setListening] = useState(false);
  const [voiceNote, setVoiceNote] = useState('');
  const bottom = useRef<HTMLDivElement>(null);
  const recogniser = useRef<{ stop: () => void } | null>(null);
  const [voiceSupported, setVoiceSupported] = useState(false);
  const online = link === 'online';

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
    <Panel title="Chat" className="chat" meta={<span className="muted small">{providerLabel ?? 'Goal parser unavailable'}</span>}>
      <div className="thread" aria-live="polite">
        {tasks.length === 0 && (
          <Empty icon={<CircleDot size={20} />} title={online ? 'Tell DREAM what to do' : 'Not connected'}>
            {online ? 'Use a supported goal such as "inspect cabinet" or "goto entrance". Progress and outcomes appear here.' : 'Connect to a session to send tasks.'}
          </Empty>
        )}
        {tasks.map((t) => (
          <div key={t.id} className="exchange" data-task-id={t.id}>
            <div className="bubble user">{t.text}</div>
            <AgentBlock task={t} thread={threads[t.id]} stopping={stopping.has(t.id)} />
          </div>
        ))}
        <div ref={bottom} />
      </div>
      {notice && <div className="inline-error" role="alert"><AlertTriangle size={14} /> {notice}</div>}
      <div className="composer-wrap">
        {online && examples.length > 0 && !draft && (
          <div className="examples">
            {examples.map((x) => (
              <button key={x} className="chip-btn" onClick={() => setDraft(x)}>{x}</button>
            ))}
          </div>
        )}
        <div className={`composer ${online ? '' : 'disabled'}`}>
          <textarea
            aria-label="Task"
            rows={1}
            maxLength={500}
            disabled={!online}
            placeholder={online ? 'Ask DREAM to do something…' : 'Connect to send a task'}
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
            {voiceSupported ? <Mic size={17} /> : <MicOff size={17} />}
          </button>
          <button className="btn primary send" aria-label="Send task" disabled={!online || !draft.trim() || busy} onClick={() => void send()}>
            {busy ? <Loader2 size={16} className="spin" /> : <Send size={16} />}
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
  return (
    <Panel
      title="Robot view"
      className="robot"
      meta={
        <span className={`live ${camera && !stale ? 'on' : ''}`}>
          <i /> {!camera ? 'No frame' : stale ? 'Last frame (stale)' : 'Latest frame'}
        </span>
      }
    >
      <div className={`viewport ${stale ? 'stale' : ''}`}>
        {camera ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={camera.data_url} alt="Robot head camera frame" />
        ) : (
          <Empty icon={<Camera size={22} />} title="No camera frame yet">
            {link === 'online' ? 'A frame appears once the robot has looked around.' : 'Connect to see what the robot sees.'}
          </Empty>
        )}
        {stale && camera && <div className="stale-badge"><WifiOff size={13} /> Disconnected, showing last received frame</div>}
        <div className="view-tags">
          <span>{snapshot?.environment ?? 'Source not labelled'}</span>
          {camera && <span>frame at {clock(camera.time_s)}</span>}
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

export function MapPanel({ snapshot }: { snapshot: Snapshot | null }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const map = snapshot?.map;
  useEffect(() => {
    const c = canvas.current;
    if (!c || !map?.cells?.length) return;
    const ctx = c.getContext('2d');
    if (!ctx) return;
    c.width = map.cells.length;
    c.height = map.cells[0].length;
    ctx.clearRect(0, 0, c.width, c.height);
    const fills = ['#141414', '#2c2c2c', '#e8e8e8']; // unknown, free, occupied
    for (const { x, y, value } of mapPixels(map.cells)) {
      ctx.fillStyle = fills[value] ?? fills[0];
      ctx.fillRect(x, y, 1, 1);
    }
    for (const o of snapshot?.objects ?? []) {
      const p = worldToPixel(map, o.position[0], o.position[1]);
      if (!p) continue;
      ctx.strokeStyle = '#f5b94a';
      ctx.lineWidth = 0.9;
      ctx.strokeRect(p.px - 1.5, p.py - 1.5, 3, 3);
    }
    const [x, y, yaw] = snapshot?.odometry ?? [];
    const r = x == null ? null : worldToPixel(map, x, y);
    if (r) {
      ctx.save();
      ctx.translate(r.px, r.py);
      ctx.rotate(-(yaw ?? 0));
      ctx.fillStyle = '#6ee7a8';
      ctx.beginPath();
      ctx.moveTo(3.2, 0);
      ctx.lineTo(-2, 2);
      ctx.lineTo(-2, -2);
      ctx.fill();
      ctx.restore();
    }
  }, [map, snapshot?.objects, snapshot?.odometry]);

  return (
    <Panel title="Spatial map" className="map" meta={<span className="muted small">Top-down · this session</span>}>
      {map?.cells?.length ? (
        <div className="mapbox">
          <canvas ref={canvas} aria-label="Occupancy map with robot position" />
          <div className="legend">
            <span><i className="k-free" /> free</span>
            <span><i className="k-occ" /> occupied</span>
            <span><i className="k-obj" /> object candidate</span>
            <span><i className="k-bot" /> robot</span>
          </div>
        </div>
      ) : (
        <Empty icon={<Compass size={22} />} title="No map yet">The map is built from what the robot has observed.</Empty>
      )}
      <p className="footnote">
        {map ? `${map.resolution_m} m per cell · odometry is ${map.odometry}` : 'No spatial data in this session.'}
        {snapshot?.odometry ? ` · x ${snapshot.odometry[0].toFixed(2)} m, y ${snapshot.odometry[1].toFixed(2)} m` : ''}
      </p>
    </Panel>
  );
}

/* ---------- current task, graph and tool outcomes ------------------------------------------- */

export function TaskPanel({
  current, graphTask, thread, snapshot, stopping, onAction,
}: {
  current?: Task; graphTask?: Task; thread?: Thread; snapshot: Snapshot | null; stopping: boolean;
  onAction: (id: string, action: 'pause' | 'resume' | 'cancel') => void;
}) {
  const graph = snapshot?.task_graph;
  const forCurrent = graph && graphTask && graph.task_id === graphTask.id ? graph : null;
  const activeState = current?.state;
  const nodes: Node[] = useMemo(
    () =>
      (forCurrent?.nodes ?? []).map((n, i) => {
        const status = thread?.tools[i]?.status ?? 'running';
        return {
          id: n.id,
          position: { x: (i % 3) * 150, y: Math.floor(i / 3) * 70 },
          data: { label: n.label },
          className: `flow-node tone-${toolTone(status)}`,
          draggable: false,
        };
      }),
    [forCurrent, thread],
  );
  const edges: Edge[] = useMemo(
    () => (forCurrent?.edges ?? []).map((e, i) => ({ id: `e${i}`, source: e.source, target: e.target, animated: false })),
    [forCurrent],
  );

  return (
    <Panel title="Current task" className="task" meta={<span className="muted small">{activeState ? activeState.toUpperCase() : 'IDLE'}</span>}>
      {current && ['running', 'suspended', 'queued'].includes(current.state) ? (
        <div className="task-body">
          <p className="task-text">{current.text}</p>
          <div className="controls">
            <button className="btn" disabled={current.state !== 'running' || stopping} onClick={() => onAction(current.id, 'pause')}>
              <Pause size={15} /> Pause
            </button>
            <button className="btn" disabled={current.state !== 'suspended'} onClick={() => onAction(current.id, 'resume')}>
              <Play size={15} /> Resume
            </button>
            <button className="btn danger" disabled={stopping} onClick={() => onAction(current.id, 'cancel')}>
              <Square size={14} /> {stopping ? 'Stopping…' : 'Cancel'}
            </button>
          </div>
          <p className="muted small">
            Pause and cancel take effect at the robot&apos;s next safe boundary; the robot is never commanded directly from here.
          </p>
        </div>
      ) : (
        <Empty icon={<CircleDot size={20} />} title="No active task">Submit a goal in the chat to begin.</Empty>
      )}

      <Tabs.Root defaultValue="execution"><Tabs.List className="tabs" aria-label="Task details"><Tabs.Trigger value="execution">Execution</Tabs.Trigger><Tabs.Trigger value="graph">Graph</Tabs.Trigger></Tabs.List><Tabs.Content value="graph"><div className="section">
        <div className="section-head"><span>Execution graph</span><span>Observed order, not a plan</span></div>
        {forCurrent ? (
          <div className="flow" key={nodes.length}>
            <ReactFlow
              nodes={nodes} edges={edges} fitView fitViewOptions={{ padding: 0.2, maxZoom: 1 }} nodesDraggable={false} nodesConnectable={false}
              elementsSelectable={false} zoomOnScroll={false} proOptions={{ hideAttribution: true }}
            >
              <Background gap={16} size={1} color="rgba(50,65,80,0.14)" />
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
              {objects.map((o) => (
                <div key={o.id} className="row">
                  <Box size={15} />
                  <div>
                    <b>Candidate {o.id}{o.color ? ` · ${o.color}` : ''}</b>
                    <span>at ({o.position[0].toFixed(2)}, {o.position[1].toFixed(2)}) m · seen in {o.views} views</span>
                  </div>
                </div>
              ))}
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
