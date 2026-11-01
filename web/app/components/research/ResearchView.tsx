'use client';
import * as Tabs from '@radix-ui/react-tabs';
import { AnimatePresence, motion } from 'motion/react';
import { Camera, Map as MapIcon, Maximize2, Minimize2, Radio, ShieldAlert } from 'lucide-react';
import { useEffect, useState, type ReactNode } from 'react';
import { Group, Panel, Separator, useDefaultLayout } from 'react-resizable-panels';
import { describeMeta } from '../../../modelChoice.mjs';
import { useCameraState } from '../../lib/camera';
import { RUNNABLE, relativeAge, stateTone, suspendCopy, text, wallTime } from '../../lib/format';
import { layoutStorage, useMediaQuery, usePref } from '../../lib/prefs';
import { useSession } from '../../lib/session';
import { useUi } from '../../lib/ui';
import type { Task } from '../../types';
import { CameraFrame } from '../camera/CameraFrame';
import { Composer } from '../chat/Composer';
import { PriorityEdit, TaskControls, TaskQueue } from '../task/TaskControls';
import { Quiet, Segmented, StateChip, Tip } from '../ui';
import { WorldMap } from '../world/WorldMap';
import { Diagnostics, MemoryView } from './Inspectors';
import { JsonView } from './JsonView';
import { SceneGraph } from './SceneGraph';
import { TaskGraph } from './TaskGraph';
import { Timeline } from './Timeline';

type PaneId = 'inspect' | 'observe' | 'context';

function PaneHeader({ title, children, pane, focused, onFocus }: { title: ReactNode; children?: ReactNode; pane: PaneId; focused: boolean; onFocus: (p: PaneId | null) => void }) {
  return (
    <header className="pane-head">
      <h2>{title}</h2>
      <div className="pane-tools">
        {children}
        <Tip label={focused ? 'Exit focus' : 'Focus this view'} kbd={focused ? 'Esc' : undefined}>
          <button className="icon-btn sm" aria-label={focused ? 'Exit focus' : `Focus ${typeof title === 'string' ? title : 'view'}`} onClick={() => onFocus(focused ? null : pane)}>
            {focused ? <Minimize2 size={13} /> : <Maximize2 size={13} />}
          </button>
        </Tip>
      </div>
    </header>
  );
}

/* ---------- mission rail: current task, queue, experiment history ---------- */

function MissionRail({ inspectedId, onInspect }: { inspectedId?: string; onInspect: (id: string) => void }) {
  const { shown, tasks, stopping } = useSession();
  const runnable = shown && RUNNABLE.includes(shown.state);
  const history = [...tasks].reverse();
  return (
    <aside className="mission scroll" aria-label="Mission">
      <section className="mission-current">
        <div className="rail-label">{runnable ? 'Current task' : shown ? 'Last task' : 'Current task'}</div>
        {shown ? (
          <>
            <button className={`mission-task ${inspectedId === shown.id ? 'selected' : ''}`} onClick={() => onInspect(shown.id)}>
              <span className="mission-text">{shown.text}</span>
              <StateChip state={shown.state} />
            </button>
            {shown.state === 'suspended' && <p className={`note ${shown.resume === 'ask' ? 'tone-muted' : 'tone-warn'}`}>{suspendCopy(shown)}</p>}
            {stopping.has(shown.id) && runnable && <p className="note tone-info" role="status">Waiting for next safe point</p>}
            {runnable && <div className="mission-actions"><TaskControls task={shown} size="sm" />{shown.state !== 'running' && <PriorityEdit task={shown} />}</div>}
            {runnable && <p className="faint small">Pause and cancel take effect at the robot&apos;s next safe boundary; the robot is never commanded directly from here.</p>}
          </>
        ) : (
          <Quiet icon={<Radio size={16} />} title="Idle">Submit a supported goal from the conversation.</Quiet>
        )}
      </section>
      <TaskQueue tasks={tasks} shownId={shown?.id} />
      <section className="mission-history">
        <div className="rail-label">History <span className="faint">· {tasks.length}</span></div>
        {history.length ? (
          <ol>
            {history.map((t) => (
              <li key={t.id}>
                <button className={inspectedId === t.id ? 'selected' : ''} onClick={() => onInspect(t.id)} aria-current={inspectedId === t.id ? 'true' : undefined}>
                  <span className={`dot tone-${stateTone(t.state)}`} aria-hidden />
                  <span className="h-text">{t.text}</span>
                  <span className="h-meta mono">{t.steps} st · {relativeAge(t.updated)}</span>
                  <span className="sr-only">{t.state}</span>
                </button>
              </li>
            ))}
          </ol>
        ) : <p className="faint small">No tasks in this session yet.</p>}
      </section>
      <section className="mission-compose">
        <div className="rail-label">New goal</div>
        <Composer variant="panel" />
      </section>
    </aside>
  );
}

/* ---------- inspection: reasoning / tool timeline, graph, raw ---------- */

function Verification({ task }: { task: Task }) {
  const { feed, snapshot } = useSession();
  const thread = feed.threads[task.id];
  const end = thread?.end;
  const claim = thread?.claim;
  if (!end && !claim && !snapshot?.stop_reason) return null;
  return (
    <div className="verify">
      {end && (
        <div className={`verify-cell tone-${end.state === 'succeeded' ? 'good' : end.state === 'cancelled' ? 'warn' : end.state === 'suspended' ? 'muted' : 'bad'}`}>
          <span className="verify-k">Runtime outcome</span>
          <span className="verify-v"><b>{end.state}</b> {text(end.reason)}</span>
        </div>
      )}
      {claim && (
        <div className={`verify-cell tone-${claim.success ? 'good' : 'bad'}`}>
          <span className="verify-k">Agent finish claim · self-reported</span>
          <span className="verify-v"><b>{claim.success ? 'success' : 'not achieved'}</b> {text(claim.reason)}</span>
        </div>
      )}
      {snapshot?.stop_reason && task.state === 'running' && (
        <div className="verify-cell tone-bad"><span className="verify-k"><ShieldAlert size={12} aria-hidden /> Safety stop</span><span className="verify-v">{snapshot.stop_reason}</span></div>
      )}
    </div>
  );
}

function Inspect({ task, focused, onFocus }: { task?: Task; focused: boolean; onFocus: (p: PaneId | null) => void }) {
  const { feed, snapshot, models } = useSession();
  const [tab, setTab] = usePref<'timeline' | 'graph' | 'raw'>('research-tab', 'timeline', (v) => ['timeline', 'graph', 'raw'].includes(v as string));
  const thread = task ? feed.threads[task.id] : undefined;
  const model = task ? describeMeta(models.choices, task.meta) : null;
  return (
    <section className="pane inspect">
      <Tabs.Root value={tab} onValueChange={(v) => setTab(v as typeof tab)} className="inspect-tabs">
        <PaneHeader pane="inspect" focused={focused} onFocus={onFocus} title="Reasoning & actions">
          <Tabs.List className="tabs" aria-label="Inspection view">
            <Tabs.Trigger value="timeline">Timeline</Tabs.Trigger>
            <Tabs.Trigger value="graph">Graph</Tabs.Trigger>
            <Tabs.Trigger value="raw">Raw</Tabs.Trigger>
          </Tabs.List>
        </PaneHeader>
        {task ? (
          <div className="inspect-body scroll">
            <div className="inspect-title">
              <h1>{task.text}</h1>
              <div className="inspect-meta">
                <StateChip state={task.state} />
                {model && <span className="meta-pill">{model}</span>}
                <span className="meta-pill mono">{task.steps} steps</span>
                {task.suspensions > 0 && <span className="meta-pill mono">{task.suspensions} suspensions</span>}
                <span className="meta-pill mono">priority {task.priority}</span>
                <span className="meta-pill mono" title={new Date(task.created * 1000).toString()}>started {wallTime(task.created)}</span>
              </div>
            </div>
            <Verification task={task} />
            <Tabs.Content value="timeline"><Timeline thread={thread} /></Tabs.Content>
            <Tabs.Content value="graph" className="graph-tab"><TaskGraph snapshot={snapshot} task={task} thread={thread} /></Tabs.Content>
            <Tabs.Content value="raw" className="raw-tab">
              <JsonView value={task} label="Task (GET /tasks)" />
              <JsonView value={thread ?? null} label="Agent-event thread (AG-UI feed)" />
            </Tabs.Content>
          </div>
        ) : (
          <div className="inspect-body"><Quiet title="Nothing to inspect yet">Each task&apos;s tool calls, arguments, results and outcome appear here once it runs.</Quiet></div>
        )}
      </Tabs.Root>
    </section>
  );
}

/* ---------- observation and context ---------- */

function Observation({ focused, onFocus }: { focused: boolean; onFocus: (p: PaneId | null) => void }) {
  const { snapshot } = useSession();
  const cam = useCameraState();
  const [view, setView] = usePref<'camera' | 'map'>('research-observe', 'camera', (v) => v === 'camera' || v === 'map');
  return (
    <section className="pane observe">
      <PaneHeader pane="observe" focused={focused} onFocus={onFocus} title="Observation">
        <Segmented label="Observation view" value={view} onChange={setView} size="sm" options={[
          { id: 'camera', label: 'Camera', icon: <Camera size={12} aria-hidden /> },
          { id: 'map', label: 'Map', icon: <MapIcon size={12} aria-hidden /> },
        ]} />
      </PaneHeader>
      <div className="observe-body">
        {view === 'camera' ? (
          cam.source === 'none'
            ? <Quiet icon={<Camera size={16} />} title="No camera frame">{cam.disconnected ? 'The session is not connected.' : 'Frames stream from the media server or arrive when a tool observes.'}</Quiet>
            : <div className="observe-frame"><CameraFrame /><span className="observe-src mono">{cam.transportLabel} · {cam.statusLabel}</span></div>
        ) : <WorldMap snapshot={snapshot} legend />}
      </div>
    </section>
  );
}

function Context({ focused, onFocus }: { focused: boolean; onFocus: (p: PaneId | null) => void }) {
  const { snapshot } = useSession();
  const [tab, setTab] = usePref<'world' | 'memory' | 'diagnostics' | 'graph'>('research-context', 'diagnostics', (v) => ['world', 'memory', 'diagnostics', 'graph'].includes(v as string));
  return (
    <section className="pane context">
      <Tabs.Root value={tab} onValueChange={(v) => setTab(v as typeof tab)} className="context-tabs">
        <PaneHeader pane="context" focused={focused} onFocus={onFocus} title={
          <Tabs.List className="tabs" aria-label="Context">
            <Tabs.Trigger value="diagnostics">Diagnostics</Tabs.Trigger>
            <Tabs.Trigger value="world">Map</Tabs.Trigger>
            <Tabs.Trigger value="memory">Memory</Tabs.Trigger>
            <Tabs.Trigger value="graph" title="Semantic scene graph — memory entities and stored relationships">Graph</Tabs.Trigger>
          </Tabs.List>
        } />
        <div className="context-body scroll">
          <Tabs.Content value="diagnostics"><Diagnostics /></Tabs.Content>
          <Tabs.Content value="world" className="world-tab"><WorldMap snapshot={snapshot} legend /></Tabs.Content>
          <Tabs.Content value="memory"><MemoryView /></Tabs.Content>
          <Tabs.Content value="graph" className="sgraph-tab"><SceneGraph /></Tabs.Content>
        </div>
      </Tabs.Root>
    </section>
  );
}

export function ResearchView() {
  const { shown, tasks } = useSession();
  const { inspected, setInspected } = useUi();
  const [focused, setFocused] = useState<PaneId | null>(null);
  const compact = useMediaQuery('(max-width: 980px)');
  const cols = useDefaultLayout({ id: 'research-cols', storage: layoutStorage });
  const side = useDefaultLayout({ id: 'research-side', storage: layoutStorage });
  const task = tasks.find((t) => t.id === inspected) ?? shown;
  useEffect(() => {
    if (!focused) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setFocused(null); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [focused]);

  const pane = (id: PaneId) => id === 'inspect'
    ? <Inspect task={task} focused={focused === id} onFocus={setFocused} />
    : id === 'observe' ? <Observation focused={focused === id} onFocus={setFocused} /> : <Context focused={focused === id} onFocus={setFocused} />;

  if (compact) {
    return (
      <div className="research is-compact">
        <Tabs.Root defaultValue="mission" className="research-mobile">
          <Tabs.List className="tabs research-mobile-tabs" aria-label="Research sections">
            <Tabs.Trigger value="mission">Mission</Tabs.Trigger>
            <Tabs.Trigger value="inspect">Actions</Tabs.Trigger>
            <Tabs.Trigger value="observe">Observe</Tabs.Trigger>
            <Tabs.Trigger value="context">Context</Tabs.Trigger>
          </Tabs.List>
          <Tabs.Content value="mission" className="research-mobile-pane"><MissionRail inspectedId={task?.id} onInspect={setInspected} /></Tabs.Content>
          <Tabs.Content value="inspect" className="research-mobile-pane">{pane('inspect')}</Tabs.Content>
          <Tabs.Content value="observe" className="research-mobile-pane">{pane('observe')}</Tabs.Content>
          <Tabs.Content value="context" className="research-mobile-pane">{pane('context')}</Tabs.Content>
        </Tabs.Root>
      </div>
    );
  }

  return (
    <div className="research">
      <Group orientation="horizontal" id="research-cols" className="research-group" defaultLayout={cols.defaultLayout} onLayoutChanged={cols.onLayoutChanged}>
        <Panel id="mission" defaultSize={280} minSize={230} maxSize={420} className="research-panel">
          <MissionRail inspectedId={task?.id} onInspect={setInspected} />
        </Panel>
        <Separator className="resize-handle" />
        <Panel id="inspect" minSize="32%" className="research-panel">{focused === 'inspect' ? <div className="pane-placeholder" /> : pane('inspect')}</Panel>
        <Separator className="resize-handle" />
        <Panel id="side" defaultSize="34%" minSize={280} className="research-panel">
          <Group orientation="vertical" id="research-side" defaultLayout={side.defaultLayout} onLayoutChanged={side.onLayoutChanged}>
            <Panel id="observe" defaultSize="42%" minSize={160}>{focused === 'observe' ? <div className="pane-placeholder" /> : pane('observe')}</Panel>
            <Separator className="resize-handle horizontal" />
            <Panel id="context" minSize={160}>{focused === 'context' ? <div className="pane-placeholder" /> : pane('context')}</Panel>
          </Group>
        </Panel>
      </Group>
      <AnimatePresence>
        {focused && (
          <motion.div className="focus-layer" initial={{ opacity: 0, scale: 0.985 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.985 }} transition={{ duration: 0.18, ease: [0.22, 1, 0.36, 1] }}>
            {pane(focused)}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
