'use client';
import { useCallback, useEffect, useRef, useState } from 'react';
import { Camera, Brain, ListChecks, MessageSquare, LayoutDashboard, Map, Settings, X, Check } from 'lucide-react';
import * as Tabs from '@radix-ui/react-tabs';
import * as Dialog from '@radix-ui/react-dialog';
import * as Tooltip from '@radix-ui/react-tooltip';
import * as Dropdown from '@radix-ui/react-dropdown-menu';
import {
  Chat, ConnectCard, ConnectionBadge, MapPanel, MemoryPanel, NotifyButton, RobotView, TaskOverview, TaskPanel, Toasts, schedulerOrder, type Toast,
} from './components';
import { applyEvents, emptyFeed, type AgEvent, type Feed } from './feed.mjs';
import type { Link, Snapshot, Status, Task } from './types';
import { createPollGate, fetchWithTimeout } from './connection.mjs';

class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

const TABS = [
  { id: 'chat', label: 'Chat', icon: MessageSquare },
  { id: 'robot', label: 'Robot', icon: Camera },
  { id: 'task', label: 'Task', icon: ListChecks },
  { id: 'memory', label: 'Memory', icon: Brain },
] as const;
type Tab = (typeof TABS)[number]['id'];

export default function Page() {
  const [token, setToken] = useState<string | null>(null);
  const [connectionAttempt, setConnectionAttempt] = useState(0);
  const gate = useRef(createPollGate()); // null until sessionStorage was read
  const [link, setLink] = useState<Link>('idle');
  const [error, setError] = useState('');
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [status, setStatus] = useState<Status | null>(null);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [feed, setFeed] = useState<Feed>(emptyFeed());
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [stopping, setStopping] = useState<Set<string>>(new Set());
  const [chatNotice, setChatNotice] = useState('');
  const [tab, setTab] = useState<Tab>('robot');
  const [mobile, setMobile] = useState(false);
  useEffect(() => {
    const query = window.matchMedia('(max-width: 850px)');
    const update = () => setMobile(query.matches);
    update();
    query.addEventListener('change', update);
    return () => query.removeEventListener('change', update);
  }, []);
  const [view, setView] = useState<'overview' | 'activity' | 'world' | 'memory'>('overview');
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [themeMode, setThemeMode] = useState<'system' | 'light' | 'dark'>('system');
  useEffect(() => { setThemeMode(window.dreamTheme?.get() ?? 'system'); }, []);
  const chooseTheme = (m: string) => { const mode = m === 'light' || m === 'dark' ? m : 'system'; window.dreamTheme?.set(mode); setThemeMode(mode); };
  const [notify, setNotify] = useState(false);
  const cursor = useRef(0);
  const streamId = useRef<string | null>(null);
  const feedRef = useRef<Feed>(emptyFeed());
  const lastLink = useRef<Link>('idle');
  const toastKey = useRef(0);
  const notifyRef = useRef(false);
  notifyRef.current = notify;

  useEffect(() => {
    try {
      setToken(sessionStorage.getItem('dream-token') ?? '');
    } catch {
      setToken('');
    }
  }, []);

  const api = useCallback(
    async (path: string, method = 'GET', body?: unknown) => {
      let response: Response;
      let data: any;
      try {
        ({ response, data } = await fetchWithTimeout(fetch, path, {
          method,
          headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
          body: body === undefined ? undefined : JSON.stringify(body),
          cache: 'no-store',
        }));
      } catch {
        throw new ApiError('The DREAM backend did not respond.', 0);
      }
      if (!response.ok) throw new ApiError(data.error || `Request failed (${response.status})`, response.status);
      return data;
    },
    [token],
  );

  const pushToasts = useCallback((notices: { tone: Toast['tone']; title: string; body?: unknown }[]) => {
    const fresh = notices.map((n) => ({ ...n, key: ++toastKey.current })) as Toast[];
    if (!fresh.length) return;
    setToasts((old) => [...old, ...fresh].slice(-4));
    fresh.forEach((t) => setTimeout(() => setToasts((old) => old.filter((x) => x.key !== t.key)), 7000));
    if (notifyRef.current && typeof Notification !== 'undefined' && Notification.permission === 'granted') {
      fresh.filter((t) => t.tone !== 'info').forEach((t) => new Notification(`DREAM: ${t.title}`, { body: typeof t.body === 'string' ? t.body : undefined }));
    }
  }, []);

  const poll = useCallback(async () => {
    if (!token) return;
    return gate.current.run(async (isCurrent) => {
    const reconnecting = lastLink.current !== 'online';
    if (reconnecting) {
      setLink('connecting');
      // A restarted backend renumbers its events: start over rather than miss them.
      cursor.current = 0;
      feedRef.current = emptyFeed();
      setFeed(feedRef.current);
    }
    try {
      const [s, t, snap, firstEvents] = await Promise.all([
        api('/status'), api('/tasks'), api('/snapshot'), api(`/ag-ui/events?after=${cursor.current}`),
      ]);
      if (!isCurrent()) return;
      let ev = firstEvents;
      if (streamId.current && streamId.current !== ev.stream_id) {
        cursor.current = 0;
        feedRef.current = emptyFeed();
        setFeed(feedRef.current);
        ev = await api('/ag-ui/events?after=0');
        if (!isCurrent()) return;
      }
      streamId.current = ev.stream_id;
      const initial = cursor.current === 0;
      cursor.current = ev.cursor;
      if (ev.events.length) {
        const { feed: next, notices } = applyEvents(feedRef.current, ev.events as AgEvent[]);
        feedRef.current = next;
        setFeed(next);
        if (!initial) pushToasts(notices);
      }
      setStatus(s);
      setTasks(t);
      setSnapshot(snap);
      setStopping((old) => new Set([...old].filter((id) => ['running', 'queued'].includes((t as Task[]).find((x) => x.id === id)?.state ?? ''))));
      setError('');
      lastLink.current = 'online';
      setLink('online');
    } catch (e) {
      if (!isCurrent()) return;
      const unauthorised = e instanceof ApiError && e.status === 401;
      lastLink.current = unauthorised ? 'unauthorised' : 'offline';
      setLink(lastLink.current);
      setError(unauthorised ? 'The access token was rejected. Check the token and try again.' : e instanceof Error ? e.message : 'Connection failed.');
    }
    });
  }, [api, token, pushToasts]);

  useEffect(() => {
    if (!token) return;
    let stopped = false;
    let timer: ReturnType<typeof setTimeout>;
    const loop = async () => {
      await poll();
      if (!stopped && lastLink.current !== 'unauthorised') timer = setTimeout(loop, lastLink.current === 'online' ? 1200 : 3500);
    };
    void loop();
    return () => {
      stopped = true;
      gate.current.invalidate();
      clearTimeout(timer);
    };
  }, [token, poll, connectionAttempt]);

  function connect(value: string) {
    try {
      sessionStorage.setItem('dream-token', value);
    } catch {
      /* private mode: the token simply is not remembered */
    }
    gate.current.invalidate();
    lastLink.current = 'idle';
    setLink('connecting');
    setConnectionAttempt((n) => n + 1);
    setToken(value);
  }

  async function submit(textValue: string) {
    try {
      await api('/tasks', 'POST', { text: textValue, priority: 0, provider: snapshot?.provider });
      setChatNotice('');
      await poll();
      return true;
    } catch (e) {
      setChatNotice(e instanceof Error ? e.message : 'The task could not be submitted.');
      return false;
    }
  }

  async function act(id: string, action: 'pause' | 'resume' | 'cancel') {
    try {
      const response = await api(`/tasks/${encodeURIComponent(id)}/${action}`, 'POST', {});
      if (['pausing', 'cancelling'].includes(response.result)) setStopping((old) => new Set(old).add(id));
      await poll();
    } catch (e) {
      pushToasts([{ tone: 'bad', title: `Could not ${action} the task`, body: e instanceof Error ? e.message : '' }]);
    }
  }

  async function setPriority(id: string, priority: number) {
    try {
      await api(`/tasks/${encodeURIComponent(id)}/priority`, 'POST', { priority });
      await poll();
    } catch (e) {
      pushToasts([{ tone: 'bad', title: 'Could not change the priority', body: e instanceof Error ? e.message : '' }]);
    }
  }

  async function toggleNotify() {
    if (notify) return setNotify(false);
    if (typeof Notification === 'undefined') {
      return pushToasts([{ tone: 'warn', title: 'Desktop notifications are not supported in this browser' }]);
    }
    const permission = Notification.permission === 'granted' ? 'granted' : await Notification.requestPermission();
    setNotify(permission === 'granted');
    if (permission !== 'granted') pushToasts([{ tone: 'warn', title: 'Notifications were not enabled' }]);
  }

  const ordered = [...tasks].sort((a, b) => a.created - b.created);
  // The running task, else the next one the scheduler would pick (priority DESC, created ASC).
  const current =
    ordered.find((t) => t.id === status?.running) ?? [...ordered].filter((t) => ['running', 'suspended', 'queued'].includes(t.state)).sort(schedulerOrder)[0];
  const shown = current ?? ordered.at(-1);
  const graphTask = tasks.find((t) => t.id === snapshot?.task_graph?.task_id);
  const threads = feed.threads;
  const everConnected = snapshot !== null;
  const needsConnect = token === '' || link === 'unauthorised' || (!everConnected && link !== 'connecting' && link !== 'online');

  if (token === null) return null;

  const chat = <Chat tasks={ordered} threads={threads} running={status?.running ?? null} stopping={stopping} link={link}
    examples={snapshot?.provider === 'structured' ? ['inspect cabinet', 'goto entrance'] : ['inspect cabinet']}
    providerLabel={snapshot?.provider_label} onSend={submit} notice={chatNotice} />;
  const robot = <RobotView snapshot={snapshot} link={link} current={shown} thread={shown ? threads[shown.id] : undefined} />;
  const map = <MapPanel snapshot={snapshot} />;
  const task = <TaskPanel current={shown} tasks={ordered} graphTask={graphTask} thread={graphTask ? threads[graphTask.id] : undefined} snapshot={snapshot} stopping={stopping} onAction={act} onPriority={setPriority} />;
  const memory = <MemoryPanel snapshot={snapshot} tasks={ordered} />;
  const navItems = [
    { id: 'overview' as const, label: 'Overview', icon: LayoutDashboard },
    { id: 'activity' as const, label: 'Activity', icon: ListChecks },
    { id: 'world' as const, label: 'World', icon: Map },
    { id: 'memory' as const, label: 'Memory', icon: Brain },
  ];

  return (
    <Tooltip.Provider delayDuration={250}>
    <div className="app" data-tab={tab}>
      <aside className="sidebar">
        <div className="side-brand"><span className="mark"><i /><i /><i /></span><span>DREAM</span></div>
        <div className="side-label">WORKSPACE</div>
        <nav aria-label="Workspace">
          {navItems.map(({id, label, icon: Icon}) => <Tooltip.Root key={id}><Tooltip.Trigger asChild>
            <button className={view === id ? 'side-link on' : 'side-link'} aria-current={view === id ? 'page' : undefined} onClick={() => setView(id)}><Icon size={18}/><span>{label}</span></button>
          </Tooltip.Trigger><Tooltip.Portal><Tooltip.Content side="right" className="tooltip">{label}<Tooltip.Arrow /></Tooltip.Content></Tooltip.Portal></Tooltip.Root>)}
        </nav>
        <div className="side-footer"><span className="side-dot"/> Local session <small>MuJoCo simulation</small></div>
      </aside>
      <div className="workspace">
        <header className="topbar">
          <div className="breadcrumb"><span>Mission</span><span className="slash">/</span><b>Workspace</b></div>
          <div className="top-right"><ConnectionBadge link={link} /><NotifyButton enabled={notify} onToggle={toggleNotify} />
            <Dropdown.Root><Dropdown.Trigger asChild><button className="icon-btn" aria-label="Session settings"><Settings size={18}/></button></Dropdown.Trigger>
              <Dropdown.Portal><Dropdown.Content className="dropdown" align="end" sideOffset={8}><Dropdown.Item className="dropdown-item" onSelect={() => setSettingsOpen(true)}>Change access token</Dropdown.Item><Dropdown.Item className="dropdown-item" onSelect={toggleNotify}>{notify ? 'Disable' : 'Enable'} notifications</Dropdown.Item>
                <Dropdown.Separator className="dropdown-sep" /><Dropdown.Label className="dropdown-label">THEME</Dropdown.Label>
                <Dropdown.RadioGroup value={themeMode} onValueChange={chooseTheme}>
                  {(['system', 'light', 'dark'] as const).map((m) => <Dropdown.RadioItem key={m} value={m} className="dropdown-item" onSelect={(e) => e.preventDefault()}>{m[0].toUpperCase() + m.slice(1)}<Dropdown.ItemIndicator className="check"><Check size={16} aria-hidden /></Dropdown.ItemIndicator></Dropdown.RadioItem>)}
                </Dropdown.RadioGroup></Dropdown.Content></Dropdown.Portal>
            </Dropdown.Root>
          </div>
        </header>
        {link === 'offline' && everConnected && <div className="banner" role="alert">Connection lost. Showing the last data received; it is not live. Retrying…</div>}
        {needsConnect ? <main className="center"><ConnectCard link={link} error={error} onConnect={connect} /></main> : <main className="workspace-body">
          <div className="main-area">
            <div className="workspace-heading"><div><div className="eyebrow">MISSION CONTROL</div><h1>{view === 'overview' ? 'Robot workspace' : view === 'activity' ? 'Task activity' : view === 'world' ? 'Spatial world' : 'Memory & history'}</h1></div><span className="source-label">UNITREE G1 · SIMULATION</span></div>
            {!mobile && <div className="desktop-content">
              {view === 'overview' && <><Tabs.Root defaultValue="camera" className="visual-tabs"><div className="visual-toolbar"><Tabs.List aria-label="Robot view" className="view-switch"><Tabs.Trigger value="camera">Camera</Tabs.Trigger><Tabs.Trigger value="map">Map</Tabs.Trigger></Tabs.List><span>ROBOT OBSERVATIONS</span></div><Tabs.Content value="camera" className="visual-content">{robot}</Tabs.Content><Tabs.Content value="map" className="visual-content">{map}</Tabs.Content></Tabs.Root>
                <TaskOverview tasks={ordered} shown={shown} stopping={stopping} onAction={act} onPriority={setPriority} onNavigate={() => setView('activity')} /></>}
              {view === 'activity' && task}{view === 'world' && map}{view === 'memory' && memory}
            </div>}
            {mobile && <div className="mobile-content">{tab === 'robot' && <Tabs.Root defaultValue="camera" className="visual-tabs"><Tabs.List className="view-switch" aria-label="Robot view"><Tabs.Trigger value="camera">Camera</Tabs.Trigger><Tabs.Trigger value="map">Map</Tabs.Trigger></Tabs.List><Tabs.Content value="camera">{robot}</Tabs.Content><Tabs.Content value="map">{map}</Tabs.Content></Tabs.Root>}{tab === 'task' && task}{tab === 'memory' && memory}</div>}
          </div>
          <aside className="conversation" hidden={mobile && tab !== 'chat'}>{chat}</aside>
        </main>}
        <nav className="tabbar" aria-label="Sections">{TABS.map(({ id, label, icon: Icon }) => <button key={id} className={tab === id ? 'on' : ''} aria-current={tab === id ? 'page' : undefined} onClick={() => setTab(id)}><Icon size={20}/><span>{label}</span></button>)}</nav>
      </div>
      <Dialog.Root open={settingsOpen} onOpenChange={setSettingsOpen}><Dialog.Portal><Dialog.Overlay className="dialog-overlay"/><Dialog.Content className="dialog-content"><Dialog.Title>Session connection</Dialog.Title><Dialog.Description>Update the token used for this browser tab.</Dialog.Description><Dialog.Close className="dialog-close" aria-label="Close"><X size={18}/></Dialog.Close><ConnectCard link={link} error={error} onConnect={(value) => { connect(value); setSettingsOpen(false); }}/></Dialog.Content></Dialog.Portal></Dialog.Root>
      <Toasts toasts={toasts} onDismiss={(k) => setToasts((old) => old.filter((t) => t.key !== k))} />
    </div>
    </Tooltip.Provider>
  );
}
