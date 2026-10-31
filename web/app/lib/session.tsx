'use client';
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { toast } from 'sonner';
import { useModelChoice } from '../ModelPicker';
import { buildTaskBody, type ModelChoice, type ProviderEntry } from '../../modelChoice.mjs';
import { applyEvents, emptyFeed, type AgEvent, type Feed, type Notice } from '../feed.mjs';
import { createPollGate, fetchWithTimeout } from '../connection.mjs';
import type { Link, Snapshot, Status, Task } from '../types';
import { RUNNABLE, schedulerOrder } from './format';
import type { Fixture } from './fixtures';

class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

export type TaskAction = 'pause' | 'resume' | 'cancel';

export type Session = {
  token: string | null;
  link: Link;
  error: string;
  snapshot: Snapshot | null;
  status: Status | null;
  /** Oldest first. */
  tasks: Task[];
  feed: Feed;
  stopping: Set<string>;
  chatNotice: string;
  models: { choices: ProviderEntry[]; choice: ModelChoice | null; select: (c: ModelChoice) => void };
  /** The running task, else the next one the scheduler would pick, else the latest task. */
  shown: Task | undefined;
  graphTask: Task | undefined;
  everConnected: boolean;
  needsConnect: boolean;
  stream: { id: string | null; cursor: number };
  notify: boolean;
  /** Set only for development interface-preview fixtures; never robot data. */
  fixture: Fixture | null;
  connect: (token: string) => void;
  submit: (text: string) => Promise<boolean>;
  act: (id: string, action: TaskAction) => Promise<void>;
  setPriority: (id: string, priority: number) => Promise<void>;
  toggleNotify: () => Promise<void>;
};

const SessionContext = createContext<Session | null>(null);

export function useSession(): Session {
  const s = useContext(SessionContext);
  if (!s) throw new Error('useSession outside SessionProvider');
  return s;
}

const toneToToast = { good: toast.success, bad: toast.error, warn: toast.warning, info: toast.info } as const;

function showNotices(notices: Pick<Notice, 'tone' | 'title' | 'body'>[], desktop: boolean) {
  for (const n of notices) {
    const body = n.body == null ? undefined : typeof n.body === 'string' ? n.body : JSON.stringify(n.body);
    toneToToast[n.tone](n.title, { description: body, duration: 7000 });
  }
  if (desktop && typeof Notification !== 'undefined' && Notification.permission === 'granted') {
    notices.filter((t) => t.tone !== 'info').forEach((t) => new Notification(`DREAM: ${t.title}`, { body: typeof t.body === 'string' ? t.body : undefined }));
  }
}

export function SessionProvider({ fixture, children }: { fixture: Fixture | null; children: ReactNode }) {
  const [token, setToken] = useState<string | null>(null);
  const [connectionAttempt, setConnectionAttempt] = useState(0);
  const gate = useRef(createPollGate());
  const [link, setLink] = useState<Link>('idle');
  const [error, setError] = useState('');
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [status, setStatus] = useState<Status | null>(null);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [feed, setFeed] = useState<Feed>(emptyFeed());
  const [stopping, setStopping] = useState<Set<string>>(new Set());
  const [chatNotice, setChatNotice] = useState('');
  const [notify, setNotify] = useState(false);
  const [streamInfo, setStreamInfo] = useState<{ id: string | null; cursor: number }>({ id: null, cursor: 0 });
  const models = useModelChoice((fixture?.snapshot ?? snapshot)?.provider_choices, (fixture?.snapshot ?? snapshot)?.provider);
  const cursor = useRef(0);
  const streamId = useRef<string | null>(null);
  const feedRef = useRef<Feed>(emptyFeed());
  const lastLink = useRef<Link>('idle');
  const notifyRef = useRef(false);
  notifyRef.current = notify;

  useEffect(() => {
    if (fixture) return setToken('');
    try {
      setToken(sessionStorage.getItem('dream-token') ?? '');
    } catch {
      setToken('');
    }
  }, [fixture]);

  const api = useCallback(
    async (path: string, method = 'GET', body?: unknown) => {
      let response: Response;
      let data: any; // eslint-disable-line @typescript-eslint/no-explicit-any
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
        setStreamInfo((old) => (old.id === ev.stream_id && old.cursor === ev.cursor ? old : { id: ev.stream_id, cursor: ev.cursor }));
        if (ev.events.length) {
          const { feed: next, notices } = applyEvents(feedRef.current, ev.events as AgEvent[]);
          feedRef.current = next;
          setFeed(next);
          if (!initial) showNotices(notices, notifyRef.current);
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
  }, [api, token]);

  useEffect(() => {
    if (!token) return;
    let stopped = false;
    let timer: ReturnType<typeof setTimeout>;
    const loop = async () => {
      await poll();
      if (!stopped && lastLink.current !== 'unauthorised') timer = setTimeout(loop, lastLink.current === 'online' ? 1200 : 3500);
    };
    void loop();
    const g = gate.current;
    return () => {
      stopped = true;
      g.invalidate();
      clearTimeout(timer);
    };
  }, [token, poll, connectionAttempt]);

  const connect = useCallback((value: string) => {
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
  }, []);

  const submit = useCallback(async (textValue: string) => {
    if (fixture) {
      toast.info('Interface preview fixture', { description: 'Nothing was sent; fixtures have no backend.' });
      return false;
    }
    if (!models.choice) {
      setChatNotice('No provider is available for new tasks. Check the backend configuration.');
      return false;
    }
    try {
      // A 422 (for example "model is not available for ...") lands in the chat notice; the draft is kept by the caller.
      await api('/tasks', 'POST', buildTaskBody(textValue, models.choice));
      setChatNotice('');
      await poll();
      return true;
    } catch (e) {
      setChatNotice(e instanceof Error ? e.message : 'The task could not be submitted.');
      return false;
    }
  }, [api, fixture, models.choice, poll]);

  const act = useCallback(async (id: string, action: TaskAction) => {
    if (fixture) return void toast.info('Interface preview fixture', { description: `${action} was not sent; fixtures have no backend.` });
    try {
      const response = await api(`/tasks/${encodeURIComponent(id)}/${action}`, 'POST', {});
      if (['pausing', 'cancelling'].includes(response.result)) setStopping((old) => new Set(old).add(id));
      await poll();
    } catch (e) {
      toast.error(`Could not ${action} the task`, { description: e instanceof Error ? e.message : '' });
    }
  }, [api, fixture, poll]);

  const setPriority = useCallback(async (id: string, priority: number) => {
    if (fixture) return void toast.info('Interface preview fixture', { description: 'Priority was not sent.' });
    try {
      await api(`/tasks/${encodeURIComponent(id)}/priority`, 'POST', { priority });
      await poll();
    } catch (e) {
      toast.error('Could not change the priority', { description: e instanceof Error ? e.message : '' });
    }
  }, [api, fixture, poll]);

  const toggleNotify = useCallback(async () => {
    if (notify) return setNotify(false);
    if (typeof Notification === 'undefined') return void toast.warning('Desktop notifications are not supported in this browser');
    const permission = Notification.permission === 'granted' ? 'granted' : await Notification.requestPermission();
    setNotify(permission === 'granted');
    if (permission !== 'granted') toast.warning('Notifications were not enabled');
  }, [notify]);

  const value = useMemo<Session>(() => {
    const live = fixture
      ? { link: fixture.link, snapshot: fixture.snapshot, status: fixture.status, tasks: fixture.tasks, feed: fixture.feed, stopping: new Set<string>() }
      : { link, snapshot, status, tasks, feed, stopping };
    const ordered = [...live.tasks].sort((a, b) => a.created - b.created);
    const current = ordered.find((t) => t.id === live.status?.running)
      ?? [...ordered].filter((t) => RUNNABLE.includes(t.state)).sort(schedulerOrder)[0];
    const everConnected = live.snapshot !== null;
    return {
      token, ...live, error: fixture ? '' : error, tasks: ordered, chatNotice, models,
      shown: current ?? ordered.at(-1),
      graphTask: ordered.find((t) => t.id === live.snapshot?.task_graph?.task_id),
      everConnected,
      needsConnect: !fixture && (token === '' || link === 'unauthorised' || (!everConnected && link !== 'connecting' && link !== 'online')),
      stream: fixture ? { id: 'fixture', cursor: 0 } : streamInfo,
      notify, fixture, connect, submit, act, setPriority, toggleNotify,
    };
  }, [fixture, link, snapshot, status, tasks, feed, stopping, token, error, chatNotice, models, streamInfo, notify, connect, submit, act, setPriority, toggleNotify]);

  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;
}
