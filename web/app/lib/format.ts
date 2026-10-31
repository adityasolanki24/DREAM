import type { ProviderEntry } from '../../modelChoice.mjs';
import type { Task } from '../types';

export const text = (v: unknown): string => (v == null ? '' : typeof v === 'string' ? v : JSON.stringify(v));
export const clock = (seconds: number) => `${seconds.toFixed(1)} s`;

export function relativeAge(epochSeconds: number): string {
  const d = Math.max(0, Date.now() / 1000 - epochSeconds);
  if (d < 5) return 'just now';
  if (d < 90) return `${Math.round(d)} s ago`;
  if (d < 5400) return `${Math.round(d / 60)} min ago`;
  if (d < 172800) return `${Math.round(d / 3600)} h ago`;
  return `${Math.round(d / 86400)} d ago`;
}

export const wallTime = (epochSeconds: number) =>
  new Date(epochSeconds * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

export const RUNNABLE = ['running', 'suspended', 'queued'];
export const TERMINAL = ['succeeded', 'failed', 'cancelled'];

/** The scheduler's order: priority DESC, created ASC. */
export const schedulerOrder = (a: Task, b: Task) => b.priority - a.priority || a.created - b.created;

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

export const toolTone = (status: string) =>
  status === 'succeeded' ? 'good' : status === 'running' ? 'info' : status === 'interrupted' ? 'warn' : 'bad';

export const stateTone = (state: string) =>
  state === 'running' ? 'info' : state === 'succeeded' ? 'good' : state === 'failed' ? 'bad' : state === 'suspended' || state === 'cancelled' ? 'warn' : 'muted';

export const STATE_LABEL: Record<string, string> = {
  queued: 'Queued',
  running: 'Running',
  suspended: 'Suspended',
  succeeded: 'Runtime reports success',
  failed: 'Failed',
  cancelled: 'Cancelled',
};

/** Compact provider/model label for tight places; the full label stays available as a tooltip. */
export function shortModelLabel(choices: ProviderEntry[], provider: string | undefined, model: string | null | undefined): string | null {
  if (!provider) return null;
  if (provider === 'structured') return 'Structured parser';
  if (model) return model;
  return choices.find((c) => c.id === provider)?.label ?? provider;
}
