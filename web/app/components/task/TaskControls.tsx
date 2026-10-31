'use client';
import { Pause, Play, Square } from 'lucide-react';
import { useEffect, useState } from 'react';
import { RUNNABLE, schedulerOrder, suspendCopy } from '../../lib/format';
import { useSession } from '../../lib/session';
import type { Task } from '../../types';
import { StateChip, Tip } from '../ui';

/**
 * Pause / Resume / Cancel for a runnable task. They take effect at the robot's next safe boundary; nothing here
 * commands the robot directly. `tone="stage"` is the variant drawn over the camera.
 */
export function TaskControls({ task, size, tone }: { task: Task; size?: 'sm'; tone?: 'stage' }) {
  const { act, stopping, link } = useSession();
  if (!RUNNABLE.includes(task.state)) return null;
  const isStopping = stopping.has(task.id);
  const offline = link !== 'online';
  return (
    <div className={`task-controls ${tone === 'stage' ? 'on-stage' : ''}`}>
      {task.state === 'suspended' ? (
        <Tip label="Resume at the next safe point" kbd="P"><button className={`btn ${size ?? ''}`} disabled={offline} onClick={() => act(task.id, 'resume')}><Play size={13} aria-hidden />Resume</button></Tip>
      ) : (
        <Tip label="Pause at the next safe point" kbd="P"><button className={`btn ${size ?? ''}`} disabled={task.state !== 'running' || isStopping || offline} onClick={() => act(task.id, 'pause')}><Pause size={13} aria-hidden />Pause</button></Tip>
      )}
      <Tip label="Cancel at the next safe point" kbd="⇧X"><button className={`btn danger ${size ?? ''}`} disabled={isStopping || offline} onClick={() => act(task.id, 'cancel')}><Square size={11} aria-hidden fill="currentColor" />Cancel</button></Tip>
    </div>
  );
}

export function PriorityEdit({ task }: { task: Task }) {
  const { setPriority } = useSession();
  const [value, setValue] = useState(String(task.priority));
  useEffect(() => setValue(String(task.priority)), [task.priority]);
  const parsed = Number(value);
  const valid = value.trim() !== '' && Number.isInteger(parsed) && parsed >= -1000 && parsed <= 1000;
  function commit() {
    if (!valid) return setValue(String(task.priority));
    if (parsed !== task.priority) void setPriority(task.id, parsed);
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

/** Waiting tasks in the scheduler's order (priority, then oldest first). */
export function TaskQueue({ tasks, shownId }: { tasks: Task[]; shownId?: string }) {
  const waiting = tasks.filter((t) => ['queued', 'suspended'].includes(t.state) && t.id !== shownId).sort(schedulerOrder);
  if (!waiting.length) return null;
  return (
    <div className="queue">
      <div className="rail-label">Up next <span className="faint">· priority, then oldest</span></div>
      <ol>
        {waiting.map((t, i) => (
          <li key={t.id}>
            <span className="rank mono" aria-hidden>{i + 1}</span>
            <div className="queue-main">
              <b>{t.text}</b>
              <div className="queue-meta">
                <StateChip state={t.state} />
                {t.state === 'suspended' && <span className={t.resume === 'ask' ? 'muted small' : 'amber small'}>{suspendCopy(t)}</span>}
              </div>
            </div>
            <PriorityEdit task={t} />
          </li>
        ))}
      </ol>
    </div>
  );
}
