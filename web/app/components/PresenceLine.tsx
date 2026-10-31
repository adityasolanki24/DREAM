import type { Presence } from '../lib/presence';
import type { Task } from '../types';

/** "observe_scene · looking with the head camera · inspect cabinet": the tool as code, the rest in plain words. */
export function PresenceLine({ presence, task, className = '' }: { presence: Presence; task?: Task; className?: string }) {
  const active = task && presence.state !== 'idle' && presence.state !== 'offline' && presence.state !== 'listening';
  return (
    <p className={`presence-line ${className}`}>
      {presence.tool && <><code>{presence.tool}</code><span className="sep"> · </span></>}
      <span>{presence.detail}</span>
      {active && <><span className="sep"> · </span><span className="presence-task">{task.text}</span></>}
    </p>
  );
}
