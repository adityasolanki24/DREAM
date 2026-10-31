import type { Thread } from '../feed.mjs';
import type { Link, Task } from '../types';

/** DREAM's visible state, derived only from the task, the agent-event feed, the safety stop and the link. */
export type PresenceState =
  | 'offline' | 'idle' | 'listening' | 'thinking' | 'navigating' | 'observing' | 'interacting' | 'verifying'
  | 'recovering' | 'paused' | 'waiting' | 'completed' | 'failed' | 'safety-stopped';

/** `tool` is the running tool's name (shown as code); `detail` is plain language. */
export type Presence = { state: PresenceState; title: string; detail: string; tool?: string };

/** A finished task stops defining DREAM's state after this long; it is idle again. */
const RECENT_S = 90;

export function presenceFor(
  task: Task | undefined, thread: Thread | undefined, safetyStop: string | null, link: Link, listening = false,
): Presence {
  if (link !== 'online') return { state: 'offline', title: link === 'connecting' ? 'Connecting' : 'Not connected', detail: link === 'connecting' ? 'Reaching the robot backend' : 'No live session' };
  if (listening) return { state: 'listening', title: 'Listening', detail: 'Speak a goal; it is transcribed into the composer' };
  if (safetyStop) return { state: 'safety-stopped', title: 'Safety stopped', detail: safetyStop };
  if (!task) return { state: 'idle', title: 'Ready', detail: 'Waiting for a goal' };
  const recent = Date.now() / 1000 - task.updated < RECENT_S;
  const last = { succeeded: 'the runtime reported success', failed: 'it failed', cancelled: 'it was cancelled' }[task.state];
  if (last && !recent) return { state: 'idle', title: 'Ready', detail: `Last task “${task.text}”: ${last}` };
  if (task.state === 'succeeded') return { state: 'completed', title: 'Task ended', detail: 'The runtime reports success' };
  if (task.state === 'failed') return { state: 'failed', title: 'Task failed', detail: task.result ?? 'See the activity for details' };
  if (task.state === 'cancelled') return { state: 'idle', title: 'Ready', detail: 'The last task was cancelled' };
  if (task.state === 'queued') return { state: 'waiting', title: 'Queued', detail: 'Waiting for the robot to be free' };
  if (task.state === 'suspended') return { state: 'paused', title: 'Paused', detail: task.resume === 'ask' ? 'Waiting for you to resume' : 'Resumes when next in priority' };
  const action = thread?.tools.at(-1);
  if (action?.status === 'failed') return { state: 'recovering', title: 'Recovering', detail: 'failed; choosing what to do next', tool: action.name };
  if (action?.status === 'running') {
    const name = action.name.toLowerCase();
    if (/navigate|search|move_to|go_to|explore|walk|approach/.test(name)) return { state: 'navigating', title: 'Navigating', detail: 'moving through the space', tool: action.name };
    if (/look|observe|inspect|camera|view|perceive|scan/.test(name)) return { state: 'observing', title: 'Observing', detail: 'looking with the head camera', tool: action.name };
    if (/verify|finish|check/.test(name)) return { state: 'verifying', title: 'Verifying', detail: 'checking the result', tool: action.name };
    return { state: 'interacting', title: 'Acting', detail: 'running a tool', tool: action.name };
  }
  return { state: 'thinking', title: 'Thinking', detail: 'Planning the next action' };
}

/** The orb's visual family for each state. */
export const ORB_MODE: Record<PresenceState, 'rest' | 'listen' | 'think' | 'act' | 'observe' | 'warn' | 'stop' | 'done' | 'off'> = {
  offline: 'off', idle: 'rest', listening: 'listen', thinking: 'think', verifying: 'think', navigating: 'act', interacting: 'act',
  observing: 'observe', recovering: 'warn', paused: 'rest', waiting: 'rest', completed: 'done', failed: 'warn', 'safety-stopped': 'stop',
};
