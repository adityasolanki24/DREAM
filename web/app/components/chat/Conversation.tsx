'use client';
import { AnimatePresence, motion } from 'motion/react';
import { CheckCircle2, CircleSlash, Loader2, Pause, XCircle } from 'lucide-react';
import { useEffect, useRef } from 'react';
import { argSummary, endTitle, type Thread, type Tool } from '../../feed.mjs';
import { describeMeta, type ProviderEntry } from '../../../modelChoice.mjs';
import { RUNNABLE, TERMINAL, shortModelLabel, suspendCopy, text, toolTone } from '../../lib/format';
import type { Task } from '../../types';
import ThoughtLine from '../reactbits/ThoughtLine';
import { StrandsGlyph } from '../Strands';
import { StateChip } from '../ui';

export function ToolIcon({ status, size = 13 }: { status: string; size?: number }) {
  if (status === 'running') return <Loader2 size={size} className="spin" aria-hidden />;
  if (status === 'succeeded') return <CheckCircle2 size={size} aria-hidden />;
  if (status === 'interrupted') return <Pause size={size} aria-hidden />;
  if (status === 'rejected') return <CircleSlash size={size} aria-hidden />;
  return <XCircle size={size} aria-hidden />;
}

export function ToolStep({ tool }: { tool: Tool }) {
  const args = argSummary(tool.args);
  const note = tool.status !== 'succeeded' && tool.status !== 'running' ? `${tool.status}${tool.message ? `: ${tool.message}` : ''}` : tool.message;
  return (
    <li className={`tool tone-${toolTone(tool.status)}`} data-status={tool.status}>
      <span className="tool-icon"><ToolIcon status={tool.status} /></span>
      <div className="tool-body">
        <span className="tool-call"><code className={tool.status === 'running' ? 'shimmer' : ''}>{tool.name}</code>{args && <span className="tool-args">{args}</span>}</span>
        {note && <span className="tool-note">{note}</span>}
      </div>
    </li>
  );
}

export function AgentTurn({ task, thread, stopping, modelLabel, modelTitle, compact = false }: { task: Task; thread?: Thread; stopping: boolean; modelLabel: string | null; modelTitle?: string | null; compact?: boolean }) {
  // A live task state beats a stale "suspended" end marker left in the thread by an earlier pause.
  const state = RUNNABLE.includes(task.state) ? task.state : (thread?.end?.state ?? task.state);
  const done = TERMINAL.includes(state);
  const reason = text(thread?.end?.reason ?? task.result);
  const tools = thread?.tools ?? [];
  const shownTools = compact ? tools.slice(-3) : tools;
  return (
    <div className={`msg-agent state-${state}`}>
      <div className="agent-head">
        <StrandsGlyph size={14} />
        <span className="agent-name">DREAM</span>
        {modelLabel && <span className="agent-model" title={`Ran with ${modelTitle ?? modelLabel}`}>{modelLabel}</span>}
        <StateChip state={state} />
      </div>
      {stopping && !done && <p className="note tone-info"><Loader2 size={13} className="spin" aria-hidden /> Waiting for next safe point</p>}
      {compact && tools.length > shownTools.length && <p className="steps-more">{tools.length - shownTools.length} earlier steps</p>}
      {shownTools.length > 0 && <ol className="steps">{shownTools.map((t) => <ToolStep key={t.id} tool={t} />)}</ol>}
      {task.state === 'queued' && <p className="agent-line muted">Waiting for the robot to be free.</p>}
      {state === 'running' && !stopping && !tools.length && (
        <div className="agent-line agent-thought">
          <ThoughtLine label="Planning the first action…" fontSize={13.5} color="var(--text-secondary)" glyphColor="var(--accent)" showTimer />
        </div>
      )}
      {done && (
        <p className={`outcome tone-${state === 'succeeded' ? 'good' : state === 'cancelled' ? 'warn' : 'bad'}`}>
          <b>{endTitle(state)}.</b> {reason}
          {state === 'succeeded' && <small> This is the runtime&apos;s own report; no independent check is shown here.</small>}
          {state === 'cancelled' && <small> The task was stopped and was not completed.</small>}
        </p>
      )}
      {state === 'suspended' && <p className={`note ${task.resume === 'ask' ? 'tone-muted' : 'tone-warn'}`}><Pause size={13} aria-hidden /> {suspendCopy(task)}</p>}
    </div>
  );
}

export function Exchange({ task, thread, stopping, choices, compact }: { task: Task; thread?: Thread; stopping: boolean; choices: ProviderEntry[]; compact?: boolean }) {
  return (
    <div className="exchange" data-task-id={task.id}>
      <div className="msg-user"><p>{task.text}</p></div>
      <AgentTurn task={task} thread={thread} stopping={stopping} modelLabel={shortModelLabel(choices, task.meta?.provider, task.meta?.model)} modelTitle={describeMeta(choices, task.meta)} compact={compact} />
    </div>
  );
}

/** The whole conversation, oldest first, pinned to the newest activity. */
export function ThreadView({ tasks, threads, stopping, choices, running, empty }: {
  tasks: Task[]; threads: Record<string, Thread>; stopping: Set<string>; choices: ProviderEntry[]; running: string | null; empty?: React.ReactNode;
}) {
  const scroller = useRef<HTMLDivElement>(null);
  const pinned = useRef(true);
  const lastTools = tasks.map((t) => threads[t.id]?.tools.length ?? 0).join(',');
  useEffect(() => {
    const el = scroller.current;
    if (el && pinned.current) el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
  }, [tasks.length, lastTools, running]);
  return (
    <div
      className="thread scroll" ref={scroller} aria-live="polite"
      onScroll={(e) => { const el = e.currentTarget; pinned.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80; }}
    >
      <div className="thread-inner">
        {tasks.length === 0 && empty}
        <AnimatePresence initial={false}>
          {tasks.map((t) => (
            <motion.div key={t.id} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}>
              <Exchange task={t} thread={threads[t.id]} stopping={stopping.has(t.id)} choices={choices} />
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </div>
  );
}
