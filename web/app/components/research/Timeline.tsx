'use client';
import { ChevronRight } from 'lucide-react';
import { useState } from 'react';
import { argSummary, type Thread, type Tool } from '../../feed.mjs';
import { toolTone } from '../../lib/format';
import { ToolIcon } from '../chat/Conversation';
import { Quiet } from '../ui';
import { JsonView } from './JsonView';

function Row({ tool, index, open, onToggle }: { tool: Tool; index: number; open: boolean; onToggle: () => void }) {
  const args = argSummary(tool.args);
  return (
    <li className={`tl-row tone-${toolTone(tool.status)} ${open ? 'open' : ''}`} data-status={tool.status}>
      <button className="tl-head" onClick={onToggle} aria-expanded={open}>
        <span className="tl-index mono">{String(index + 1).padStart(2, '0')}</span>
        <span className="tl-icon"><ToolIcon status={tool.status} /></span>
        <span className="tl-call"><code className={`tl-name ${tool.status === 'running' ? 'shimmer' : ''}`}>{tool.name}</code>{args && <span className="tl-args mono">{args}</span>}</span>
        <span className="tl-status">{tool.status}</span>
        <ChevronRight size={14} className="tl-chev" aria-hidden />
      </button>
      {tool.message && !open && <p className="tl-message">{tool.message}</p>}
      {open && (
        <div className="tl-detail">
          <dl className="kv">
            <div><dt>Call id</dt><dd className="mono">{tool.id}</dd></div>
            <div><dt>Status</dt><dd>{tool.status}</dd></div>
            <div className="kv-wide"><dt>Result message</dt><dd>{tool.message || <span className="faint">none</span>}</dd></div>
          </dl>
          <JsonView value={tool.args} label="Arguments" />
        </div>
      )}
    </li>
  );
}

/** Every tool call of a task in order: the observed sequence, with arguments and results inspectable. */
export function Timeline({ thread }: { thread?: Thread }) {
  const [open, setOpen] = useState<Set<string>>(new Set());
  const tools = thread?.tools ?? [];
  if (!tools.length) return <Quiet title="No tool calls yet">Tool calls appear here, in order, as the robot executes them.</Quiet>;
  const toggle = (id: string) => setOpen((o) => { const n = new Set(o); if (n.has(id)) n.delete(id); else n.add(id); return n; });
  return (
    <div className="timeline">
      <div className="tl-toolbar">
        <span className="faint small">{tools.length} calls · observed sequence, not a plan</span>
        <button className="text-btn" onClick={() => setOpen(open.size ? new Set() : new Set(tools.map((t) => t.id)))}>{open.size ? 'Collapse all' : 'Expand all'}</button>
      </div>
      <ol className="tl-list">{tools.map((t, i) => <Row key={t.id} tool={t} index={i} open={open.has(t.id)} onToggle={() => toggle(t.id)} />)}</ol>
    </div>
  );
}
