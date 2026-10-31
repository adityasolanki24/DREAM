'use client';
import { Background, Handle, Position, ReactFlow, type Edge, type Node, type NodeProps } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { useMemo } from 'react';
import type { Thread } from '../../feed.mjs';
import { toolTone } from '../../lib/format';
import type { Snapshot, Task } from '../../types';
import { ToolIcon } from '../chat/Conversation';
import { Quiet } from '../ui';

type StepData = { label: string; status: string; current: boolean; index: number };

function StepNode({ data }: NodeProps<Node<StepData>>) {
  const failed = !['succeeded', 'running', 'interrupted'].includes(data.status);
  return (
    <div className={`step tone-${toolTone(data.status)} ${data.current ? 'current' : ''}`}>
      <Handle type="target" position={Position.Top} isConnectable={false} />
      <span className="step-index mono">{data.index + 1}</span>
      <ToolIcon status={data.status} />
      <span className="step-label">{data.label}</span>
      {failed && <span className="step-fail">{data.status}</span>}
      <Handle type="source" position={Position.Bottom} isConnectable={false} />
    </div>
  );
}
const nodeTypes = { step: StepNode };

/** The snapshot's observed tool graph for the task it belongs to. */
export function TaskGraph({ snapshot, task, thread }: { snapshot: Snapshot | null; task?: Task; thread?: Thread }) {
  const graph = snapshot?.task_graph;
  const forTask = graph && task && graph.task_id === task.id ? graph : null;
  const lastIndex = (forTask?.nodes.length ?? 0) - 1;
  const nodes: Node[] = useMemo(
    () => (forTask?.nodes ?? []).map((n, i) => ({
      id: n.id, type: 'step', position: { x: 0, y: i * 66 }, draggable: false,
      data: { label: n.label, status: thread?.tools[i]?.status ?? 'running', current: i === lastIndex && task?.state === 'running', index: i },
    })),
    [forTask, thread, lastIndex, task?.state],
  );
  const edges: Edge[] = useMemo(
    () => (forTask?.edges ?? []).map((e, i) => ({ id: `e${i}`, source: e.source, target: e.target, style: { stroke: 'var(--graph-edge)', strokeWidth: 1.2 } })),
    [forTask],
  );
  if (!forTask) return <Quiet title="No graph for this task">The backend publishes the observed tool graph for the task it is currently executing or most recently ran.</Quiet>;
  return (
    <div className="flow" key={nodes.length}>
      <ReactFlow nodes={nodes} edges={edges} nodeTypes={nodeTypes} fitView fitViewOptions={{ padding: 0.25, maxZoom: 1.3 }}
        nodesDraggable={false} nodesConnectable={false} elementsSelectable={false} zoomOnScroll={false} proOptions={{ hideAttribution: true }}>
        <Background gap={18} size={1.2} color="var(--dot-grid)" />
      </ReactFlow>
    </div>
  );
}
