/**
 * Fold AG-UI events (translated from the scheduler's real events by the backend) into
 * per-task threads. Pure and immutable so it can be tested without a browser.
 */
export const emptyFeed = () => ({ threads: {}, calls: {}, runs: {} });

const blankThread = (id) => ({ id, text: null, state: 'queued', tools: [], end: null, claim: null });

function parse(json, fallback) {
  try {
    return json ? JSON.parse(json) : fallback;
  } catch {
    return fallback;
  }
}

/** Returns {feed, notices}; notices are short, user-facing transitions worth announcing. */
export function applyEvents(previous, events) {
  const feed = { threads: { ...previous.threads }, calls: { ...previous.calls }, runs: { ...previous.runs } };
  const notices = [];
  const edit = (id, change) => {
    const thread = { ...(feed.threads[id] ?? blankThread(id)) };
    change(thread);
    feed.threads[id] = thread;
    return thread;
  };
  for (const e of events) {
    switch (e.type) {
      case 'RUN_STARTED':
        feed.runs[e.runId] = e.threadId;
        edit(e.threadId, (t) => {
          t.state = 'running';
          t.end = null;
        });
        notices.push({ tone: 'info', title: 'Task started', thread: e.threadId });
        break;
      case 'TOOL_CALL_START': {
        const thread = e.toolCallId.replace(/-\d+$/, '');
        feed.calls[e.toolCallId] = thread;
        edit(thread, (t) => {
          t.tools = [...t.tools, { id: e.toolCallId, name: e.toolCallName, args: {}, status: 'running', message: '' }];
        });
        break;
      }
      case 'TOOL_CALL_ARGS': {
        const thread = feed.calls[e.toolCallId];
        if (thread)
          edit(thread, (t) => {
            t.tools = t.tools.map((x) => (x.id === e.toolCallId ? { ...x, args: parse(e.delta, {}) } : x));
          });
        break;
      }
      case 'TOOL_CALL_RESULT': {
        const thread = feed.calls[e.toolCallId];
        const result = parse(typeof e.content === 'string' ? e.content : '', {});
        if (thread) {
          edit(thread, (t) => {
            t.tools = t.tools.map((x) =>
              x.id === e.toolCallId ? { ...x, status: result.status ?? 'unknown', message: result.message ?? '' } : x,
            );
          });
          const name = feed.threads[thread].tools.find((x) => x.id === e.toolCallId)?.name;
          if (result.status && !['succeeded', 'interrupted'].includes(result.status))
            notices.push({ tone: 'warn', title: `${name} ${result.status}`, body: result.message, thread });
        }
        break;
      }
      case 'RUN_FINISHED': {
        const { state, reason } = e.result ?? { state: 'ended', reason: '' };
        delete feed.runs[e.runId];
        edit(e.threadId, (t) => {
          t.state = state;
          t.end = { state, reason };
        });
        notices.push({ tone: state === 'succeeded' ? 'good' : 'warn', title: endTitle(state), body: reason, thread: e.threadId });
        break;
      }
      case 'RUN_ERROR': {
        const thread = feed.runs[e.runId];
        delete feed.runs[e.runId];
        if (thread) {
          edit(thread, (t) => {
            t.state = 'failed';
            t.end = { state: 'failed', reason: e.message };
          });
          notices.push({ tone: 'bad', title: 'Task failed', body: e.message, thread });
        }
        break;
      }
      case 'CUSTOM': {
        const v = e.value ?? {};
        if (!v.id) break;
        if (v.kind === 'accepted') edit(v.id, (t) => void (t.text = v.text));
        else if (v.kind === 'finish') edit(v.id, (t) => void (t.claim = { success: !!v.success, reason: v.reason }));
        else if (v.kind === 'cancelled' && !feed.threads[v.id]?.end)
          edit(v.id, (t) => {
            t.state = 'cancelled';
            t.end = { state: 'cancelled', reason: v.reason };
          });
        break;
      }
      default:
    }
  }
  return { feed, notices };
}

export function endTitle(state) {
  return (
    {
      succeeded: 'Runtime reports success',
      failed: 'Task failed',
      cancelled: 'Task cancelled',
      suspended: 'Task paused',
    }[state] ?? 'Task ended'
  );
}

/** Short, human summary of tool arguments. */
export function argSummary(args) {
  const entries = Object.entries(args ?? {});
  if (!entries.length) return '';
  const show = (v) => (typeof v === 'number' ? String(Number(v.toFixed(2))) : typeof v === 'object' ? JSON.stringify(v) : String(v));
  return entries.map(([k, v]) => `${k} ${show(v)}`).join(' · ');
}
