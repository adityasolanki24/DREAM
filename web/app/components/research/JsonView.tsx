'use client';
import { Check, Copy } from 'lucide-react';
import { useState, type ReactNode } from 'react';

/** Pretty-printed JSON with restrained token colouring and a copy button. */
export function JsonView({ value, label }: { value: unknown; label?: string }) {
  const [copied, setCopied] = useState(false);
  const json = JSON.stringify(value, null, 2) ?? 'undefined';
  return (
    <div className="json">
      <div className="json-bar">
        {label && <span className="json-label">{label}</span>}
        <button className="icon-btn sm" aria-label="Copy JSON" onClick={async () => {
          try { await navigator.clipboard.writeText(json); setCopied(true); setTimeout(() => setCopied(false), 1200); } catch { /* clipboard unavailable */ }
        }}>{copied ? <Check size={13} /> : <Copy size={13} />}</button>
      </div>
      <pre className="scroll">{highlight(json)}</pre>
    </div>
  );
}

function highlight(json: string): ReactNode[] {
  const out: ReactNode[] = [];
  const re = /("(?:\\.|[^"\\])*")(\s*:)?|\b(true|false|null)\b|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let k = 0;
  while ((m = re.exec(json))) {
    if (m.index > last) out.push(json.slice(last, m.index));
    if (m[1]) out.push(<span key={k++} className={m[2] ? 'j-key' : 'j-str'}>{m[1]}</span>, m[2] ?? '');
    else if (m[3]) out.push(<span key={k++} className="j-lit">{m[3]}</span>);
    else out.push(<span key={k++} className="j-num">{m[4]}</span>);
    last = re.lastIndex;
  }
  out.push(json.slice(last));
  return out;
}
