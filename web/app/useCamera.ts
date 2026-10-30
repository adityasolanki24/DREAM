"use client";
import { useEffect, useState } from 'react';
import { FrameOrder } from './camera-client.mjs';

export function useCamera(token: string, enabled = true) {
  const [state, setState] = useState({ url: '', status: 'waiting', fps: 0, dropped: 0,
    ageMs: 0, transportMs: 0, decodeMs: 0, encodeMs: 0 });
  useEffect(() => {
    if (!token || !enabled) return;
    const order = new FrameOrder();
    let stopped = false, after = '', url = '', lastFrame = performance.now();
    let ageAtLastFrame = 0;
    let rendered: number[] = [], nextMeta = 0, nextReport = 0, dropped = 0, encodeMs = 0;
    const abort = new AbortController();
    const headers = { Authorization: `Bearer ${token}` };
    const loop = async () => {
      while (!stopped) {
        const tick = performance.now();
        try {
          if (tick >= nextMeta) {
            const meta = await fetch('/camera/meta', { headers, signal: AbortSignal.any([abort.signal, AbortSignal.timeout(3000)]), cache: 'no-store' });
            if (!meta.ok) throw new Error('media unavailable');
            const data = await meta.json(); dropped = data.dropped_frames; encodeMs = data.encode_ms || 0;
            nextMeta = tick + 1000;
          }
          const start = performance.now();
          const r = await fetch(`/camera/frame?after=${encodeURIComponent(after)}`, {
            headers, signal: AbortSignal.any([abort.signal, AbortSignal.timeout(3000)]), cache: 'no-store' });
          if (r.status === 204) {
            if (!stopped) setState(old => ({ ...old, status: performance.now()-lastFrame > 1500 ? 'simulation frozen / waiting for sensing' : 'live', fps: performance.now()-lastFrame > 1500 ? 0 : old.fps, ageMs: ageAtLastFrame + performance.now()-lastFrame, dropped }));
          } else {
            if (!r.ok) throw new Error('media disconnected');
            const blob = await r.blob();
            const transportMs = performance.now()-start;
            const epoch = r.headers.get('X-Stream-Id');
            const sequence = Number(r.headers.get('X-Frame-Sequence'));
            if (order.accept(epoch, sequence)) {
              const candidate = URL.createObjectURL(blob);
              const image = new Image(); image.src = candidate;
              const decoding = performance.now();
              try { await image.decode(); } catch (error) { URL.revokeObjectURL(candidate); throw error; }
              const decodeMs = performance.now()-decoding;
              if (stopped) { URL.revokeObjectURL(candidate); break; }
              const previous = url; url = candidate;
              after = r.headers.get('X-Frame-Id') || '';
              lastFrame = performance.now(); rendered = rendered.filter(t => lastFrame-t < 1000); rendered.push(lastFrame);
              // Server residence + half measured RTT + decode: an estimate, not synchronized glass-to-glass.
              const residence = Math.max(0, (Number(r.headers.get('X-Server-Unix-S'))-Number(r.headers.get('X-Captured-Unix-S')))*1000);
              const ageMs = residence + transportMs/2 + decodeMs; ageAtLastFrame = ageMs;
              setState({ url, status: 'live', fps: rendered.length, dropped, ageMs, transportMs, decodeMs, encodeMs });
              requestAnimationFrame(() => { if (previous) URL.revokeObjectURL(previous); });
              if (lastFrame > nextReport) {
                nextReport = lastFrame+1000;
                void fetch('/camera/client-metrics', { method: 'POST', headers: { ...headers, 'Content-Type': 'application/json' }, signal: abort.signal,
                  body: JSON.stringify({ frame_id: after, transport_ms: transportMs, decode_ms: decodeMs, estimated_age_ms: ageMs, fps: rendered.length, dropped }) }).catch(() => {});
              }
            }
          }
        } catch { if (!stopped) setState(old => ({ ...old, status: 'video disconnected' })); }
        if (!stopped) await new Promise(resolve => setTimeout(resolve, Math.max(0, 100-(performance.now()-tick))));
      }
    };
    void loop();
    return () => { stopped = true; abort.abort(); if (url) URL.revokeObjectURL(url); };
  }, [token, enabled]);
  return state;
}
