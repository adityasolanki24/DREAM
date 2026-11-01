"use client";
import { useEffect, useState } from 'react';
import { FrameOrder, MultipartJpeg } from './camera-client.mjs';

type CameraHookState = {
  url: string; status: string; fps: number; dropped: number;
  ageMs: number; transportMs: number; decodeMs: number; encodeMs: number;
  /** 'push' (MJPEG stream, one long-lived connection) or 'poll' (fallback, one request per frame). */
  mode: 'push' | 'poll' | '';
};

const FROZEN_AFTER_MS = 1500;

/**
 * The robot head camera as binary JPEG. Prefers the backend's push stream (/camera/stream): frames
 * arrive as they are captured, with no per-frame round trip, which is what makes a tunnelled or
 * tailnet link near real time. Falls back to conditional polling (/camera/frame?after=) when the
 * stream is refused (for example the two-client limit) or drops, and retries the stream later.
 */
export function useCamera(token: string, enabled = true) {
  const [state, setState] = useState<CameraHookState>({ url: '', status: 'waiting', fps: 0, dropped: 0,
    ageMs: 0, transportMs: 0, decodeMs: 0, encodeMs: 0, mode: '' });
  useEffect(() => {
    if (!token || !enabled) return;
    const order = new FrameOrder();
    let stopped = false, after = '', url = '', lastFrame = performance.now();
    let rendered: number[] = [], nextReport = 0, dropped = 0, encodeMs = 0, rttMs = 0;
    const abort = new AbortController();
    const headers = { Authorization: `Bearer ${token}` };
    const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

    const show = async (blob: Blob, h: (name: string) => string | null, mode: 'push' | 'poll', transportMs: number) => {
      if (!order.accept(h('X-Stream-Id'), Number(h('X-Frame-Sequence')))) return;
      const candidate = URL.createObjectURL(blob);
      const image = new Image(); image.src = candidate;
      const decoding = performance.now();
      try { await image.decode(); } catch { URL.revokeObjectURL(candidate); return; }
      const decodeMs = performance.now() - decoding;
      if (stopped) { URL.revokeObjectURL(candidate); return; }
      const previous = url; url = candidate;
      after = h('X-Frame-Id') || after;
      lastFrame = performance.now(); rendered = rendered.filter((t) => lastFrame - t < 1000); rendered.push(lastFrame);
      // Server residence + half the measured round trip + decode: an estimate, not synchronized glass-to-glass.
      const residence = Math.max(0, (Number(h('X-Server-Unix-S')) - Number(h('X-Captured-Unix-S'))) * 1000) || 0;
      const ageMs = residence + (mode === 'push' ? rttMs / 2 : transportMs / 2) + decodeMs;
      setState({ url, status: 'live', fps: rendered.length, dropped, ageMs, transportMs, decodeMs, encodeMs, mode });
      requestAnimationFrame(() => { if (previous) URL.revokeObjectURL(previous); });
      if (lastFrame > nextReport) {
        nextReport = lastFrame + 1000;
        void fetch('/camera/client-metrics', { method: 'POST', headers: { ...headers, 'Content-Type': 'application/json' }, signal: abort.signal,
          body: JSON.stringify({ frame_id: after, transport_ms: transportMs, decode_ms: decodeMs, estimated_age_ms: ageMs, fps: rendered.length, dropped }) }).catch(() => {});
      }
    };

    // Metadata (drops, encode time, round trip) once a second, independent of frame delivery.
    const meta = async () => {
      while (!stopped) {
        const start = performance.now();
        try {
          const r = await fetch('/camera/meta', { headers, signal: AbortSignal.any([abort.signal, AbortSignal.timeout(3000)]), cache: 'no-store' });
          if (r.ok) { const d = await r.json(); dropped = d.dropped_frames; encodeMs = d.encode_ms || 0; rttMs = performance.now() - start; }
        } catch { /* reported through the frame loop */ }
        // A live connection with no new frame means the simulation is not stepping (e.g. model inference), not a network fault.
        if (!stopped && url && performance.now() - lastFrame > FROZEN_AFTER_MS) {
          setState((old) => ({ ...old, status: 'simulation frozen / waiting for sensing', fps: 0 }));
        }
        await sleep(1000);
      }
    };

    const push = async (): Promise<boolean> => {
      const r = await fetch('/camera/stream', { headers, signal: abort.signal, cache: 'no-store' });
      if (!r.ok || !r.body) return false;
      const reader = r.body.getReader();
      const parser = new MultipartJpeg();
      let got = false;
      for (;;) {
        const started = performance.now();
        const { value, done } = await reader.read();
        if (done || stopped) return got;
        for (const part of parser.push(value)) {
          got = true;
          const h = (n: string) => part.headers[n.toLowerCase()] ?? null;
          await show(new Blob([part.body as BlobPart], { type: 'image/jpeg' }), h, 'push', performance.now() - started);
        }
      }
    };

    const poll = async (forMs: number) => {
      const until = performance.now() + forMs;
      while (!stopped && performance.now() < until) {
        const tick = performance.now();
        try {
          const start = performance.now();
          const r = await fetch(`/camera/frame?after=${encodeURIComponent(after)}`, {
            headers, signal: AbortSignal.any([abort.signal, AbortSignal.timeout(3000)]), cache: 'no-store' });
          if (r.status !== 204) {
            if (!r.ok) throw new Error('media disconnected');
            const blob = await r.blob();
            await show(blob, (n) => r.headers.get(n), 'poll', performance.now() - start);
          }
        } catch { if (!stopped) setState((old) => ({ ...old, status: 'video disconnected' })); }
        await sleep(Math.max(0, 100 - (performance.now() - tick)));
      }
    };

    const loop = async () => {
      while (!stopped) {
        let streamed = false;
        try { streamed = await push(); } catch { /* refused, dropped or aborted */ }
        if (stopped) break;
        // Stream refused or ended: keep the picture moving by polling, then try the stream again.
        if (!streamed) setState((old) => ({ ...old, mode: 'poll' }));
        await poll(streamed ? 2000 : 10000);
      }
    };
    void meta();
    void loop();
    return () => { stopped = true; abort.abort(); if (url) URL.revokeObjectURL(url); };
  }, [token, enabled]);
  return state;
}
