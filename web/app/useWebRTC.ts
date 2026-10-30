"use client";
import { useEffect, useRef, useState } from 'react';

/** Authenticated WHEP signalling; ICE/TCP is a private-IAP fallback, not native UDP. */
export function useWebRTC(token: string) {
  const video = useRef<HTMLVideoElement>(null);
  const [state, setState] = useState({ live: false, fps: 0, dropped: 0, jitterMs: 0 });
  useEffect(() => {
    if (!token) return;
    let stopped = false, pc: RTCPeerConnection | undefined, location = '', timer: ReturnType<typeof setInterval>;
    let retry: ReturnType<typeof setTimeout>;
    const abort = new AbortController();
    const headers = { Authorization: `Bearer ${token}` };
    const run = async () => {
      try {
        const status = await fetch('/media/status', { headers, signal: abort.signal }).then(r => r.json());
        if (stopped) return;
        if (!status.available) { retry = setTimeout(() => void run(), 3000); return; }
        pc = new RTCPeerConnection({ iceServers: [] });
        pc.addTransceiver('video', { direction: 'recvonly' });
        pc.ontrack = e => { if (video.current) { video.current.srcObject = new MediaStream([e.track]); void video.current.play().catch(() => {}); } };
        pc.onconnectionstatechange = () => { if (!stopped) setState(s => ({ ...s, live: pc?.connectionState === 'connected' })); if (!stopped && pc?.connectionState === 'failed') { pc.close(); clearInterval(timer); retry = setTimeout(() => void run(), 3000); } };
        await pc.setLocalDescription(await pc.createOffer());
        await new Promise<void>(resolve => {
          if (pc?.iceGatheringState === 'complete') return resolve();
          const timeout = setTimeout(resolve, 3000);
          pc!.onicegatheringstatechange = () => { if (pc?.iceGatheringState === 'complete') { clearTimeout(timeout); resolve(); } };
        });
        if (stopped) return;
        const response = await fetch('/media/whep', { method: 'POST',
          headers: { ...headers, 'Content-Type': 'application/sdp' }, body: pc.localDescription?.sdp,
          signal: AbortSignal.any([abort.signal, AbortSignal.timeout(10000)]) });
        if (!response.ok) throw new Error('WHEP unavailable');
        location = response.headers.get('Location') || '';
        await pc.setRemoteDescription({ type: 'answer', sdp: await response.text() });
        let previousFrames = 0, previousTime = performance.now();
        timer = setInterval(async () => {
          if (!pc || stopped) return;
          const stats = await pc.getStats();
          stats.forEach(report => {
            if (report.type !== 'inbound-rtp' || report.kind !== 'video') return;
            const now = performance.now(), frames = report.framesDecoded || 0;
            const sample = { live: pc?.connectionState === 'connected', fps: Math.round((frames-previousFrames)*1000/(now-previousTime)),
              dropped: report.framesDropped || 0, jitterMs: report.jitterBufferEmittedCount ? report.jitterBufferDelay*1000/report.jitterBufferEmittedCount : 0 };
            setState(sample); previousFrames = frames; previousTime = now;
          });
        }, 1000);
      } catch { if (!stopped) { pc?.close(); setState(s => ({ ...s, live: false })); retry = setTimeout(() => void run(), 3000); } }
    };
    void run();
    return () => { stopped = true; abort.abort(); clearInterval(timer); clearTimeout(retry); pc?.close();
      if (location.startsWith('/media/whep/')) void fetch(location, { method: 'DELETE', headers }).catch(() => {}); };
  }, [token]);
  return { ...state, video };
}
