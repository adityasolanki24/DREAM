'use client';
import { createContext, useContext, useEffect, useMemo, useRef, useState, useSyncExternalStore, type ReactNode } from 'react';
import { createCameraMediaStore, initialCameraMedia, useCamera, type CameraHookState, type CameraMediaStore } from '../useCamera';
import { useWebRTC } from '../useWebRTC';
import { useSession } from './session';
import { elapsedAgeS, STALL_GRACE_MS } from '../camera-view.mjs';

export type FrameSource = 'webrtc' | 'jpeg' | 'snapshot' | 'none';

/**
 * One camera connection for the whole app. It lives above the mode switch, so changing mode, panel or layout never
 * renegotiates WebRTC or restarts JPEG polling; every surface reads the same stream or frame URL.
 */
export type CameraState = {
  source: FrameSource;
  stream: MediaStream | null;
  url: string | null;
  /** Seconds since the shown frame was captured (an estimate for JPEG; robot-clock difference for snapshot frames). */
  ageS: number | null;
  fps: number;
  dropped: number;
  /** The stream is connected but no new frames arrive (simulation paused or not sensing). */
  frozen: boolean;
  /** The session link is down; whatever is shown is the last frame received. */
  disconnected: boolean;
  transportLabel: string;
  statusLabel: string;
  rtc: { live: boolean; fps: number; dropped: number; jitterMs: number };
  jpeg: { status: string; fps: number; dropped: number; ageMs: number; transportMs: number; decodeMs: number; encodeMs: number; active: boolean; at: number };
  /** Development fixtures only: a labelled test pattern, never presented as a robot frame. */
  testPattern: boolean;
};

/**
 * The fields that change when the connection changes, not with every frame. Most surfaces read only this, so a
 * frame arriving does not re-render them; the per-frame fields (frame URL, fps, age, timings) reach only the
 * components that display them, through the frame store below.
 */
export type CameraSummary = {
  source: FrameSource;
  stream: MediaStream | null;
  frozen: boolean;
  disconnected: boolean;
  transportLabel: string;
  testPattern: boolean;
  rtcLive: boolean;
  jpegActive: boolean;
  jpegStatus: string;
};

type Inputs = {
  rtc: { live: boolean; stream: MediaStream | null; fps: number; dropped: number; jitterMs: number };
  snapshot: ReturnType<typeof useSession>['snapshot'];
  link: ReturnType<typeof useSession>['link'];
  fixture: ReturnType<typeof useSession>['fixture'];
};
const SummaryContext = createContext<CameraSummary | null>(null);
const InputsContext = createContext<Inputs | null>(null);
const MediaContext = createContext<CameraMediaStore | null>(null);

export function useCameraState(): CameraSummary {
  const c = useContext(SummaryContext);
  if (!c) throw new Error('useCameraState outside CameraProvider');
  return c;
}

function derive(i: Inputs, media: CameraHookState): CameraState {
  const { rtc, snapshot, link, fixture } = i;
  const snapCamera = snapshot?.camera?.data_url ? snapshot.camera : null;
  const source: FrameSource = rtc.live ? 'webrtc' : media.url ? 'jpeg' : snapCamera ? 'snapshot' : 'none';
  const ageS = source === 'jpeg' ? media.ageMs / 1000 : source === 'snapshot' && snapshot && snapCamera ? Math.max(0, snapshot.robot_time_s - snapCamera.time_s) : null;
  const frozen = source === 'webrtc' ? rtc.fps === 0 : source === 'jpeg' ? media.status.startsWith('simulation frozen') : false;
  const transportLabel = source === 'webrtc' ? 'WebRTC · ICE/TCP' : source === 'jpeg' ? (media.mode === 'poll' ? 'JPEG polling' : 'MJPEG push') : source === 'snapshot' ? 'Snapshot frame' : 'No video';
  const statusLabel = source === 'webrtc'
    ? (frozen ? 'Simulation paused or not sensing' : `${rtc.fps} fps`)
    : source === 'jpeg' ? (frozen ? 'Simulation paused or not sensing' : media.status === 'live' ? `${media.fps} fps` : media.status)
      : source === 'snapshot' ? 'Updates when a tool observes' : 'Waiting for frames';
  return {
    source, stream: rtc.live ? rtc.stream : null, url: media.url || snapCamera?.data_url || null, ageS,
    fps: source === 'webrtc' ? rtc.fps : media.fps, dropped: source === 'webrtc' ? rtc.dropped : media.dropped,
    frozen, disconnected: link !== 'online', transportLabel, statusLabel,
    rtc: { live: rtc.live, fps: rtc.fps, dropped: rtc.dropped, jitterMs: rtc.jitterMs },
    jpeg: { ...media, active: Boolean(media.url) },
    testPattern: Boolean(fixture?.testPattern && snapCamera),
  };
}

/** The full per-frame state; re-renders on every frame, so use it only in the component that shows frame data. */
export function useCameraFrame(): CameraState {
  const store = useContext(MediaContext);
  const inputs = useContext(InputsContext);
  if (!store || !inputs) throw new Error('useCameraFrame outside CameraProvider');
  const media = useSyncExternalStore(store.subscribe, store.get, store.get);
  return useMemo(() => derive(inputs, media), [inputs, media]);
}

/** Age of the shown frame in seconds: the capture-time estimate, plus real elapsed time while frames have stopped. */
export function useCameraAge(): number | null {
  const cam = useCameraFrame();
  const [, tick] = useState(0);
  const at = cam.source === 'jpeg' ? cam.jpeg.at : null;
  useEffect(() => {
    if (at == null) return;
    const id = setInterval(() => { if (performance.now() - at > STALL_GRACE_MS) tick((n) => n + 1); }, 500);
    return () => clearInterval(id);
  }, [at]);
  return elapsedAgeS(cam.ageS, at, performance.now());
}

export function CameraProvider({ children }: { children: ReactNode }) {
  const { token, snapshot, link, fixture } = useSession();
  const rtc = useWebRTC(token ?? '');
  const [store] = useState<CameraMediaStore>(createCameraMediaStore);
  useCamera(token ?? '', !rtc.live, store);
  const inputs = useMemo<Inputs>(() => ({ rtc, snapshot, link, fixture }), [rtc, snapshot, link, fixture]);
  // The provider re-renders only when these coarse media facts change, never for each frame.
  const hasUrl = useSyncExternalStore(store.subscribe, () => Boolean(store.get().url), () => false);
  const status = useSyncExternalStore(store.subscribe, () => store.get().status, () => 'waiting');
  const mode = useSyncExternalStore(store.subscribe, () => store.get().mode, () => '');
  const last = useRef<CameraSummary | null>(null);
  const summary = useMemo<CameraSummary>(() => {
    const d = derive(inputs, { ...initialCameraMedia(), url: hasUrl ? 'x' : '', status, mode: mode as CameraHookState['mode'] });
    const next: CameraSummary = { source: d.source, stream: d.stream, frozen: d.frozen, disconnected: d.disconnected, transportLabel: d.transportLabel,
      testPattern: d.testPattern, rtcLive: d.rtc.live, jpegActive: d.jpeg.active, jpegStatus: d.jpeg.status };
    const prev = last.current;
    // Keep the previous object when nothing it carries changed, so surfaces reading it are not disturbed.
    if (prev && (Object.keys(next) as (keyof CameraSummary)[]).every((k) => Object.is(prev[k], next[k]))) return prev;
    last.current = next;
    return next;
  }, [inputs, hasUrl, status, mode]);
  return <MediaContext.Provider value={store}><InputsContext.Provider value={inputs}><SummaryContext.Provider value={summary}>{children}</SummaryContext.Provider></InputsContext.Provider></MediaContext.Provider>;
}

/** A <video> bound to the shared stream; mounting another surface reuses the same MediaStream. */
export function StreamVideo({ stream, className, label }: { stream: MediaStream; className?: string; label: string }) {
  const ref = useRef<HTMLVideoElement>(null);
  useEffect(() => {
    const v = ref.current;
    if (!v) return;
    if (v.srcObject !== stream) v.srcObject = stream;
    void v.play().catch(() => {});
  }, [stream]);
  return <video ref={ref} className={className} autoPlay playsInline muted aria-label={label} />;
}
