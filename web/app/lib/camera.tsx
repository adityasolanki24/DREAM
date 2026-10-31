'use client';
import { createContext, useContext, useEffect, useMemo, useRef, type ReactNode } from 'react';
import { useCamera } from '../useCamera';
import { useWebRTC } from '../useWebRTC';
import { useSession } from './session';

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
  jpeg: { status: string; fps: number; dropped: number; ageMs: number; transportMs: number; decodeMs: number; encodeMs: number; active: boolean };
  /** Development fixtures only: a labelled test pattern, never presented as a robot frame. */
  testPattern: boolean;
};

const CameraContext = createContext<CameraState | null>(null);

export function useCameraState(): CameraState {
  const c = useContext(CameraContext);
  if (!c) throw new Error('useCameraState outside CameraProvider');
  return c;
}

export function CameraProvider({ children }: { children: ReactNode }) {
  const { token, snapshot, link, fixture } = useSession();
  const rtc = useWebRTC(token ?? '');
  const media = useCamera(token ?? '', !rtc.live);
  const value = useMemo<CameraState>(() => {
    const snapCamera = snapshot?.camera?.data_url ? snapshot.camera : null;
    const source: FrameSource = rtc.live ? 'webrtc' : media.url ? 'jpeg' : snapCamera ? 'snapshot' : 'none';
    const ageS = source === 'jpeg' ? media.ageMs / 1000 : source === 'snapshot' && snapshot && snapCamera ? Math.max(0, snapshot.robot_time_s - snapCamera.time_s) : null;
    const frozen = source === 'webrtc' ? rtc.fps === 0 : source === 'jpeg' ? media.status.startsWith('simulation frozen') : false;
    const transportLabel = source === 'webrtc' ? 'WebRTC · ICE/TCP' : source === 'jpeg' ? 'JPEG stream' : source === 'snapshot' ? 'Snapshot frame' : 'No video';
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
  }, [rtc.live, rtc.stream, rtc.fps, rtc.dropped, rtc.jitterMs, media, snapshot, link, fixture]);
  return <CameraContext.Provider value={value}>{children}</CameraContext.Provider>;
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
