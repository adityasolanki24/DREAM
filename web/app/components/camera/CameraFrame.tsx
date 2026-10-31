'use client';
import { WifiOff } from 'lucide-react';
import { StreamVideo, useCameraState } from '../../lib/camera';

export const STALE_AFTER_S = 8;

export function ageText(ageS: number | null): string {
  if (ageS == null) return '—';
  return ageS < 10 ? `${ageS.toFixed(1)} s` : ageS < 120 ? `${Math.round(ageS)} s` : `${Math.round(ageS / 60)} min`;
}

/**
 * The robot's head camera, filling its container (object-fit: contain). Renders the shared WebRTC stream or the
 * newest JPEG/snapshot frame; returns null when there is no frame so the caller can design its own empty state.
 */
export function CameraFrame({ className = '', badges = true }: { className?: string; badges?: boolean }) {
  const cam = useCameraState();
  if (cam.source === 'none') return null;
  const stale = cam.source !== 'webrtc' && cam.ageS != null && cam.ageS > STALE_AFTER_S;
  return (
    <div className={`camera-frame ${className} ${cam.disconnected ? 'is-disconnected' : ''} ${stale ? 'is-stale' : ''}`}>
      {cam.source === 'webrtc' && cam.stream ? (
        <StreamVideo stream={cam.stream} label="Robot head camera live video" />
      ) : (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={cam.url ?? ''} alt={cam.testPattern ? 'Interface preview test pattern (not a camera frame)' : 'Robot head camera frame'} draggable={false} />
      )}
      {badges && cam.disconnected && <span className="frame-flag tone-warn"><WifiOff size={12} aria-hidden /> Disconnected · last frame received</span>}
      {/* "paused or not sensing" is stated by each surface's own source label, so it is not repeated over the frame */}
      {badges && !cam.disconnected && !cam.frozen && stale && <span className="frame-flag tone-warn">Last observed {ageText(cam.ageS)} ago</span>}
    </div>
  );
}
