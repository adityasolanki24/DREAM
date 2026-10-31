'use client';
import { ORB_MODE, type PresenceState } from '../lib/presence';

/**
 * DREAM's presence. Each visual family maps to a real state from presenceFor(): resting, listening, thinking,
 * acting (navigating or manipulating), observing, warning (recovering or failed), safety stop, done and offline.
 * Pure CSS/SVG animation, compositor-only transforms, paused under prefers-reduced-motion; no WebGL, so it never
 * competes with camera decoding.
 */
export function Orb({ state, size = 200, className = '' }: { state: PresenceState; size?: number; className?: string }) {
  const mode = ORB_MODE[state];
  return (
    <div className={`orb ${className}`} data-mode={mode} style={{ ['--s' as string]: `${size}px` }} aria-hidden>
      <div className="orb-halo" />
      <svg className="orb-rings" viewBox="0 0 200 200">
        <circle className="ring ring-base" cx="100" cy="100" r="97" />
        <circle className="ring ring-arc arc-a" cx="100" cy="100" r="97" pathLength={100} />
        <circle className="ring ring-arc arc-b" cx="100" cy="100" r="89" pathLength={100} />
        <g className="orbiters">
          <circle cx="197" cy="100" r="2.6" />
          <circle cx="51.5" cy="184" r="1.8" />
          <circle cx="51.5" cy="16" r="1.4" />
        </g>
        <circle className="ripple ripple-1" cx="100" cy="100" r="66" />
        <circle className="ripple ripple-2" cx="100" cy="100" r="66" />
      </svg>
      <div className="orb-core">
        <div className="orb-scan" />
        <div className="orb-iris" />
        <div className="orb-sheen" />
      </div>
    </div>
  );
}

/** The 16px mark used beside DREAM's messages and in the wordmark. */
export function OrbGlyph({ size = 14 }: { size?: number }) {
  return <span className="orb-glyph" style={{ width: size, height: size }} aria-hidden />;
}
