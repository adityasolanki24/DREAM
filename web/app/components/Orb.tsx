'use client';
import { useEffect, useRef, useState } from 'react';
import { ORB_MODE, type PresenceState } from '../lib/presence';
import ReactBitsOrb from './orb/Orb';

/**
 * DREAM's presence: the React Bits Orb (vendored unmodified in ./orb/Orb.tsx) driven only through its own props.
 * Each visual family maps to a real state from presenceFor(); pointer gimmicks (follow cursor, click pulse, rotate on
 * hover) are off so the orb only ever reflects what the robot and model are actually doing.
 */
type Look = { colors: [string, string, string]; speed: number; glow: number; wobble: number; highlight: number; trail: number; awake: boolean; opacity: number };

const LOOKS: Record<ReturnType<typeof modeOf>, Look> = {
  rest:    { colors: ['#648AFF', '#5DE4F4', '#0E1430'], speed: 0.45, glow: 1.0, wobble: 0.8, highlight: 0.8, trail: 1.0, awake: false, opacity: 1 },
  listen:  { colors: ['#5DE4F4', '#E6F2FF', '#10203A'], speed: 0.9, glow: 1.5, wobble: 1.3, highlight: 1.2, trail: 1.2, awake: true, opacity: 1 },
  think:   { colors: ['#986BFF', '#648AFF', '#140F3A'], speed: 1.25, glow: 1.2, wobble: 1.0, highlight: 1.1, trail: 1.6, awake: false, opacity: 1 },
  act:     { colors: ['#648AFF', '#986BFF', '#10123A'], speed: 1.8, glow: 1.3, wobble: 1.6, highlight: 1.3, trail: 1.8, awake: true, opacity: 1 },
  observe: { colors: ['#5DE4F4', '#648AFF', '#0B2030'], speed: 0.7, glow: 1.4, wobble: 0.6, highlight: 1.5, trail: 1.0, awake: false, opacity: 1 },
  warn:    { colors: ['#E18BDF', '#DDB26A', '#2A1424'], speed: 0.6, glow: 1.0, wobble: 1.1, highlight: 0.8, trail: 1.0, awake: false, opacity: 1 },
  stop:    { colors: ['#EE8C80', '#E8705F', '#2A0E0E'], speed: 0.15, glow: 1.2, wobble: 0.2, highlight: 0.5, trail: 0.4, awake: false, opacity: 1 },
  done:    { colors: ['#8DC9A3', '#5DE4F4', '#0C2620'], speed: 0.5, glow: 1.3, wobble: 0.8, highlight: 1.0, trail: 1.0, awake: false, opacity: 1 },
  off:     { colors: ['#7E838F', '#4A5058', '#15161A'], speed: 0.06, glow: 1.0, wobble: 0.3, highlight: 0.55, trail: 0.4, awake: false, opacity: 0.8 },
};
function modeOf(state: PresenceState) { return ORB_MODE[state]; }

// Verifying is concentrated rather than restless; waiting and paused slow right down.
const TWEAK: Partial<Record<PresenceState, Partial<Look>>> = {
  verifying: { speed: 0.45, glow: 1.6, wobble: 0.5 },
  paused: { speed: 0.15 },
  waiting: { speed: 0.25 },
};

const hex = (h: string) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
const toHex = (c: number[]) => `#${c.map((v) => Math.round(v).toString(16).padStart(2, '0')).join('')}`;
const mix = (a: number, b: number, t: number) => a + (b - a) * t;
function blend(a: Look, b: Look, t: number): Look {
  return {
    colors: a.colors.map((c, i) => toHex(hex(c).map((v, k) => mix(v, hex(b.colors[i])[k], t)))) as Look['colors'],
    speed: mix(a.speed, b.speed, t), glow: mix(a.glow, b.glow, t), wobble: mix(a.wobble, b.wobble, t),
    highlight: mix(a.highlight, b.highlight, t), trail: mix(a.trail, b.trail, t), awake: t > 0.5 ? b.awake : a.awake, opacity: mix(a.opacity, b.opacity, t),
  };
}
const lookFor = (s: PresenceState): Look => ({ ...LOOKS[modeOf(s)], ...TWEAK[s] });

export function Orb({ state, size = 200, className = '' }: { state: PresenceState; size?: number; className?: string }) {
  const [look, setLook] = useState<Look>(() => lookFor(state));
  const [flash, setFlash] = useState(false);
  const shown = useRef(look);
  const previous = useRef(state);
  const small = size < 48;

  useEffect(() => {
    const from = shown.current;
    const to = lookFor(state);
    const start = performance.now();
    const timer = setInterval(() => {
      const t = Math.min(1, (performance.now() - start) / 600);
      shown.current = blend(from, to, t * t * (3 - 2 * t));
      setLook(shown.current);
      if (t >= 1) clearInterval(timer);
    }, 40);
    // A brief brightening only when a task has genuinely just been reported successful.
    const was = previous.current;
    previous.current = state;
    let off: ReturnType<typeof setTimeout> | undefined;
    if (state === 'completed' && was !== 'completed') { setFlash(true); off = setTimeout(() => setFlash(false), 1400); }
    return () => { clearInterval(timer); if (off) clearTimeout(off); };
  }, [state]);

  return (
    <div className={`orb-host ${className}`} data-state={state} style={{ width: size, height: size }} aria-hidden>
      <ReactBitsOrb
        colors={look.colors}
        backgroundColor="#080a10"
        speed={look.speed}
        glow={look.glow * (small ? 1.35 : 1) + (flash ? 0.8 : 0)}
        wobble={look.wobble}
        highlight={look.highlight * (small ? 1.2 : 1)}
        trail={look.trail}
        forceHoverState={look.awake || flash}
        hoverIntensity={0.8}
        opacity={look.opacity}
        bleed={1}
        followCursor={false}
        rotateOnHover={false}
        clickPulse={false}
        intro
        // Below 48px the ring loses contrast on the dark HUD; a denser backing and higher light transport keep it legible.
        dpr={small ? 2 : undefined}
      />
    </div>
  );
}

/** The 16px static mark beside DREAM's messages and in the wordmark (not animated). */
export function OrbGlyph({ size = 14 }: { size?: number }) {
  return <span className="orb-glyph" style={{ width: size, height: size }} aria-hidden />;
}
