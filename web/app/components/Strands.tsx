'use client';
import { useEffect, useId, useRef, useState } from 'react';
import { ORB_MODE, type PresenceState } from '../lib/presence';
import ReactBitsStrands from './reactbits/Strands';

/**
 * DREAM's presence: the React Bits Strands (vendored unmodified in ./reactbits/Strands.tsx) driven only
 * through its own props. Each visual family maps to a real state from presenceFor(); pointer gimmicks
 * (follow cursor, click pulse) are off so the strands only ever reflect what the robot and the model
 * are actually doing. Ported from the previous Orb presence map with the same families and palettes.
 */
type Mode = ReturnType<typeof modeOf>;
type Look = { colors: [string, string]; energy: number; speed: number; listening: boolean; opacity: number };

const LOOKS: Record<Mode, Look> = {
  rest:    { colors: ['#648AFF', '#5DE4F4'], energy: 0.25, speed: 0.45, listening: false, opacity: 1 },
  listen:  { colors: ['#5DE4F4', '#E6F2FF'], energy: 0.8,  speed: 0.9,  listening: true,  opacity: 1 },
  think:   { colors: ['#986BFF', '#648AFF'], energy: 0.5,  speed: 1.25, listening: false, opacity: 1 },
  act:     { colors: ['#648AFF', '#986BFF'], energy: 0.95, speed: 1.8,  listening: false, opacity: 1 },
  observe: { colors: ['#5DE4F4', '#648AFF'], energy: 0.35, speed: 0.7,  listening: false, opacity: 1 },
  warn:    { colors: ['#E18BDF', '#DDB26A'], energy: 0.55, speed: 0.6,  listening: false, opacity: 1 },
  stop:    { colors: ['#EE8C80', '#E8705F'], energy: 0.1,  speed: 0.15, listening: false, opacity: 1 },
  done:    { colors: ['#8DC9A3', '#5DE4F4'], energy: 0.4,  speed: 0.5,  listening: false, opacity: 1 },
  off:     { colors: ['#7E838F', '#4A5058'], energy: 0.05, speed: 0.06, listening: false, opacity: 0.85 },
};
function modeOf(state: PresenceState) { return ORB_MODE[state]; }

// Per-state nuance kept from the Orb presence map: verifying is concentrated rather than restless;
// paused and waiting slow right down.
const TWEAK: Partial<Record<PresenceState, { energy?: number; speed?: number }>> = {
  verifying: { energy: 0.3, speed: 0.45 },
  paused: { speed: 0.15 },
  waiting: { speed: 0.25 },
};

const hex = (h: string) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
const toHex = (c: number[]) => `#${c.map((v) => Math.round(v).toString(16).padStart(2, '0')).join('')}`;
const mix = (a: number, b: number, t: number) => a + (b - a) * t;
function blend(a: Look, b: Look, t: number): Look {
  return {
    colors: a.colors.map((c, i) => toHex(hex(c).map((v, k) => mix(v, hex(b.colors[i])[k], t)))) as Look['colors'],
    energy: mix(a.energy, b.energy, t), speed: mix(a.speed, b.speed, t),
    listening: t > 0.5 ? b.listening : a.listening, opacity: mix(a.opacity, b.opacity, t),
  };
}
const lookFor = (s: PresenceState): Look => ({ ...LOOKS[modeOf(s)], ...TWEAK[s] });

/** True while the :root theme is light; the canvas components take lightMode to stay legible there. */
export function useThemeIsLight() {
  const [light, setLight] = useState(false);
  useEffect(() => {
    const el = document.documentElement;
    const sync = () => setLight(el.dataset.theme === 'light');
    sync();
    const mo = new MutationObserver(sync);
    mo.observe(el, { attributes: true, attributeFilter: ['data-theme'] });
    return () => mo.disconnect();
  }, []);
  return light;
}

/** The presence orb: one live Strands canvas per visible presence, not per message. */
export function DreamStrands({ state, size = 200, className = '' }: { state: PresenceState; size?: number; className?: string }) {
  const [look, setLook] = useState<Look>(() => lookFor(state));
  const [flash, setFlash] = useState(false);
  const shown = useRef(look);
  const previous = useRef(state);
  const small = size < 48;
  const light = useThemeIsLight();

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
    <div className={`strands-host ${className}`} data-state={state} style={{ width: size, height: size }} aria-hidden>
      <ReactBitsStrands
        colors={look.colors}
        energy={Math.min(1, look.energy + (flash ? 0.35 : 0))}
        speed={look.speed}
        glass
        listening={look.listening}
        opacity={look.opacity}
        interactive={false}
        clickPulse={false}
        intro
        lightMode={light}
        // Small slots need extra pixel density to stay crisp (the same rule the Orb host used).
        maxDevicePixelRatio={small ? 2 : undefined}
      />
    </div>
  );
}

/** Bundled strands as a flat static mark, for message heads where one canvas per message would be wasteful. */
export function StrandsGlyph({ size = 14, className = '' }: { size?: number; className?: string }) {
  const id = useId();
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className={`strands-glyph ${className}`} aria-hidden>
      <defs>
        <linearGradient id={id} x1="4" y1="21" x2="21" y2="3" gradientUnits="userSpaceOnUse">
          <stop stopColor="#648AFF" />
          <stop offset="1" stopColor="#5DE4F4" />
        </linearGradient>
      </defs>
      <path d="M2.5 19.5C9 18.5 14 13.5 21.5 3.5" stroke={`url(#${id})`} strokeWidth="2.1" strokeLinecap="round" />
      <path d="M5 21C10.5 20 14.5 15.5 17.5 9.5" stroke={`url(#${id})`} strokeWidth="1.3" strokeLinecap="round" opacity="0.65" />
      <path d="M1.5 15C6.5 12.5 10.5 7.5 13.5 1.5" stroke={`url(#${id})`} strokeWidth="1.1" strokeLinecap="round" opacity="0.45" />
    </svg>
  );
}

/** The wordmark's live strip: bare strands, no glass, DREAM's own palette. */
export function StrandsMark({ width = 64, height = 20, className = '' }: { width?: number; height?: number; className?: string }) {
  const light = useThemeIsLight();
  return (
    <div className={`brand-strands ${className}`} style={{ width, height }} aria-hidden>
      <ReactBitsStrands
        glass={false}
        colors={light ? ['#33618F', '#3E9BB8'] : ['#648AFF', '#5DE4F4']}
        energy={0.35}
        speed={0.55}
        interactive={false}
        clickPulse={false}
        intro={false}
        lightMode={light}
      />
    </div>
  );
}
