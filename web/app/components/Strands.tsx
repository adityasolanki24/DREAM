'use client';
import { useEffect, useId, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react';
import { ORB_MODE, type PresenceState } from '../lib/presence';
import { detentsCrossed, dragStep, pointerAngle, SETTLE_V, stepSpin, TAP_RAD } from '../lib/orbSpin.mjs';
import ReactBitsStrands from './reactbits/Strands';

/**
 * DREAM's presence: the React Bits Strands (vendored in ./reactbits/Strands.tsx) driven only through
 * its own props. Each visual family maps to a real state from presenceFor(), so the strands always
 * reflect what the robot and the model are actually doing. Instances marked `interactive` are also
 * graspable: a drag spins the strands inside the glass (inertia, detent haptics and a tap pulse)
 * while the sphere shading stays fixed, as with real glass. Ported from the previous Orb presence
 * map with the same families and palettes.
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

/** Haptics, guarded: desktop browsers ignore it, and iOS Safari ships no API — always best effort. */
const buzz = (pattern: number | number[]) => {
  try {
    if (typeof navigator !== 'undefined' && 'vibrate' in navigator) navigator.vibrate(pattern);
  } catch {
    /* vibration is disallowed or unsupported; feedback stays visual */
  }
};

/**
 * The presence orb: one live Strands canvas per visible presence, not per message. With
 * `interactive` the orb is graspable — drag to spin the strands inside the glass (with inertia,
 * detent ticks and a tap pulse) while the sphere itself stays fixed, as with real glass. The
 * state families keep reflecting the robot either way. Small rails and chips stay passive.
 */
export function DreamStrands({
  state, size = 200, className = '', interactive = false,
}: { state: PresenceState; size?: number; className?: string; interactive?: boolean }) {
  const [look, setLook] = useState<Look>(() => lookFor(state));
  const [flash, setFlash] = useState(false);
  const shown = useRef(look);
  const previous = useRef(state);
  const small = size < 48;
  const light = useThemeIsLight();

  // Drag-to-spin. `spin` holds the truth (roll + angular velocity); the `roll` state only mirrors
  // it into the renderer's prop. Pointers drive it per event, the released coast per rAF frame.
  const hostRef = useRef<HTMLDivElement | null>(null);
  const [roll, setRoll] = useState(0);
  const [dragging, setDragging] = useState(false);
  const spin = useRef({ roll: 0, vel: 0 });
  const gesture = useRef<null | { id: number; pa: number | null; t0: number; total: number; moved: boolean }>(null);
  const coastRaf = useRef(0);
  const engaged = useRef(-1e9);  // last release (performance.now()); gates the completed-flash haptic

  const stopCoast = () => { if (coastRaf.current) { cancelAnimationFrame(coastRaf.current); coastRaf.current = 0; } };
  useEffect(() => () => { if (coastRaf.current) cancelAnimationFrame(coastRaf.current); }, []);

  const centreOf = (e: { clientX: number; clientY: number }) => {
    const el = hostRef.current;
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return { cx: r.left + r.width / 2, cy: r.top + r.height / 2, radius: Math.min(r.width, r.height) / 2 };
  };

  const startCoast = () => {
    stopCoast();
    if (Math.abs(spin.current.vel) <= SETTLE_V) { spin.current.vel = 0; return; }
    let last = performance.now();
    const tick = (now: number) => {
      const dt = Math.min(0.05, Math.max(0.001, (now - last) / 1000));  // a returned tab never teleports the spin
      last = now;
      const s = stepSpin(spin.current.roll, spin.current.vel, dt);
      spin.current.roll = s.angle;
      spin.current.vel = s.vel;
      setRoll(s.angle);
      coastRaf.current = s.settled ? 0 : requestAnimationFrame(tick);
    };
    coastRaf.current = requestAnimationFrame(tick);
  };

  const onDown = (e: ReactPointerEvent<HTMLDivElement>) => {
    stopCoast();
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;  // reduced motion: the orb stays a passive presence
    if (e.pointerType === 'mouse' && e.button !== 0) return;  // respond to the primary button only
    if (gesture.current) return;  // a second pointer never steals the spin
    const c = centreOf(e);
    if (!c) return;
    try { hostRef.current?.setPointerCapture(e.pointerId); } catch { /* capture is an enhancement */ }
    gesture.current = { id: e.pointerId, pa: null, t0: e.timeStamp, total: 0, moved: false };
    spin.current.vel = 0;  // catching the glass stops its momentum
    setDragging(true);
    buzz(5);  // light acknowledgment on first contact
  };

  const onMove = (e: ReactPointerEvent<HTMLDivElement>) => {
    const g = gesture.current;
    if (!g || e.pointerId !== g.id) return;
    const c = centreOf(e);
    if (!c) return;
    const pa = pointerAngle(e.clientX, e.clientY, c.cx, c.cy);
    if (g.pa == null) {
      // Near the core the angle is noise: wait for a usable handle direction before rolling.
      if (Math.hypot(e.clientX - c.cx, e.clientY - c.cy) <= c.radius * 0.25) return;
      g.pa = pa;
      g.t0 = e.timeStamp;
      return;
    }
    const step = dragStep({ angle: spin.current.roll, vel: spin.current.vel }, g.pa, pa, g.t0, e.timeStamp);
    g.pa = pa;
    g.t0 = e.timeStamp;
    g.total += step.d;
    spin.current.roll = step.angle;
    spin.current.vel = step.vel;
    if (!g.moved && Math.abs(g.total) > TAP_RAD) g.moved = true;
    setRoll(step.angle);
    if (detentsCrossed(g.total - step.d, g.total) > 0) buzz(2);  // a tick per quarter turn, like a physical detent
  };

  const endDrag = (e: ReactPointerEvent<HTMLDivElement>, lively: boolean) => {
    const g = gesture.current;
    if (!g || e.pointerId !== g.id) return;
    gesture.current = null;
    setDragging(false);
    engaged.current = performance.now();
    try {
      if (hostRef.current?.hasPointerCapture(e.pointerId)) hostRef.current.releasePointerCapture(e.pointerId);
    } catch { /* the capture may already be gone */ }
    if (!lively) return;  // cancelled (palm rejection, teardown): no inertia, no haptic
    if (!g.moved) { buzz(12); return; }  // a tap answers with one clear pulse
    if (e.timeStamp - g.t0 > 100) spin.current.vel = 0;  // the pointer was parked; a stale flick speed must not fire
    buzz(Math.abs(spin.current.vel) > 3 ? 10 : 6);  // a thrown release answers stronger than a slow one
    startCoast();
  };

  const onUp = (e: ReactPointerEvent<HTMLDivElement>) => endDrag(e, true);
  const onCancel = (e: ReactPointerEvent<HTMLDivElement>) => endDrag(e, false);

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
    if (state === 'completed' && was !== 'completed') {
      setFlash(true);
      off = setTimeout(() => setFlash(false), 1400);
      // While the user is still with the orb (2.5 s), a success gets a three-beat haptic.
      if (interactive && performance.now() - engaged.current < 2500) buzz([12, 60, 18]);
    }
    return () => { clearInterval(timer); if (off) clearTimeout(off); };
  }, [state, interactive]);

  return (
    <div
      ref={hostRef}
      className={`strands-host ${className}${interactive ? ' is-interactive' : ''}${dragging ? ' is-dragging' : ''}`}
      data-state={state}
      style={{ width: size, height: size }}
      aria-hidden
      onPointerDown={interactive ? onDown : undefined}
      onPointerMove={interactive ? onMove : undefined}
      onPointerUp={interactive ? onUp : undefined}
      onPointerCancel={interactive ? onCancel : undefined}
      onLostPointerCapture={interactive ? (e) => endDrag(e, false) : undefined}
    >
      <ReactBitsStrands
        colors={look.colors}
        energy={Math.min(1, look.energy + (flash ? 0.35 : 0))}
        speed={look.speed}
        glass
        listening={look.listening}
        opacity={look.opacity}
        roll={interactive ? roll : 0}
        interactive={interactive}
        clickPulse={interactive}
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
