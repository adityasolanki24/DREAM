'use client';
import { useEffect, useRef, useState } from 'react';
import type { PresenceState } from '../../lib/presence';
import Aurora from './Aurora';

/**
 * DREAM's ambient identity: the React Bits Aurora (vendored unmodified in ./Aurora.tsx) driven only through its own
 * props. Every look is keyed to a real PresenceState (task, agent events, safety stop, link); nothing here invents
 * progress. Parameters ease between states locally, so only this component re-renders and the camera provider above
 * the mode switch is never touched.
 */

export const DREAM_PALETTE = ['#5DE4F4', '#648AFF', '#986BFF', '#E18BDF'];
const WARNING = ['#F2A07A', '#E8705F', '#C2566E', '#E8705F'];
const DIM = ['#4C6A86', '#4F5E8C', '#5C5480', '#6A5876'];
const SOFT_FAIL = ['#8C7BE0', '#B07BCB', '#D99AB5', '#986BFF'];

type Look = { speed: number; amplitude: number; brightness: number; glow: number; height: number; blend: number; stops: string[] };

const LOOKS: Record<PresenceState, Look> = {
  // Resting states stay quieter so the backdrop does not compete with content; active states keep their energy.
  idle:            { speed: 0.35, amplitude: 0.8, brightness: 0.60, glow: 0.45, height: 0.48, blend: 0.6, stops: DREAM_PALETTE },
  waiting:         { speed: 0.25, amplitude: 0.7, brightness: 0.55, glow: 0.42, height: 0.46, blend: 0.6, stops: DREAM_PALETTE },
  paused:          { speed: 0.12, amplitude: 0.6, brightness: 0.52, glow: 0.4, height: 0.44, blend: 0.6, stops: DREAM_PALETTE },
  listening:       { speed: 0.5, amplitude: 0.9, brightness: 0.92, glow: 0.7, height: 0.54, blend: 0.55, stops: DREAM_PALETTE },
  thinking:        { speed: 0.85, amplitude: 1.05, brightness: 0.8, glow: 0.65, height: 0.5, blend: 0.5, stops: DREAM_PALETTE },
  navigating:      { speed: 1.15, amplitude: 1.3, brightness: 0.82, glow: 0.62, height: 0.52, blend: 0.45, stops: DREAM_PALETTE },
  interacting:     { speed: 1.0, amplitude: 1.2, brightness: 0.84, glow: 0.66, height: 0.52, blend: 0.45, stops: DREAM_PALETTE },
  observing:       { speed: 0.5, amplitude: 0.7, brightness: 0.8, glow: 0.7, height: 0.46, blend: 0.4, stops: ['#5DE4F4', '#648AFF', '#5DE4F4', '#986BFF'] },
  verifying:       { speed: 0.4, amplitude: 0.55, brightness: 0.78, glow: 0.72, height: 0.42, blend: 0.35, stops: DREAM_PALETTE },
  completed:       { speed: 0.35, amplitude: 0.8, brightness: 0.66, glow: 0.52, height: 0.48, blend: 0.6, stops: DREAM_PALETTE },
  recovering:      { speed: 0.55, amplitude: 0.9, brightness: 0.66, glow: 0.55, height: 0.46, blend: 0.6, stops: SOFT_FAIL },
  failed:          { speed: 0.3, amplitude: 0.7, brightness: 0.58, glow: 0.5, height: 0.44, blend: 0.65, stops: SOFT_FAIL },
  'safety-stopped': { speed: 0.18, amplitude: 0.5, brightness: 0.72, glow: 0.6, height: 0.46, blend: 0.5, stops: WARNING },
  offline:         { speed: 0.08, amplitude: 0.5, brightness: 0.36, glow: 0.4, height: 0.4, blend: 0.7, stops: DIM },
};

const hex = (h: string) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
const toHex = (c: number[]) => `#${c.map((v) => Math.round(v).toString(16).padStart(2, '0')).join('')}`;
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
function mixLook(a: Look, b: Look, t: number): Look {
  return {
    speed: lerp(a.speed, b.speed, t), amplitude: lerp(a.amplitude, b.amplitude, t), brightness: lerp(a.brightness, b.brightness, t),
    glow: lerp(a.glow, b.glow, t), height: lerp(a.height, b.height, t), blend: lerp(a.blend, b.blend, t),
    stops: a.stops.map((s, i) => { const x = hex(s), y = hex(b.stops[i]); return toHex(x.map((v, k) => lerp(v, y[k], t))); }),
  };
}

const EASE_MS = 900;
const PULSE_MS = 1800;
const STEP_MS = 50; // ~20 Hz parameter updates while easing; the shader itself keeps its own frame rate

export function DreamAurora({ state, dim = false, className = '' }: { state: PresenceState; dim?: boolean; className?: string }) {
  const [look, setLook] = useState<Look>(LOOKS[state]);
  const [pulse, setPulse] = useState(0);
  const [lightMode, setLightMode] = useState(false);
  const from = useRef<Look>(LOOKS[state]);
  const shown = useRef<Look>(LOOKS[state]);
  const previous = useRef<PresenceState>(state);

  useEffect(() => {
    const read = () => setLightMode(document.documentElement.getAttribute('data-theme') === 'light');
    read();
    const mo = new MutationObserver(read);
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    return () => mo.disconnect();
  }, []);

  // Ease to the new state's look.
  useEffect(() => {
    from.current = shown.current;
    const target = LOOKS[state];
    const start = performance.now();
    const timer = setInterval(() => {
      const t = Math.min(1, (performance.now() - start) / EASE_MS);
      const eased = t * t * (3 - 2 * t);
      shown.current = mixLook(from.current, target, eased);
      setLook(shown.current);
      if (t >= 1) clearInterval(timer);
    }, STEP_MS);
    return () => clearInterval(timer);
  }, [state]);

  // A brief luminous pulse only when a task has genuinely just been reported successful.
  useEffect(() => {
    const was = previous.current;
    previous.current = state;
    if (state !== 'completed' || was === 'completed') return;
    const start = performance.now();
    const timer = setInterval(() => {
      const t = (performance.now() - start) / PULSE_MS;
      setPulse(t >= 1 ? 0 : Math.sin(Math.PI * Math.min(1, t * 1.4)) * (1 - t));
      if (t >= 1) clearInterval(timer);
    }, STEP_MS);
    return () => clearInterval(timer);
  }, [state]);

  return (
    <div className={`dream-aurora ${className}`} data-state={state} aria-hidden>
      {/* static fallback: shown when WebGL2 is unavailable (the canvas stays transparent) */}
      <div className="dream-aurora-fallback" />
      <Aurora
        colorStops={look.stops}
        speed={look.speed}
        amplitude={look.amplitude}
        brightness={(look.brightness + pulse * 0.45) * (dim ? 0.55 : 1)}
        glow={look.glow + pulse * 0.25}
        height={look.height + pulse * 0.05}
        blend={look.blend}
        layers={2}
        scale={0.9}
        opacity={dim ? 0.75 : 1}
        mouseInteraction={false}
        lightMode={lightMode}
        dpr={1}
      />
    </div>
  );
}
