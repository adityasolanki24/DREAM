/**
 * Pointer-drag spin physics for the presence orb. Pure functions: the React component only wires
 * pointer events in, so the behaviour is testable under plain `node --test` without a browser.
 * Angles are radians; screen coordinates come from pointer events. (.mjs beside the TS libs,
 * same pattern as lib/sceneGraph.mjs.)
 */

/** Total angular movement under which a gesture still counts as a tap rather than a drag (rad, about 5 degrees). */
export const TAP_RAD = 0.09;
/** A haptic tick every time an active drag accumulates this much (rad, a quarter turn). */
export const DETENT_RAD = Math.PI / 2;
/** Angular speed under which a released spin counts as settled (rad/s). */
export const SETTLE_V = 0.02;

/** Shortest signed distance between two angles, wrapped to (-PI, PI]. */
export function shortestDelta(from, to) {
  let d = (to - from) % (Math.PI * 2);
  if (d > Math.PI) d -= Math.PI * 2;
  else if (d <= -Math.PI) d += Math.PI * 2;
  return d;
}

/** Angle of a pointer position around a centre, in screen coordinates (y grows downward). */
export function pointerAngle(x, y, cx, cy) {
  return Math.atan2(y - cy, x - cx);
}

/**
 * One integration step of a released spin: exponential friction with time constant `tau`.
 * Returned `settled` means the caller can drop its animation loop and keep the angle.
 */
export function stepSpin(angle, vel, dt, { tau = 0.5, eps = SETTLE_V } = {}) {
  if (dt <= 0) return { angle, vel, settled: false };
  const next = vel * Math.exp(-dt / tau);
  if (Math.abs(next) < eps) return { angle: angle + next * dt, vel: 0, settled: true };
  return { angle: angle + next * dt, vel: next, settled: false };
}

/**
 * Drag event -> spin update. `pv`/`pa` are the previous and current pointer angles about the
 * centre, `t0`/`t` are the previous and current event timestamps (performance.now() ms). The spin
 * angle advances by the shortest path (a crossing of the +-PI boundary wraps); the velocity is a
 * fixed-rate mix of the instantaneous velocity, and `dt` is floored so a stall cannot read as an
 * infinite flick.
 */
export function dragStep(spin, pv, pa, t0, t) {
  const d = shortestDelta(pv, pa);
  const dt = Math.max(4, t - t0) / 1000;
  const iv = d / dt;
  return {
    angle: spin.angle + d,
    vel: spin.vel * 0.6 + iv * 0.4,  // per-event EMA, fast enough to feel bound to the pointer
    d,
  };
}

/** How many detent ticks an accumulated drag has crossed between two totals. */
export function detentsCrossed(prev, total) {
  return Math.floor(Math.abs(total) / DETENT_RAD) - Math.floor(Math.abs(prev) / DETENT_RAD);
}
