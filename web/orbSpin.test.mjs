import { test } from 'node:test';
import { deepStrictEqual, ok, strictEqual } from 'node:assert';
import {
  DETENT_RAD,
  detentsCrossed,
  dragStep,
  pointerAngle,
  SETTLE_V,
  shortestDelta,
  stepSpin,
  TAP_RAD,
} from './app/lib/orbSpin.mjs';

const close = (a, b, epsMax = 1e-9) => ok(Math.abs(a - b) < epsMax, `${a} should be within ${epsMax} of ${b}`);

test('shortestDelta takes the short way across the wrap', () => {
  strictEqual(shortestDelta(0, 0.3), 0.3);
  strictEqual(shortestDelta(0.3, 0), -0.3);
  close(shortestDelta(3.1, -3.1), 2 * Math.PI - 6.2, 1e-12);      // +0.083, not -6.2
  close(shortestDelta(-3.1, 3.1), -(2 * Math.PI - 6.2), 1e-12);
  close(shortestDelta(Math.PI - 0.01, -Math.PI + 0.01), 0.02, 1e-12);
  close(shortestDelta(0, 6.5), 6.5 - 2 * Math.PI, 1e-12);
  strictEqual(shortestDelta(Math.PI, -Math.PI) <= Math.PI, true);  // the boundary lands inside (including -PI)
});

test('pointerAngle follows screen axes (y points down)', () => {
  strictEqual(pointerAngle(10, 5, 5, 5), 0);            // east
  strictEqual(pointerAngle(5, 10, 5, 5), Math.PI / 2);  // south on screen
  strictEqual(pointerAngle(5, 0, 5, 5), -Math.PI / 2);  // north on screen
  strictEqual(pointerAngle(0, 5, 5, 5), Math.PI);       // west
});

test('stepSpin decays monotonically and always settles', () => {
  let angle = 0;
  let vel = 20;
  let settled = false;
  let frames = 0;
  let prevSpeed = 20;
  while (!settled && frames < 4000) {
    const r = stepSpin(angle, vel, 1 / 60);
    ok(Math.abs(r.vel) < prevSpeed + 1e-15, 'friction never speeds the spin up');
    prevSpeed = r.vel;
    angle = r.angle;
    vel = r.vel;
    settled = r.settled;
    frames++;
  }
  ok(settled, 'a released spin settles on its own');
  strictEqual(vel, 0);
  ok(angle > 0 && angle < 20 * 0.5, `displacement ${angle} is bounded by v0 * tau (10 rad)`);
});

test('stepSpin with no elapsed time is a no-op, not a settled spin', () => {
  deepStrictEqual(stepSpin(0.7, 5, 0), { angle: 0.7, vel: 5, settled: false });
});

test('dragStep tracks the pointer across the wrap so a drag never sawtooths', () => {
  let spin = { angle: 0, vel: 0 };
  let pa = 3.1;  // near +PI
  for (let i = 0; i < 4; i++) {
    const nextPa = pa + 0.1 > Math.PI ? pa + 0.1 - 2 * Math.PI : pa + 0.1;
    const r = dragStep(spin, pa, nextPa, i * 16, (i + 1) * 16);
    spin = { angle: r.angle, vel: r.vel };
    ok(r.d > 0.09 && r.d < 0.11, `each event advances ~0.1 rad, got ${r.d}`);
    pa = nextPa;
  }
  close(spin.angle, 0.4, 1e-9);  // the walked path, not a jumped sawtooth
});

test('dragStep floors dt so a stall cannot read as an infinite flick', () => {
  // A 1 ms event gap is floored to 4 ms, so a fast tiny wiggle cannot read as a huge flick:
  // 0.01 rad over 1 ms reports at most 0.01 / 0.004 = 2.5 rad/s (then the EMA weight -> 1.0).
  const r = dragStep({ angle: 0, vel: 0 }, 0, 0.01, 0, 1);
  ok(Math.abs(r.vel - 1) < 1e-9, `vel ${r.vel} should be 1 (iv = 2.5 * 0.4)`);
  close(r.angle, 0.01);
  // and a genuine 5 s stall decays the instantaneous velocity toward zero: no phantom flick
  const stall = dragStep({ angle: 0, vel: 0 }, 0, 0.01, 0, 5000);
  ok(Math.abs(stall.vel) < 0.01, `a 5 s stall reports ~zero speed, got ${stall.vel}`);
});

test('dragStep blends event velocity (EMA weight 0.4 per event)', () => {
  const r1 = dragStep({ angle: 0, vel: 0 }, 0, 0.1, 0, 16);   // iv = 6.25 -> 2.5
  const r2 = dragStep({ angle: r1.angle, vel: r1.vel }, 0.1, 0.2, 16, 32);
  close(r1.vel, 2.5);
  close(r2.vel, 4);   // 0.6 * 2.5 + 0.4 * 6.25
  close(r2.angle, 0.2);
});

test('a quarter turn is one detent, in both directions', () => {
  strictEqual(detentsCrossed(0, Math.PI / 2), 1);
  strictEqual(detentsCrossed(0, Math.PI), 2);
  strictEqual(detentsCrossed(Math.PI / 2, Math.PI), 1);
  strictEqual(detentsCrossed(0, -Math.PI / 2), 1);
  strictEqual(detentsCrossed(0, 0.3), 0);
  strictEqual(detentsCrossed(0, 0), 0);
  strictEqual(detentsCrossed(0, (3 * Math.PI) / 2), 3);
});

test('interaction constants sit in their calibrated ranges', () => {
  ok(TAP_RAD > 0 && TAP_RAD < 0.3, 'tap threshold: small but not zero');
  ok(SETTLE_V > 0 && SETTLE_V < 0.1, 'settle speed: near rest');
  strictEqual(DETENT_RAD, Math.PI / 2);
});
