// Pure helpers for the camera display: a tiny external store so per-frame state reaches only the components
// that show it, and the elapsed-time age used while frames have stopped arriving.

/** After this long without a frame the shown frame is stale and its age is counted in real elapsed time. */
export const STALL_GRACE_MS = 1500;

/**
 * Age of the shown frame. Between frames the capture-time estimate is kept as is (it must not drift with the
 * frame rate); once no frame has arrived for longer than the grace period, the real time since that frame is
 * added. This never lowers the age and never turns an unknown age into a number.
 */
export function elapsedAgeS(baseAgeS, lastFrameAtMs, nowMs) {
  if (baseAgeS == null) return null;
  if (lastFrameAtMs == null) return baseAgeS;
  const elapsed = nowMs - lastFrameAtMs;
  return elapsed > STALL_GRACE_MS ? baseAgeS + elapsed / 1000 : baseAgeS;
}

export function createFrameStore(initial) {
  let value = initial;
  const listeners = new Set();
  return {
    get: () => value,
    select: (fn) => fn(value),
    set(next) { value = next; for (const l of [...listeners]) l(); },
    subscribe(l) { listeners.add(l); return () => { listeners.delete(l); }; },
  };
}
