export const STALL_GRACE_MS: number;
export function elapsedAgeS(baseAgeS: number | null, lastFrameAtMs: number | null, nowMs: number): number | null;
export function createFrameStore<T>(initial: T): {
  get(): T;
  select<R>(fn: (value: T) => R): R;
  set(next: T): void;
  subscribe(listener: () => void): () => void;
};
