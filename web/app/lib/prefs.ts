'use client';
import { useCallback, useEffect, useState } from 'react';

/** Interface preferences in localStorage (never the access token). Reads after mount so the static export hydrates cleanly. */
export function usePref<T>(key: string, initial: T, valid?: (v: unknown) => boolean): [T, (next: T | ((old: T) => T)) => void] {
  const storageKey = `dream-ui:${key}`;
  const [value, setValue] = useState<T>(initial);
  useEffect(() => {
    try {
      const raw = localStorage.getItem(storageKey);
      if (raw == null) return;
      const parsed = JSON.parse(raw);
      if (!valid || valid(parsed)) setValue(parsed);
    } catch {
      /* unavailable or corrupt: keep the default */
    }
  }, [storageKey]); // eslint-disable-line react-hooks/exhaustive-deps
  const set = useCallback((next: T | ((old: T) => T)) => {
    setValue((old) => {
      const resolved = typeof next === 'function' ? (next as (o: T) => T)(old) : next;
      try { localStorage.setItem(storageKey, JSON.stringify(resolved)); } catch { /* private mode */ }
      return resolved;
    });
  }, [storageKey]);
  return [value, set];
}

export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia(query);
    const update = () => setMatches(mq.matches);
    update();
    mq.addEventListener('change', update);
    return () => mq.removeEventListener('change', update);
  }, [query]);
  return matches;
}

export const layoutStorage = {
  getItem: (k: string) => { try { return localStorage.getItem(`dream-ui:${k}`); } catch { return null; } },
  setItem: (k: string, v: string) => { try { localStorage.setItem(`dream-ui:${k}`, v); } catch { /* private mode */ } },
};
