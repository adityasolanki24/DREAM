'use client';
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';

/** One chat draft for the whole app: it survives mode changes and reloads of this tab (sessionStorage). */
type Draft = { draft: string; setDraft: (v: string | ((old: string) => string)) => void; focus: () => void; register: (el: HTMLTextAreaElement | null) => void };

const DraftContext = createContext<Draft | null>(null);
const KEY = 'dream-draft';

export function DraftProvider({ children }: { children: ReactNode }) {
  const [draft, setDraftState] = useState('');
  const inputs = useRef(new Set<HTMLTextAreaElement>());
  useEffect(() => {
    try { setDraftState(sessionStorage.getItem(KEY) ?? ''); } catch { /* unavailable */ }
  }, []);
  const setDraft = useCallback((v: string | ((old: string) => string)) => {
    setDraftState((old) => {
      const next = typeof v === 'function' ? v(old) : v;
      try { if (next) sessionStorage.setItem(KEY, next); else sessionStorage.removeItem(KEY); } catch { /* unavailable */ }
      return next;
    });
  }, []);
  const register = useCallback((el: HTMLTextAreaElement | null) => {
    if (el) inputs.current.add(el);
    // Detached elements are pruned lazily in focus().
  }, []);
  const focus = useCallback(() => {
    for (const el of [...inputs.current]) {
      if (!el.isConnected) { inputs.current.delete(el); continue; }
      if (el.offsetParent !== null && !el.disabled) { el.focus(); el.setSelectionRange(el.value.length, el.value.length); return; }
    }
  }, []);
  const value = useMemo(() => ({ draft, setDraft, focus, register }), [draft, setDraft, focus, register]);
  return <DraftContext.Provider value={value}>{children}</DraftContext.Provider>;
}

export function useDraft(): Draft {
  const d = useContext(DraftContext);
  if (!d) throw new Error('useDraft outside DraftProvider');
  return d;
}
