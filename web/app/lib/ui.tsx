'use client';
import { createContext, useContext } from 'react';

export const MODES = ['companion', 'live', 'research'] as const;
export type Mode = (typeof MODES)[number];
export type StageView = 'camera' | 'map';

/** Interface state shared across modes (all of it persisted in localStorage except `listening` and the palette). */
export type Ui = {
  mode: Mode;
  setMode: (m: Mode) => void;
  listening: boolean;
  setListening: (v: boolean) => void;
  paletteOpen: boolean;
  setPaletteOpen: (v: boolean) => void;
  openSettings: () => void;
  stageView: StageView;
  setStageView: (v: StageView) => void;
  liveChat: boolean;
  setLiveChat: (v: boolean | ((o: boolean) => boolean)) => void;
  companionExpanded: boolean;
  setCompanionExpanded: (v: boolean | ((o: boolean) => boolean)) => void;
  focusMode: boolean;
  setFocusMode: (v: boolean | ((o: boolean) => boolean)) => void;
  inspected: string | null;
  setInspected: (id: string | null) => void;
  themeMode: 'system' | 'light' | 'dark';
  setThemeMode: (m: 'system' | 'light' | 'dark') => void;
};

export const UiContext = createContext<Ui | null>(null);

export function useUi(): Ui {
  const u = useContext(UiContext);
  if (!u) throw new Error('useUi outside UiContext');
  return u;
}
