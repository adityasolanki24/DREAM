'use client';
import * as Dialog from '@radix-ui/react-dialog';
import * as Dropdown from '@radix-ui/react-dropdown-menu';
import * as Tooltip from '@radix-ui/react-tooltip';
import { motion } from 'motion/react';
import { Bell, BellOff, Camera, Check, Command as CommandIcon, FlaskConical, KeyRound, Keyboard, Settings, Sparkles, TestTube2, X } from 'lucide-react';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { Toaster } from 'sonner';
import { CameraProvider } from '../../lib/camera';
import { useDraft } from '../../lib/draft';
import { RUNNABLE } from '../../lib/format';
import { useMediaQuery, usePref } from '../../lib/prefs';
import { useSession } from '../../lib/session';
import { MODES, UiContext, useUi, type Mode, type StageView, type Ui } from '../../lib/ui';
import { CompanionView } from '../companion/CompanionView';
import { LiveView } from '../live/LiveView';
import { AmbientBackdrop } from '../AmbientBackdrop';
import { StrandsGlyph, StrandsMark } from '../Strands';
import { ResearchView } from '../research/ResearchView';
import { Tip } from '../ui';
import { CommandPalette } from './CommandPalette';
import { ConnectForm, ConnectScreen, LINK_LABEL } from './ConnectScreen';

const MODE_META: Record<Mode, { label: string; icon: typeof Sparkles; key: string }> = {
  companion: { label: 'Companion', icon: Sparkles, key: '1' },
  live: { label: 'Live', icon: Camera, key: '2' },
  research: { label: 'Research', icon: FlaskConical, key: '3' },
};

function ModeSwitch({ mode, setMode }: { mode: Mode; setMode: (m: Mode) => void }) {
  return (
    <nav className="mode-switch" aria-label="Interface mode">
      {MODES.map((id) => {
        const { label, icon: Icon, key } = MODE_META[id];
        return (
          <Tip key={id} label={label} kbd={`Alt ${key}`}>
            <button aria-pressed={mode === id} aria-current={mode === id ? 'page' : undefined} onClick={() => setMode(id)}>
              {mode === id && <motion.span layoutId="mode-thumb" className="mode-thumb" transition={{ type: 'spring', duration: 0.34, bounce: 0.12 }} />}
              <Icon size={14} aria-hidden /><span>{label}</span>
            </button>
          </Tip>
        );
      })}
    </nav>
  );
}

function ConnectionBadge() {
  const { link } = useSession();
  const tone = link === 'online' ? 'good' : link === 'connecting' ? 'info' : 'warn';
  return (
    <span className="status-cluster">
      <span className={`conn conn-${link}`} role="status" aria-label={LINK_LABEL[link]}>
        <span className={`dot tone-${tone} ${link === 'connecting' ? 'pulse' : ''}`} aria-hidden />
        <span className="conn-text">{LINK_LABEL[link]}</span>
      </span>
    </span>
  );
}

function SettingsMenu() {
  const ui = useUi();
  const { notify, toggleNotify } = useSession();
  return (
    <Dropdown.Root>
      <Tip label="Settings"><Dropdown.Trigger asChild><button className="icon-btn" aria-label="Session settings"><Settings size={16} /></button></Dropdown.Trigger></Tip>
      <Dropdown.Portal>
        <Dropdown.Content className="menu" align="end" sideOffset={8} collisionPadding={8}>
          <Dropdown.Item className="menu-item" onSelect={ui.openSettings}><KeyRound size={14} aria-hidden />Change access token</Dropdown.Item>
          <Dropdown.Item className="menu-item" onSelect={() => void toggleNotify()}>{notify ? <BellOff size={14} aria-hidden /> : <Bell size={14} aria-hidden />}{notify ? 'Disable' : 'Enable'} notifications</Dropdown.Item>
          <Dropdown.Item className="menu-item" onSelect={() => ui.setPaletteOpen(true)}><Keyboard size={14} aria-hidden />Commands and shortcuts<span className="kbd">⌘K</span></Dropdown.Item>
          <Dropdown.Separator className="menu-sep" />
          <Dropdown.Label className="menu-label">Theme</Dropdown.Label>
          <Dropdown.RadioGroup value={ui.themeMode} onValueChange={(v) => ui.setThemeMode(v as Ui['themeMode'])}>
            {(['dark', 'light', 'system'] as const).map((m) => (
              <Dropdown.RadioItem key={m} value={m} className="menu-item" onSelect={(e) => e.preventDefault()}>
                {m === 'system' ? 'Follow system' : m[0].toUpperCase() + m.slice(1)}
                <Dropdown.ItemIndicator className="menu-check"><Check size={14} aria-hidden /></Dropdown.ItemIndicator>
              </Dropdown.RadioItem>
            ))}
          </Dropdown.RadioGroup>
        </Dropdown.Content>
      </Dropdown.Portal>
    </Dropdown.Root>
  );
}

const typing = (t: EventTarget | null) => {
  const el = t as HTMLElement | null;
  return !!el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT' || el.isContentEditable);
};

export function AppShell() {
  const session = useSession();
  const { link, error, connect, needsConnect, everConnected, shown, act, fixture, token } = session;
  const draft = useDraft();
  const mobile = useMediaQuery('(max-width: 860px)');
  const [mode, setMode] = usePref<Mode>('mode', 'companion', (v) => MODES.includes(v as Mode));
  const [stageView, setStageView] = usePref<StageView>('stage-view', 'camera', (v) => v === 'camera' || v === 'map');
  const [liveChat, setLiveChat] = usePref('live-chat', true, (v) => typeof v === 'boolean');
  const [companionExpanded, setCompanionExpanded] = usePref('companion-expanded', false, (v) => typeof v === 'boolean');
  const [focusMode, setFocusMode] = useState(false);
  const [listening, setListening] = useState(false);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [inspected, setInspected] = useState<string | null>(null);
  const [themeMode, setThemeState] = useState<Ui['themeMode']>('dark');
  useEffect(() => { setThemeState(window.dreamTheme?.get() ?? 'dark'); }, []);
  const setThemeMode = useCallback((m: Ui['themeMode']) => { window.dreamTheme?.set(m); setThemeState(m); }, []);
  const openSettings = useCallback(() => setSettingsOpen(true), []);
  useEffect(() => { if (mode !== 'live') setFocusMode(false); }, [mode]);

  const ui = useMemo<Ui>(() => ({
    mode, setMode, listening, setListening, paletteOpen, setPaletteOpen, openSettings, stageView, setStageView, liveChat, setLiveChat,
    companionExpanded, setCompanionExpanded, focusMode, setFocusMode, inspected, setInspected, themeMode, setThemeMode,
  }), [mode, setMode, listening, paletteOpen, openSettings, stageView, setStageView, liveChat, setLiveChat, companionExpanded, setCompanionExpanded, focusMode, inspected, themeMode, setThemeMode]);

  // Keyboard: ⌘K palette, Alt+1–3 modes, single keys only outside text fields.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); setPaletteOpen((v) => !v); return; }
      if (e.altKey && !e.metaKey && !e.ctrlKey && ['1', '2', '3'].includes(e.key)) { e.preventDefault(); setMode(MODES[Number(e.key) - 1]); return; }
      if (e.metaKey || e.ctrlKey || e.altKey || typing(e.target) || paletteOpen || settingsOpen || needsConnect) return;
      if (document.querySelector('[role="menu"], [role="dialog"]')) return;
      const k = e.key;
      const runnable = shown && RUNNABLE.includes(shown.state);
      if (k === '/') { e.preventDefault(); draft.focus(); }
      else if (k === 'm' || k === 'M') setStageView((v) => (v === 'camera' ? 'map' : 'camera'));
      else if ((k === 'c' || k === 'C') && mode === 'live') setLiveChat((v) => !v);
      else if ((k === 'f' || k === 'F') && mode === 'live' && !mobile) setFocusMode((v) => !v);
      else if ((k === 'e' || k === 'E') && mode === 'companion') setCompanionExpanded((v) => !v);
      else if ((k === 'p' || k === 'P') && runnable && shown) void act(shown.id, shown.state === 'suspended' ? 'resume' : 'pause');
      else if (k === 'X' && e.shiftKey && runnable && shown) void act(shown.id, 'cancel');
      else if (k === 'Escape' && focusMode) setFocusMode(false);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [mode, setMode, paletteOpen, settingsOpen, needsConnect, shown, act, draft, setStageView, setLiveChat, setCompanionExpanded, focusMode, mobile]);

  if (token === null) return null;

  return (
    <UiContext.Provider value={ui}>
      <Tooltip.Provider delayDuration={350} skipDelayDuration={150}>
        <CameraProvider>
          <div className="app" data-mode={mode} data-focus={focusMode}>
            <AmbientBackdrop />
            <header className="topbar">
              <div className="brand"><StrandsMark width={62} height={18} /><span>DREAM</span></div>
              {!needsConnect && !mobile ? <ModeSwitch mode={mode} setMode={setMode} /> : <span />}
              <div className="top-right">
                {fixture && (
                  <span className="fixture-badge" title="Deterministic local data for reviewing the layout; not robot data">
                    <TestTube2 size={12} aria-hidden />
                    <span className="fixture-kicker">Preview fixture</span>
                    <span className="fixture-name">{fixture.name}</span>
                  </span>
                )}
                <ConnectionBadge />
                {!needsConnect && !mobile && (
                  <button className="palette-trigger" onClick={() => setPaletteOpen(true)} aria-label="Open command palette">
                    <CommandIcon size={13} aria-hidden /><span className="palette-text">Commands</span><span className="kbd">⌘K</span>
                  </button>
                )}
                <SettingsMenu />
              </div>
            </header>
            {link === 'offline' && everConnected && !fixture && <div className="banner" role="alert">Connection lost. Showing the last data received; it is not live. Retrying…</div>}
            {fixture && link === 'offline' && <div className="banner" role="alert">Connection lost. Showing the last data received; it is not live. Retrying… <span className="faint">(fixture)</span></div>}
            {needsConnect ? <ConnectScreen link={link} error={error} onConnect={connect} /> : (
              <motion.main key={mode} className={`mode mode-${mode}`} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}>
                {mode === 'companion' && <CompanionView />}
                {mode === 'live' && <LiveView />}
                {mode === 'research' && <ResearchView />}
              </motion.main>
            )}
            {!needsConnect && mobile && (
              <nav className="tabbar" aria-label="Interface modes">
                {MODES.map((id) => {
                  const { label, icon: Icon } = MODE_META[id];
                  return <button key={id} aria-current={mode === id ? 'page' : undefined} onClick={() => setMode(id)}><Icon size={19} aria-hidden /><span>{label}</span></button>;
                })}
              </nav>
            )}
          </div>
          <CommandPalette />
          <Dialog.Root open={settingsOpen} onOpenChange={setSettingsOpen}>
            <Dialog.Portal>
              <Dialog.Overlay className="dialog-overlay" />
              <Dialog.Content className="dialog">
                <Dialog.Title>Session connection</Dialog.Title>
                <Dialog.Description>Update the token used for this browser tab.</Dialog.Description>
                <Dialog.Close className="icon-btn sm dialog-close" aria-label="Close"><X size={15} /></Dialog.Close>
                <ConnectForm link={link} error={error} onConnect={(v) => { connect(v); setSettingsOpen(false); }} />
              </Dialog.Content>
            </Dialog.Portal>
          </Dialog.Root>
          {/* Mobile: the topbar stacks two rows (~104px), so toasts go above the 60px safe-area-bordered tabbar instead. */}
          <Toaster position={mobile ? 'bottom-center' : 'bottom-right'} closeButton toastOptions={{ unstyled: false, className: 'dream-toast' }} offset={mobile ? 76 : 20} />
        </CameraProvider>
      </Tooltip.Provider>
    </UiContext.Provider>
  );
}
