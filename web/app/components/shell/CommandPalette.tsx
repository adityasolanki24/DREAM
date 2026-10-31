'use client';
import * as Dialog from '@radix-ui/react-dialog';
import { Command } from 'cmdk';
import {
  Bell, BellOff, Camera, FlaskConical, KeyRound, Map as MapIcon, MessageSquare, Monitor, Moon, Pause, Play, Search, Sparkles, Square, Sun, TestTube2, TextCursorInput, Expand,
} from 'lucide-react';
import type { ReactNode } from 'react';
import { useDraft } from '../../lib/draft';
import { FIXTURE_NAMES } from '../../lib/fixtures';
import { RUNNABLE } from '../../lib/format';
import { useSession } from '../../lib/session';
import { useUi } from '../../lib/ui';
import { fillExample, useExamples } from '../chat/Composer';

function Item({ icon, children, kbd, onSelect, keywords }: { icon: ReactNode; children: ReactNode; kbd?: string; onSelect: () => void; keywords?: string[] }) {
  return (
    <Command.Item className="cmd-item" onSelect={onSelect} keywords={keywords}>
      <span className="cmd-icon">{icon}</span>
      <span className="cmd-text">{children}</span>
      {kbd && <span className="kbd">{kbd}</span>}
    </Command.Item>
  );
}

/** ⌘K: navigation and the actions the backend actually supports right now. */
export function CommandPalette() {
  const ui = useUi();
  const { shown, act, link, toggleNotify, notify, fixture } = useSession();
  const { setDraft, focus } = useDraft();
  const examples = useExamples();
  const close = () => ui.setPaletteOpen(false);
  const run = (fn: () => void) => () => { close(); requestAnimationFrame(fn); };
  const runnable = shown && RUNNABLE.includes(shown.state);
  const dev = process.env.NODE_ENV === 'development';

  return (
    <Dialog.Root open={ui.paletteOpen} onOpenChange={ui.setPaletteOpen}>
      <Dialog.Portal>
        <Dialog.Overlay className="dialog-overlay palette-overlay" />
        <Dialog.Content className="palette" aria-describedby={undefined}>
          <Dialog.Title className="sr-only">Command palette</Dialog.Title>
          <Command loop label="Command palette">
            <div className="cmd-input-row"><Search size={15} aria-hidden /><Command.Input className="cmd-input" placeholder="Go to, run or change…" autoFocus /><span className="kbd">Esc</span></div>
            <Command.List className="cmd-list scroll">
              <Command.Empty className="cmd-empty">No matching command.</Command.Empty>
              <Command.Group heading="Go to" className="cmd-group">
                <Item icon={<Sparkles size={15} />} kbd="Alt 1" onSelect={run(() => ui.setMode('companion'))}>Companion</Item>
                <Item icon={<Camera size={15} />} kbd="Alt 2" onSelect={run(() => ui.setMode('live'))}>Live</Item>
                <Item icon={<FlaskConical size={15} />} kbd="Alt 3" onSelect={run(() => ui.setMode('research'))}>Research</Item>
              </Command.Group>
              {link === 'online' && (
                <Command.Group heading="Conversation" className="cmd-group">
                  <Item icon={<TextCursorInput size={15} />} kbd="/" onSelect={run(focus)}>Write a goal</Item>
                  {examples.map((x) => (
                    <Item key={x} icon={<MessageSquare size={15} />} keywords={['goal', 'task']} onSelect={run(() => { setDraft(fillExample(x)); focus(); })}>Draft “{x}”</Item>
                  ))}
                </Command.Group>
              )}
              {runnable && shown && (
                <Command.Group heading={`Task · ${shown.text}`} className="cmd-group">
                  {shown.state === 'running' && <Item icon={<Pause size={15} />} kbd="P" onSelect={run(() => act(shown.id, 'pause'))}>Pause at the next safe point</Item>}
                  {shown.state === 'suspended' && <Item icon={<Play size={15} />} kbd="P" onSelect={run(() => act(shown.id, 'resume'))}>Resume</Item>}
                  <Item icon={<Square size={14} />} kbd="⇧X" onSelect={run(() => act(shown.id, 'cancel'))}>Cancel at the next safe point</Item>
                </Command.Group>
              )}
              <Command.Group heading="View" className="cmd-group">
                <Item icon={ui.stageView === 'camera' ? <MapIcon size={15} /> : <Camera size={15} />} kbd="M" onSelect={run(() => ui.setStageView(ui.stageView === 'camera' ? 'map' : 'camera'))}>
                  Show the {ui.stageView === 'camera' ? 'map' : 'camera'} on the live stage
                </Item>
                <Item icon={<MessageSquare size={15} />} kbd="C" onSelect={run(() => { ui.setMode('live'); ui.setLiveChat((v) => !v); })}>{ui.liveChat ? 'Hide' : 'Show'} the live conversation panel</Item>
                <Item icon={<Expand size={15} />} kbd="F" onSelect={run(() => { ui.setMode('live'); ui.setFocusMode((v) => !v); })}>{ui.focusMode ? 'Exit' : 'Enter'} stage focus</Item>
                <Item icon={<MessageSquare size={15} />} kbd="E" onSelect={run(() => { ui.setMode('companion'); ui.setCompanionExpanded((v) => !v); })}>{ui.companionExpanded ? 'Compact' : 'Expand'} the companion conversation</Item>
              </Command.Group>
              <Command.Group heading="Session" className="cmd-group">
                <Item icon={<KeyRound size={15} />} onSelect={run(ui.openSettings)}>Change access token</Item>
                <Item icon={notify ? <BellOff size={15} /> : <Bell size={15} />} onSelect={run(() => void toggleNotify())}>{notify ? 'Disable' : 'Enable'} desktop notifications</Item>
                <Item icon={<Moon size={15} />} keywords={['theme']} onSelect={run(() => ui.setThemeMode('dark'))}>Theme: dark</Item>
                <Item icon={<Sun size={15} />} keywords={['theme']} onSelect={run(() => ui.setThemeMode('light'))}>Theme: light</Item>
                <Item icon={<Monitor size={15} />} keywords={['theme']} onSelect={run(() => ui.setThemeMode('system'))}>Theme: follow system</Item>
              </Command.Group>
              {dev && (
                <Command.Group heading="Interface-preview fixtures (development only)" className="cmd-group">
                  {fixture && <Item icon={<TestTube2 size={15} />} onSelect={run(() => { location.search = ''; })}>Leave fixture</Item>}
                  {FIXTURE_NAMES.map((n) => (
                    <Item key={n} icon={<TestTube2 size={15} />} keywords={['fixture', 'preview']} onSelect={run(() => { location.search = `?fixture=${n}`; })}>Preview fixture: {n}</Item>
                  ))}
                </Command.Group>
              )}
            </Command.List>
          </Command>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
