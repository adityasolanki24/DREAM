'use client';
import { AnimatePresence, motion } from 'motion/react';
import { AlertTriangle, ArrowUp, Loader2, Mic, MicOff } from 'lucide-react';
import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { ModelPicker } from '../../ModelPicker';
import { useDraft } from '../../lib/draft';
import { useSession } from '../../lib/session';
import { useUi } from '../../lib/ui';
import { Tip } from '../ui';

type Recogniser = {
  lang: string; interimResults: boolean; start: () => void; stop: () => void;
  onresult: (e: { results: ArrayLike<ArrayLike<{ transcript: string }>> }) => void;
  onerror: (e: { error?: string }) => void; onend: () => void;
};

/** A chip with a placeholder such as "<place>" fills in the words before it, for the user to finish. */
export const fillExample = (x: string) => (x.includes('<') ? x.slice(0, x.indexOf('<')) : x);

/** Goal forms come from the chosen provider's `supports`, so chips never promise a form it rejects. */
export function useExamples(): string[] {
  const { models } = useSession();
  return models.choices.find((c) => c.id === models.choice?.provider)?.supports ?? [];
}

export function Suggestions({ className = '' }: { className?: string }) {
  const { link } = useSession();
  const { draft, setDraft, focus } = useDraft();
  const examples = useExamples();
  if (link !== 'online' || !examples.length || draft) return null;
  return (
    <div className={`suggestions ${className}`} aria-label="Supported goals">
      {examples.map((x) => (
        <button key={x} type="button" className="suggestion" onClick={() => { setDraft(fillExample(x)); requestAnimationFrame(focus); }}>{x}</button>
      ))}
    </div>
  );
}

export function Composer({ variant = 'docked', autoFocus = false }: { variant?: 'hero' | 'docked' | 'panel'; autoFocus?: boolean }) {
  const { link, models, submit, chatNotice } = useSession();
  const { setListening } = useUi();
  const { draft, setDraft, register } = useDraft();
  const [busy, setBusy] = useState(false);
  const [listening, setLocalListening] = useState(false);
  const [voiceNote, setVoiceNote] = useState('');
  const [voiceSupported, setVoiceSupported] = useState(false);
  const recogniser = useRef<{ stop: () => void } | null>(null);
  const area = useRef<HTMLTextAreaElement | null>(null);
  const online = link === 'online';
  const examples = useExamples();
  const placeholder = !online ? 'Connect to send a task'
    : examples.length ? `Ask DREAM to ${examples[0].replace(/<.*?>/, '…')}` : 'Describe a goal for DREAM…';

  useEffect(() => {
    const w = window as unknown as Record<string, unknown>;
    setVoiceSupported(Boolean(w.SpeechRecognition || w.webkitSpeechRecognition));
  }, []);
  useEffect(() => () => { recogniser.current?.stop(); setListening(false); }, [setListening]);
  useEffect(() => { if (autoFocus && online) area.current?.focus(); }, [autoFocus, online]);
  // Grow with the text up to a limit, then scroll.
  useLayoutEffect(() => {
    const el = area.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, variant === 'hero' ? 200 : 160)}px`;
  }, [draft, variant]);

  async function send() {
    const message = draft.trim();
    if (!message || !online || busy) return;
    setBusy(true);
    const ok = await submit(message);
    if (ok) setDraft('');
    setBusy(false);
  }

  function toggleVoice() {
    if (listening) { setListening(false); return recogniser.current?.stop(); }
    const w = window as unknown as Record<string, new () => Recogniser>;
    const Recognition = w.SpeechRecognition || w.webkitSpeechRecognition;
    if (!Recognition) return;
    const r = new Recognition();
    r.lang = navigator.language || 'en-US';
    r.interimResults = false;
    r.onresult = (e) => setDraft((old) => `${old}${old ? ' ' : ''}${e.results[0][0].transcript}`.trim());
    r.onerror = (e) => setVoiceNote(e.error === 'not-allowed' ? 'Microphone permission was denied.' : 'Voice input failed. You can type instead.');
    r.onend = () => { setLocalListening(false); setListening(false); };
    recogniser.current = r;
    setVoiceNote('');
    setLocalListening(true);
    setListening(true);
    r.start();
  }

  const canSend = online && !!draft.trim() && !busy;
  return (
    <div className={`composer-wrap v-${variant}`}>
      <AnimatePresence>
        {chatNotice && (
          <motion.div className="composer-notice" role="alert" initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
            <AlertTriangle size={13} aria-hidden /> {chatNotice}
          </motion.div>
        )}
      </AnimatePresence>
      <div className={`composer ${online ? '' : 'disabled'} ${listening ? 'listening' : ''}`} onClick={(e) => { if (e.target === e.currentTarget) area.current?.focus(); }}>
        <textarea
          ref={(el) => { area.current = el; register(el); }}
          aria-label="Task"
          rows={1}
          maxLength={500}
          disabled={!online}
          placeholder={placeholder}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              void send();
            }
            if (e.key === 'Escape') (e.target as HTMLTextAreaElement).blur();
          }}
        />
        <div className="composer-bar">
          <ModelPicker choices={models.choices} value={models.choice} onChange={models.select} disabled={!online} />
          <span className="composer-spacer" />
          {draft && online && <span className="composer-hint" aria-hidden><span className="kbd">↵</span> send · <span className="kbd">⇧↵</span> newline</span>}
          <Tip label={voiceSupported ? (listening ? 'Stop voice input' : 'Voice input') : 'Voice input is not supported in this browser'} side="top">
            <span>
              <button
                type="button"
                className={`icon-btn sm ${listening ? 'rec' : ''}`}
                aria-label={voiceSupported ? (listening ? 'Stop voice input' : 'Start voice input') : 'Voice input is not supported in this browser'}
                aria-pressed={listening}
                disabled={!online || !voiceSupported}
                onClick={toggleVoice}
              >
                {voiceSupported ? <Mic size={15} aria-hidden /> : <MicOff size={15} aria-hidden />}
              </button>
            </span>
          </Tip>
          <button type="button" className={`send ${canSend ? 'ready' : ''}`} aria-label="Send task" disabled={!canSend} onClick={() => void send()}>
            {busy ? <Loader2 size={15} className="spin" aria-hidden /> : <ArrowUp size={16} strokeWidth={2.4} aria-hidden />}
          </button>
        </div>
      </div>
      {voiceNote && <small className="composer-voice-note">{voiceNote}</small>}
    </div>
  );
}
