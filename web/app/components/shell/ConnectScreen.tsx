'use client';
import { ArrowRight, Loader2 } from 'lucide-react';
import { useState } from 'react';
import type { Link } from '../../types';
import StarBorder from '../reactbits/StarBorder';
import { DreamStrands, useThemeIsLight } from '../Strands';

export const LINK_LABEL: Record<Link, string> = {
  idle: 'Not connected',
  connecting: 'Connecting',
  online: 'Connected',
  offline: 'Backend unreachable',
  unauthorised: 'Token rejected',
};

const COPY: Record<Link, string> = {
  idle: "Enter this session's access token.",
  connecting: 'Reaching the robot backend…',
  online: 'You are connected.',
  offline: 'Check that the backend is running on this machine, then try again.',
  unauthorised: 'The access token was rejected. Check it and try again.',
};

/** Token form, used full-screen before the first connection and inside the settings dialog. */
export function ConnectForm({ link, error, onConnect, autoFocus = true }: { link: Link; error: string; onConnect: (token: string) => void; autoFocus?: boolean }) {
  const [value, setValue] = useState('');
  const light = useThemeIsLight();
  return (
    <form className={`connect-form connect-${link}`} onSubmit={(e) => { e.preventDefault(); if (value.trim()) onConnect(value.trim()); }}>
      <div className="connect-field">
        <input className="field" type="password" autoComplete="off" aria-label="Access token" placeholder="Session access token" autoFocus={autoFocus}
          value={value} onChange={(e) => setValue(e.target.value)} />
        <StarBorder as="button" type="submit" theme={light ? 'light' : 'dark'} radius={10} className="connect-submit"
          duration={5} disabled={!value.trim() || link === 'connecting'}>
          {link === 'connecting' ? <Loader2 size={14} className="spin" aria-hidden /> : null} Connect {link !== 'connecting' && <ArrowRight size={14} aria-hidden />}
        </StarBorder>
      </div>
      <p className={`connect-msg ${link === 'unauthorised' || link === 'offline' ? 'tone-warn' : ''}`}>{error || COPY[link]}</p>
    </form>
  );
}

export function ConnectScreen({ link, error, onConnect }: { link: Link; error: string; onConnect: (token: string) => void }) {
  return (
    <main className="connect-screen">
      <div className="connect-inner">
        <DreamStrands state={link === 'connecting' ? 'thinking' : link === 'unauthorised' ? 'failed' : 'offline'} size={132} />
        <h1>{link === 'offline' ? 'Cannot reach the robot backend' : link === 'connecting' ? 'Connecting…' : 'Connect to DREAM'}</h1>
        <ConnectForm link={link} error={error} onConnect={onConnect} />
      </div>
    </main>
  );
}
