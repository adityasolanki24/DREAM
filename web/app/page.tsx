'use client';
import { useState } from 'react';
import { AppShell } from './components/shell/AppShell';
import { DraftProvider } from './lib/draft';
import { loadFixture, type Fixture } from './lib/fixtures';
import { SessionProvider } from './lib/session';

/** Session state (polling, tasks, agent events) and the chat draft live above every mode so switching loses nothing. */
export default function Page() {
  // `?fixture=<name>` loads an interface-preview fixture in `next dev` only; production builds ignore it.
  const [fixture] = useState<Fixture | null>(() => (typeof window === 'undefined' ? null : loadFixture(new URLSearchParams(window.location.search).get('fixture'))));
  return (
    <SessionProvider fixture={fixture}>
      <DraftProvider>
        <AppShell />
      </DraftProvider>
    </SessionProvider>
  );
}
