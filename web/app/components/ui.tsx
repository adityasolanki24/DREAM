'use client';
import * as Tooltip from '@radix-ui/react-tooltip';
import { motion } from 'motion/react';
import { Ban, CheckCircle2, CircleDot, Clock, Loader2, Pause, XCircle } from 'lucide-react';
import { useId, type ReactElement, type ReactNode } from 'react';
import { STATE_LABEL, stateTone } from '../lib/format';

/** Accessible tooltip with an optional shortcut hint. */
export function Tip({ label, kbd, side = 'bottom', children }: { label: ReactNode; kbd?: string; side?: 'top' | 'bottom' | 'left' | 'right'; children: ReactElement }) {
  return (
    <Tooltip.Root>
      <Tooltip.Trigger asChild>{children}</Tooltip.Trigger>
      <Tooltip.Portal>
        <Tooltip.Content side={side} sideOffset={6} className="tooltip" collisionPadding={8}>
          {label}{kbd && <span className="kbd">{kbd}</span>}
          <Tooltip.Arrow className="tooltip-arrow" width={9} height={4} />
        </Tooltip.Content>
      </Tooltip.Portal>
    </Tooltip.Root>
  );
}

const STATE_ICON: Record<string, typeof Clock> = { queued: Clock, running: Loader2, suspended: Pause, succeeded: CheckCircle2, failed: XCircle, cancelled: Ban };

/** Task state: always an icon plus a text label, never colour alone. */
export function StateChip({ state, label }: { state: string; label?: string }) {
  const Icon = STATE_ICON[state] ?? CircleDot;
  return (
    <span className={`state-chip tone-${stateTone(state)}`}>
      <Icon size={12} aria-hidden className={state === 'running' ? 'spin' : ''} />
      {label ?? STATE_LABEL[state] ?? state}
    </span>
  );
}

/** Segmented control with a sliding thumb. */
export function Segmented<T extends string>({ value, options, onChange, label, size }: {
  value: T; options: { id: T; label: ReactNode; icon?: ReactNode; title?: string }[]; onChange: (v: T) => void; label: string; size?: 'sm';
}) {
  const layoutId = useId();
  return (
    <div className={`seg ${size ?? ''}`} role="group" aria-label={label}>
      {options.map((o) => (
        <button key={o.id} type="button" aria-pressed={value === o.id} onClick={() => onChange(o.id)} title={o.title}>
          {value === o.id && <motion.span layoutId={layoutId} className="seg-thumb" transition={{ type: 'spring', duration: 0.32, bounce: 0.12 }} />}
          {o.icon}{o.label}
        </button>
      ))}
    </div>
  );
}

/** A quiet empty or unavailable state: what is missing, why, and what makes it appear. */
export function Quiet({ icon, title, children, action }: { icon?: ReactNode; title: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="quiet">
      {icon && <div className="quiet-icon">{icon}</div>}
      <strong>{title}</strong>
      {children && <p>{children}</p>}
      {action}
    </div>
  );
}

export function Kbd({ children }: { children: ReactNode }) {
  return <span className="kbd">{children}</span>;
}
