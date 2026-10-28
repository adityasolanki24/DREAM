'use client';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { Check, ChevronDown } from 'lucide-react';
import * as Dropdown from '@radix-ui/react-dropdown-menu';
import styles from './ModelPicker.module.css';
import {
  describeChoice, normaliseProviderChoices, pickInitialChoice, readStoredChoice, writeStoredChoice,
  type ModelChoice, type ProviderEntry,
} from '../modelChoice.mjs';
import type { ProviderChoice } from './types';

function localStore(): Storage | null {
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

/**
 * The choice for new tasks. The stored choice wins while it is still valid; otherwise the default provider and
 * its default model. `select` remembers the choice in localStorage ('dream-model-choice', not sensitive).
 */
export function useModelChoice(providerChoices: ProviderChoice[] | undefined, defaultProvider: string | undefined) {
  const choices = useMemo(() => normaliseProviderChoices(providerChoices), [providerChoices]);
  const [chosen, setChosen] = useState<ModelChoice | null>(null);
  useEffect(() => {
    setChosen(readStoredChoice(localStore()));
  }, []);
  const choice = useMemo(() => pickInitialChoice(choices, defaultProvider, chosen), [choices, defaultProvider, chosen]);
  const select = useCallback((next: ModelChoice) => {
    setChosen(next);
    writeStoredChoice(localStore(), next);
  }, []);
  return { choices, choice, select };
}

const keyOf = (provider: string, model: string | null) => `${provider}|${model ?? ''}`;

function Hint({ entry }: { entry: ProviderEntry }) {
  if (entry.supports.length === 0) return null;
  return <span className={styles.itemHint}>Accepts: {entry.supports.join(' · ')}</span>;
}

type Props = {
  choices: ProviderEntry[];
  value: ModelChoice | null;
  onChange: (choice: ModelChoice) => void;
};

/** Compact pill that shows the current parser and model, and opens a menu of every provider. */
export function ModelPicker({ choices, value, onChange }: Props) {
  const byKey = new Map<string, ModelChoice>();
  for (const entry of choices) {
    if (!entry.available) continue;
    if (entry.models.length === 0) byKey.set(keyOf(entry.id, null), { provider: entry.id, model: null });
    for (const model of entry.models) byKey.set(keyOf(entry.id, model), { provider: entry.id, model });
  }
  const label = value ? describeChoice(choices, value) : 'No provider available';

  return (
    <Dropdown.Root>
      <Dropdown.Trigger asChild>
        <button type="button" className={styles.trigger} aria-label="Model for new tasks" disabled={!value}>
          <span className={styles.triggerText}>{label}</span>
          <ChevronDown size={16} aria-hidden className={styles.chev} />
        </button>
      </Dropdown.Trigger>
      <Dropdown.Portal>
        <Dropdown.Content className={styles.menu} align="start" side="top" sideOffset={8} collisionPadding={12}>
          <Dropdown.RadioGroup value={value ? keyOf(value.provider, value.model) : ''} onValueChange={(k) => {
            const next = byKey.get(k);
            if (next) onChange(next);
          }}>
            {choices.map((entry) => {
              if (entry.available && entry.models.length === 0) {
                return (
                  <Dropdown.RadioItem key={entry.id} value={keyOf(entry.id, null)} className={styles.item}>
                    <span className={styles.itemText}>
                      <span>{entry.label}</span>
                      <Hint entry={entry} />
                    </span>
                    <Dropdown.ItemIndicator className={styles.indicator}><Check size={16} aria-hidden /></Dropdown.ItemIndicator>
                  </Dropdown.RadioItem>
                );
              }
              return (
                <Dropdown.Group key={entry.id} className={styles.group}>
                  <Dropdown.Label className={styles.groupLabel}>{entry.label}</Dropdown.Label>
                  <Hint entry={entry} />
                  {entry.available ? entry.models.map((model) => (
                    <Dropdown.RadioItem key={model} value={keyOf(entry.id, model)} className={styles.item}>
                      <span className={styles.itemText}><span>{model}</span></span>
                      <Dropdown.ItemIndicator className={styles.indicator}><Check size={16} aria-hidden /></Dropdown.ItemIndicator>
                    </Dropdown.RadioItem>
                  )) : (
                    <Dropdown.Item disabled className={styles.item}>
                      <span className={styles.itemText}>
                        <span>Unavailable</span>
                        <span className={styles.reason}>{entry.reason ?? 'This provider is not configured on this backend.'}</span>
                      </span>
                    </Dropdown.Item>
                  )}
                </Dropdown.Group>
              );
            })}
          </Dropdown.RadioGroup>
        </Dropdown.Content>
      </Dropdown.Portal>
    </Dropdown.Root>
  );
}
