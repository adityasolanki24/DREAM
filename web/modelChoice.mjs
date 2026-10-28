// Pure helpers for the model picker. No React, no DOM: the picker and the submit path both use these.
// The backend (dream/runtime/phone_api.py) sends provider_choices; older snapshots only carry {id, label}.

export const STORAGE_KEY = 'dream-model-choice';
export const STRUCTURED = 'structured';

/** Turn whatever the snapshot sent into a complete list. Missing `available` means available; missing `models` means none. */
export function normaliseProviderChoices(raw) {
  if (!Array.isArray(raw)) return [];
  return raw
    .filter((x) => x && typeof x === 'object' && typeof x.id === 'string' && x.id)
    .map((x) => ({
      id: x.id,
      label: typeof x.label === 'string' && x.label ? x.label : x.id,
      available: x.available !== false,
      reason: typeof x.reason === 'string' && x.reason ? x.reason : null,
      models: Array.isArray(x.models) ? x.models.filter((m) => typeof m === 'string' && m) : [],
      default_model: typeof x.default_model === 'string' && x.default_model ? x.default_model : null,
      supports: Array.isArray(x.supports) ? x.supports.filter((s) => typeof s === 'string' && s) : [],
    }));
}

/** The model a provider uses when none is chosen: its default_model if it is listed, else its first model, else none. */
export function defaultModelFor(entry) {
  if (!entry || entry.models.length === 0) return null;
  return entry.default_model && entry.models.includes(entry.default_model) ? entry.default_model : entry.models[0];
}

/** True when a {provider, model} pair is something the backend would accept right now. */
export function isValidChoice(choices, choice) {
  if (!choice || typeof choice.provider !== 'string') return false;
  const entry = choices.find((c) => c.id === choice.provider);
  if (!entry || !entry.available) return false;
  if (entry.models.length === 0) return choice.model == null;
  return typeof choice.model === 'string' && entry.models.includes(choice.model);
}

/**
 * Pick the choice to show: the stored one if it is still valid, else the default provider, else the first
 * available provider. Returns null only when no provider is available at all.
 */
export function pickInitialChoice(choices, defaultProvider, stored) {
  if (isValidChoice(choices, stored)) return { provider: stored.provider, model: stored.model ?? null };
  const fallbacks = [defaultProvider, ...choices.map((c) => c.id)];
  for (const id of fallbacks) {
    const entry = choices.find((c) => c.id === id && c.available);
    if (entry) return { provider: entry.id, model: defaultModelFor(entry) };
  }
  return null;
}

/** Parse the stored value. Anything malformed is ignored rather than trusted. */
export function parseStoredChoice(text) {
  if (typeof text !== 'string' || !text) return null;
  try {
    const value = JSON.parse(text);
    if (!value || typeof value.provider !== 'string') return null;
    if (value.model != null && typeof value.model !== 'string') return null;
    return { provider: value.provider, model: value.model ?? null };
  } catch {
    return null;
  }
}

/** Read the remembered choice. `storage` is a Storage-like object or null (private mode, blocked storage). */
export function readStoredChoice(storage) {
  try {
    return parseStoredChoice(storage?.getItem(STORAGE_KEY) ?? null);
  } catch {
    return null;
  }
}

/** Remember the choice. Failure is silent: the picker still works for this page load. */
export function writeStoredChoice(storage, choice) {
  try {
    storage?.setItem(STORAGE_KEY, JSON.stringify({ provider: choice.provider, model: choice.model ?? null }));
    return true;
  } catch {
    return false;
  }
}

/** The POST /tasks body. `model` is omitted for the structured parser and whenever no model is chosen. */
export function buildTaskBody(text, choice, priority = 0) {
  const body = { text, priority, provider: choice.provider };
  if (choice.provider !== STRUCTURED && choice.model) body.model = choice.model;
  return body;
}

/** A short label such as "Structured parser" or "ChatGPT · gpt-6-luna". Falls back to the raw ids. */
export function describeChoice(choices, choice) {
  if (!choice) return 'No provider available';
  const entry = choices.find((c) => c.id === choice.provider);
  const label = entry ? entry.label : choice.provider;
  return choice.model && choice.provider !== STRUCTURED ? `${label} · ${choice.model}` : label;
}

/** Label for a task thread from the task's meta ({provider, model}), or null when the task carries none. */
export function describeMeta(choices, meta) {
  if (!meta || typeof meta.provider !== 'string') return null;
  return describeChoice(choices, { provider: meta.provider, model: meta.model ?? null });
}
