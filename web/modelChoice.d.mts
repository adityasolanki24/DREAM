export type ProviderEntry = {
  id: string;
  label: string;
  available: boolean;
  reason: string | null;
  models: string[];
  default_model: string | null;
  supports: string[];
};
export type ModelChoice = { provider: string; model: string | null };

export const STORAGE_KEY: string;
export const STRUCTURED: string;
export function normaliseProviderChoices(raw: unknown): ProviderEntry[];
export function defaultModelFor(entry: ProviderEntry | undefined): string | null;
export function isValidChoice(choices: ProviderEntry[], choice: ModelChoice | null | undefined): boolean;
export function pickInitialChoice(choices: ProviderEntry[], defaultProvider: string | undefined, stored: ModelChoice | null | undefined): ModelChoice | null;
export function parseStoredChoice(text: string | null | undefined): ModelChoice | null;
export function readStoredChoice(storage: Storage | null | undefined): ModelChoice | null;
export function writeStoredChoice(storage: Storage | null | undefined, choice: ModelChoice): boolean;
export function buildTaskBody(text: string, choice: ModelChoice, priority?: number): { text: string; priority: number; provider: string; model?: string };
export function describeChoice(choices: ProviderEntry[], choice: ModelChoice | null): string;
export function describeMeta(choices: ProviderEntry[], meta: { provider?: string; model?: string | null } | null | undefined): string | null;
