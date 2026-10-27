import type { AGUIEvent } from '@ag-ui/core';
export type Tool = { id: string; name: string; args: Record<string, unknown>; status: string; message: string };
export type Thread = {
  id: string;
  text: string | null;
  state: string;
  tools: Tool[];
  end: { state: string; reason: unknown } | null;
  claim: { success: boolean; reason: unknown } | null;
};
export type Feed = { threads: Record<string, Thread>; calls: Record<string, string>; runs: Record<string, string> };
export type Notice = { tone: 'info' | 'good' | 'warn' | 'bad'; title: string; body?: unknown; thread?: string };
export type AgEvent = AGUIEvent;
export function emptyFeed(): Feed;
export function applyEvents(previous: Feed, events: AgEvent[]): { feed: Feed; notices: Notice[] };
export function endTitle(state: string): string;
export function argSummary(args: Record<string, unknown>): string;
