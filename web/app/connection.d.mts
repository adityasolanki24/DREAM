export function createPollGate(): { invalidate(): void; run(work: (current: () => boolean) => Promise<void>): Promise<void> };
export function fetchWithTimeout(fetcher: typeof fetch, url: string, options: RequestInit, timeoutMs?: number): Promise<{ response: Response; data: any }>;
