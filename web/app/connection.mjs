export function createPollGate() {
  let generation = 0;
  let pending = null;
  return {
    invalidate() { generation++; pending = null; },
    run(work) {
      if (pending) return pending;
      const started = generation;
      const promise = Promise.resolve().then(() => work(() => generation === started));
      pending = promise;
      promise.finally(() => { if (pending === promise) pending = null; }).catch(() => {});
      return promise;
    },
  };
}
export async function fetchWithTimeout(fetcher, url, options, timeoutMs = 8000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetcher(url, { ...options, signal: controller.signal });
    const data = await response.json().catch((error) => {
      if (controller.signal.aborted) throw error;
      return {};
    });
    return { response, data };
  }
  finally { clearTimeout(timer); }
}
