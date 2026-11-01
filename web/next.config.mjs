// `npm run build:export` produces a static site (web/out) that the Python backend serves itself.
// `npm run dev` proxies the phone API to the backend so the browser stays same-origin.
const backend = process.env.DREAM_BACKEND_URL || 'http://127.0.0.1:8765';
const exporting = process.env.DREAM_EXPORT === '1';

/** @type {import('next').NextConfig} */
// Strict Mode is off because its development-only double effect run makes the vendored React Bits Aurora call
// WEBGL_lose_context on its own canvas and then reuse that lost context. Production never double-runs effects.
const shared = { reactStrictMode: false };
const nextConfig = exporting
  ? { ...shared, output: 'export', images: { unoptimized: true } }
  : {
      ...shared,
      devIndicators: false,
      async rewrites() {
        return ['camera/:path*', 'media/:path*', 'status', 'tasks', 'tasks/:path*', 'snapshot', 'events', 'ag-ui/:path*', 'memory'].map((p) => ({
          source: `/${p}`,
          destination: `${backend}/${p}`,
        }));
      },
    };
export default nextConfig;
