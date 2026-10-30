// `npm run build:export` produces a static site (web/out) that the Python backend serves itself.
// `npm run dev` proxies the phone API to the backend so the browser stays same-origin.
const backend = process.env.DREAM_BACKEND_URL || 'http://127.0.0.1:8765';
const exporting = process.env.DREAM_EXPORT === '1';

/** @type {import('next').NextConfig} */
const nextConfig = exporting
  ? { output: 'export', images: { unoptimized: true } }
  : {
      async rewrites() {
        return ['camera/:path*', 'media/:path*', 'status', 'tasks', 'tasks/:path*', 'snapshot', 'events', 'ag-ui/:path*', 'memory'].map((p) => ({
          source: `/${p}`,
          destination: `${backend}/${p}`,
        }));
      },
    };
export default nextConfig;
