// Static export for the Python backend to serve: DREAM_EXPORT=1 next build -> web/out
import { spawnSync } from 'node:child_process';
const run = spawnSync('npx', ['next', 'build'], { stdio: 'inherit', shell: true, env: { ...process.env, DREAM_EXPORT: '1' } });
process.exit(run.status ?? 1);
