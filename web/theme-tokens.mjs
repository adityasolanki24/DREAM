// Shared by the theme tests: parse the light and dark token blocks out of app/style.css.
import { readFileSync } from 'node:fs';

export function readTokens(theme) {
  const css = readFileSync(new URL('./app/style.css', import.meta.url), 'utf8');
  const start = css.indexOf(':root[data-theme="' + theme + '"]');
  const end = start < 0 ? -1 : css.indexOf('}', start);
  if (start < 0 || end < 0) throw new Error(`no :root[data-theme="${theme}"] block in style.css`);
  const block = css.slice(css.indexOf('{', start) + 1, end);
  const tokens = {};
  for (const [, name, value] of block.matchAll(/(--[\w-]+)\s*:\s*([^;]+);/g)) tokens[name] = value.trim();
  return tokens;
}
