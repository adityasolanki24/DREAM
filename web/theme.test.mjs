import test from 'node:test';
import assert from 'node:assert/strict';
import { readTokens } from './theme-tokens.mjs';

test('light and dark define exactly the same set of variables', () => {
  const light = Object.keys(readTokens('light')).sort();
  const dark = Object.keys(readTokens('dark')).sort();
  assert.deepEqual(light.filter((k) => !dark.includes(k)), [], 'tokens missing from dark');
  assert.deepEqual(dark.filter((k) => !light.includes(k)), [], 'tokens missing from light');
});

test('the tokens the spec requires exist in both themes', () => {
  const required = ['page', 'card', 'raised', 'hairline', 'text-primary', 'text-secondary', 'text-disabled', 'accent',
    'amber', 'danger', 'success', 'focus-ring', 'graph-edge', 'graph-node', 'dot-grid', 'camera-backdrop',
    'map-unknown', 'map-free', 'map-occupied', 'map-robot', 'map-trail'];
  for (const theme of ['light', 'dark']) {
    const t = readTokens(theme);
    for (const name of required) assert.ok(`--${name}` in t, `--${name} missing in ${theme}`);
  }
});
