import test from 'node:test';
import assert from 'node:assert/strict';
import { readTokens } from './theme-tokens.mjs';

const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
const lin = (c) => { const s = c / 255; return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4; };
const lum = ([r, g, b]) => 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
const ratio = (a, b) => { const [x, y] = [lum(a), lum(b)].sort((p, q) => q - p); return (x + 0.05) / (y + 0.05); };
const mix = (fg, bg, p) => fg.map((c, i) => c * p + bg[i] * (1 - p)); // color-mix(in srgb, fg p%, bg)

// [label, foreground, background, minimum ratio]
function pairs(t) {
  const c = (n) => hex(t[`--${n}`]);
  const out = [];
  for (const s of ['page', 'card', 'raised']) {
    out.push([`text-primary on ${s}`, c('text-primary'), c(s), 4.5], [`text-secondary on ${s}`, c('text-secondary'), c(s), 4.5]);
  }
  for (const n of ['accent', 'amber', 'danger', 'success']) {
    out.push([`${n} text on card`, c(n), c('card'), 4.5], [`${n} text on page`, c(n), c('page'), 4.5],
      [`${n} text on raised`, c(n), c('raised'), 4.5], [`${n} text on 12% tint over card`, c(n), mix(c(n), c('card'), 0.12), 4.5]);
  }
  out.push(['accent-contrast on accent', c('accent-contrast'), c('accent'), 4.5],
    ['page on text-primary (primary button)', c('page'), c('text-primary'), 4.5]);
  out.push(['hairline-strong (control border) on card', c('hairline-strong'), c('card'), 3],
    ['hairline-strong on page', c('hairline-strong'), c('page'), 3],
    ['focus-ring on card', c('focus-ring'), c('card'), 3], ['focus-ring on page', c('focus-ring'), c('page'), 3],
    ['focus-ring on raised', c('focus-ring'), c('raised'), 3],
    ['graph-edge on card', c('graph-edge'), c('card'), 3],
    ['graph node border (hairline-strong) on graph-node', c('hairline-strong'), c('graph-node'), 3],
    ['text-primary on graph-node', c('text-primary'), c('graph-node'), 4.5],
    ['text-primary on camera-backdrop', c('text-primary'), c('camera-backdrop'), 4.5],
    ['text-secondary on camera-backdrop', c('text-secondary'), c('camera-backdrop'), 4.5],
    ['map-occupied on map-free', c('map-occupied'), c('map-free'), 3], ['map-robot on map-free', c('map-robot'), c('map-free'), 3],
    ['map-trail on map-free', c('map-trail'), c('map-free'), 3], ['map-robot on map-unknown', c('map-robot'), c('map-unknown'), 3],
    ['map-occupied on map-unknown', c('map-occupied'), c('map-unknown'), 3],
    ['map-trail on map-unknown', c('map-trail'), c('map-unknown'), 3],
    ['map-free vs map-unknown (distinguishable)', c('map-free'), c('map-unknown'), 1.1]);
  return out;
}

for (const theme of ['light', 'dark']) {
  test(`${theme} theme meets WCAG contrast thresholds`, () => {
    const failures = [];
    for (const [label, fg, bg, min] of pairs(readTokens(theme))) {
      const r = ratio(fg, bg);
      if (process.env.CONTRAST_TABLE) console.log(`${theme}\t${label}\t${r.toFixed(2)}\t>=${min}`);
      if (r < min) failures.push(`${label}: ${r.toFixed(2)} < ${min}`);
    }
    assert.deepEqual(failures, []);
  });
}
