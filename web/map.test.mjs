import { test } from 'node:test';
import { deepStrictEqual } from 'node:assert';
import { mapPixels } from './app/map.mjs';
test('x-major map remains x horizontal and reverses y for canvas', () => {
  deepStrictEqual(mapPixels([[0, 1, 2], [2, 1, 0]]), [
    {x:0,y:2,value:0}, {x:0,y:1,value:1}, {x:0,y:0,value:2},
    {x:1,y:2,value:2}, {x:1,y:1,value:1}, {x:1,y:0,value:0},
  ]);
});

test('world coordinates land on the matching canvas pixel with y flipped', async () => {
  const { worldToPixel } = await import('./app/map.mjs');
  const map = { cells: [[0, 0, 0], [0, 0, 0]], origin_m: [1, 2], resolution_m: 0.5 };
  deepStrictEqual(worldToPixel(map, 1.25, 2.25), { px: 0.5, py: 2.5 });
  deepStrictEqual(worldToPixel(map, 5, 2), null);
});
