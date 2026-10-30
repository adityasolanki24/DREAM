import test from 'node:test';
import assert from 'node:assert/strict';
import { FrameOrder } from './app/camera-client.mjs';
test('stale images and retired reconnect epochs cannot replace the current frame', () => {
 const f = new FrameOrder();
 assert.ok(f.accept('first', 1)); assert.ok(f.accept('first', 3));
 assert.equal(f.accept('first', 2), false); assert.equal(f.accept('first', 3), false);
 assert.ok(f.accept('reconnected', 1)); assert.equal(f.accept('first', 100), false);
 assert.ok(f.accept('reconnected', 2));
});
