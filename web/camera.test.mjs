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

test('MJPEG push parts are reassembled across arbitrary chunk boundaries', async () => {
  const { MultipartJpeg } = await import('./app/camera-client.mjs');
  const enc = new TextEncoder();
  const part = (id, bytes) => [enc.encode(`--dreamframe\r\nContent-Type: image/jpeg\r\nContent-Length: ${bytes.length}\r\nX-Frame-Id: ${id}\r\nX-Captured-Unix-S: 10.5\r\n\r\n`), bytes, enc.encode('\r\n')];
  const whole = new Uint8Array([...part('a', new Uint8Array([1, 2, 3])), ...part('b', new Uint8Array([4, 5, 13, 10, 13, 10, 6]))].flatMap((x) => [...x]));
  for (const size of [1, 2, 7, whole.length]) {
    const p = new MultipartJpeg();
    const got = [];
    for (let i = 0; i < whole.length; i += size) got.push(...p.push(whole.subarray(i, i + size)));
    assert.deepEqual(got.map((x) => x.headers['x-frame-id']), ['a', 'b']);
    assert.deepEqual([...got[1].body], [4, 5, 13, 10, 13, 10, 6]); // CRLFCRLF inside a JPEG body is data, not a header end
    assert.equal(got[0].headers['x-captured-unix-s'], '10.5');
  }
});
