// Camera requests are independent of task/map polling. No overlapping image fetches.
export class FrameOrder {
  constructor() { this.epoch = null; this.sequence = -1; this.retired = new Set(); }
  accept(epoch, sequence) {
    if (!epoch || !Number.isFinite(sequence) || this.retired.has(epoch)) return false;
    if (this.epoch !== epoch) {
      if (this.epoch) this.retired.add(this.epoch);
      this.epoch = epoch; this.sequence = -1;
    }
    if (sequence <= this.sequence) return false;
    this.sequence = sequence; return true;
  }
}

/**
 * Incremental parser for the backend's MJPEG push stream (multipart/x-mixed-replace, boundary
 * "dreamframe", one Content-Length per part). Feed it network chunks; it returns complete parts.
 * Pure and allocation-light so a long-lived stream does not grow memory.
 */
export class MultipartJpeg {
  constructor() { this.buf = new Uint8Array(0); this.need = null; this.headers = null; }
  push(chunk) {
    const merged = new Uint8Array(this.buf.length + chunk.length);
    merged.set(this.buf); merged.set(chunk, this.buf.length);
    this.buf = merged;
    const parts = [];
    for (;;) {
      if (this.need == null) {
        const end = indexOfCrlfCrlf(this.buf);
        if (end < 0) break;
        const text = new TextDecoder().decode(this.buf.subarray(0, end));
        this.buf = this.buf.subarray(end + 4);
        const headers = {};
        for (const line of text.split('\r\n')) {
          const i = line.indexOf(':');
          if (i > 0) headers[line.slice(0, i).trim().toLowerCase()] = line.slice(i + 1).trim();
        }
        const length = Number(headers['content-length']);
        if (!Number.isFinite(length) || length < 0) continue; // boundary-only preamble
        this.need = length; this.headers = headers;
      }
      if (this.buf.length < this.need) break;
      parts.push({ headers: this.headers, body: this.buf.slice(0, this.need) });
      this.buf = this.buf.subarray(this.need);
      this.need = null; this.headers = null;
    }
    if (this.need == null && this.buf.length > 0) this.buf = this.buf.slice(); // drop consumed prefix
    return parts;
  }
}

function indexOfCrlfCrlf(b) {
  for (let i = 0; i + 3 < b.length; i++) if (b[i] === 13 && b[i + 1] === 10 && b[i + 2] === 13 && b[i + 3] === 10) return i;
  return -1;
}
