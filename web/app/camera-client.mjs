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
