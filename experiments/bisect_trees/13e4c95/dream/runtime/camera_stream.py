"""Latest-only camera publisher. Encoding and slow clients never own simulator stepping.

The simulation owner submits an immutable RGB copy; a single encoder worker has one
pending slot. Frame IDs include a stream epoch so reconnects cannot reuse an old ID.
All wall timestamps are server-clock observations, not claims of glass-to-glass latency.
"""

from __future__ import annotations

import io
import secrets
import threading
import time
from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class EncodedFrame:
    id: str
    sequence: int
    sim_time_s: float
    captured_unix_s: float
    encoded_unix_s: float
    encode_ms: float
    jpeg: bytes


class CameraStream:
    def __init__(self, encode=None, on_rgb=None):
        self.epoch = secrets.token_hex(8)
        self._condition = threading.Condition()
        self._pending = self._latest = None
        self._last_source = None
        self._sequence = self._submitted = self._dropped = self._encoded = 0
        self._closed = False
        self._error = None
        self._start = time.monotonic()
        self._encode = encode or self._jpeg
        self._on_rgb = on_rgb
        self._thread = threading.Thread(target=self._run, daemon=True, name="camera-encoder")
        self._thread.start()

    @staticmethod
    def _jpeg(rgb):
        out = io.BytesIO()
        Image.fromarray(rgb).convert("RGB").save(out, format="JPEG", quality=80)
        return out.getvalue()

    def submit(self, frame, captured_unix_s=None):
        """Nonblocking, bounded handoff. The caller must never mutate the submitted frame."""
        with self._condition:
            if self._closed or frame is self._last_source:
                return
            self._last_source = frame
            self._sequence += 1
            self._submitted += 1
            if self._pending is not None:
                self._dropped += 1
            self._pending = (
                self._sequence,
                float(frame.time),
                captured_unix_s or time.time(),
                frame.rgb.copy(),
            )
            self._condition.notify_all()

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._pending is not None or self._closed)
                if self._closed:
                    return
                sequence, sim_time, captured, rgb = self._pending
                self._pending = None
            try:
                start = time.perf_counter()
                jpeg = self._encode(rgb)
                encoded = EncodedFrame(
                    f"{self.epoch}-{sequence}",
                    sequence,
                    sim_time,
                    captured,
                    time.time(),
                    (time.perf_counter() - start) * 1000,
                    jpeg,
                )
                if self._on_rgb is not None:
                    self._on_rgb(rgb, encoded)
                with self._condition:
                    self._latest = encoded
                    self._encoded += 1
                    self._condition.notify_all()
            except Exception as error:
                with self._condition:
                    self._error = type(error).__name__

    def latest(self):
        with self._condition:
            return self._latest

    def metadata(self):
        with self._condition:
            frame = self._latest
            return {
                "stream_id": self.epoch,
                "transport": "authenticated binary JPEG",
                "frame_id": None if frame is None else frame.id,
                "sequence": 0 if frame is None else frame.sequence,
                "sim_time_s": None if frame is None else frame.sim_time_s,
                "captured_unix_s": None if frame is None else frame.captured_unix_s,
                "server_unix_s": time.time(),
                "frame_age_ms": None
                if frame is None
                else (time.time() - frame.captured_unix_s) * 1000,
                "encode_ms": None if frame is None else frame.encode_ms,
                "bytes": 0 if frame is None else len(frame.jpeg),
                "submitted_frames": self._submitted,
                "encoded_frames": self._encoded,
                "dropped_frames": self._dropped,
                "average_fps": self._encoded / max(time.monotonic() - self._start, 0.001),
                "error": self._error,
                "closed": self._closed,
            }

    def wait(self, after, timeout=1):
        with self._condition:
            self._condition.wait_for(
                lambda: self._closed or (self._latest is not None and self._latest.id != after),
                timeout=timeout,
            )
            return self._latest if self._latest is not None and self._latest.id != after else None

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._thread.join(timeout=2)
