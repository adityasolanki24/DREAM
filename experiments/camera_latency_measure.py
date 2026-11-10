"""Measure real camera delivery on one machine: real CameraStream + PhoneAPI, real-size frames.

Run: python -m experiments.camera_latency_measure --out results/camera_delivery_closeout_20261106
Optionally select scenarios with --only a,b.

Every frame carries a block-coded sequence number in its pixels; the browser decodes the number from
the displayed <img>, so latency is source timestamp -> pixels on screen, on ONE machine clock.
These are loopback results. A tailnet, WAN or IAP measurement needs a cloud session.

"""

from __future__ import annotations

import argparse
import csv
import http.client
import json
import os
import platform
import secrets
import socket
import statistics
import struct
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import numpy as np

from dream.runtime import phone_api
from dream.runtime.camera_stream import CameraStream
from dream.runtime.phone_api import PhoneAPI

ROOT = Path(__file__).resolve().parents[1]
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PLAYWRIGHT = os.environ.get(
    "DREAM_PLAYWRIGHT_CORE",
    r"C:\Users\24adi\OneDrive\DREAM\web\node_modules\playwright-core",
)
APP_ROOT = Path(os.environ.get("DREAM_WEB_OUT", r"C:\Users\24adi\OneDrive\DREAM\web\out"))
H, W = 240, 320  # dream/sim/g1_mobile.py HEAD_CAMERA (320x240)
BS = 16
BLOCKS = 19


# ---------------------------------------------------------------- clock and stamp
class Clock:
    """Wall-clock epoch seconds with sub-millisecond resolution, anchored once."""

    def __init__(self):
        self._wall, self._perf = time.time(), time.perf_counter()

    def now(self):
        return self._wall + (time.perf_counter() - self._perf)


def checksum(seq):
    return (seq ^ (seq >> 4) ^ (seq >> 8)) & 15


def stamp_bits(seq):
    seq &= 4095
    bits = [1, 0, 1] + [(seq >> (11 - i)) & 1 for i in range(12)]
    return bits + [(checksum(seq) >> (3 - i)) & 1 for i in range(4)]


def make_base(seed=7):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:H, 0:W]
    base = np.stack(
        [
            120 + 80 * np.sin(x / 37.0) * np.cos(y / 23.0),
            110 + 70 * np.cos(x / 29.0 + y / 41.0),
            100 + 60 * np.sin((x + y) / 31.0),
        ],
        axis=-1,
    )
    return base, [rng.normal(0, 3, (H, W, 3)) for _ in range(8)]  # camera.py rgb_noise = 3


def decode_stamp(rgb):
    bits = []
    for k in range(BLOCKS):
        block = rgb[BS // 4 : 3 * BS // 4, k * BS + BS // 4 : k * BS + 3 * BS // 4]
        bits.append(1 if block.mean() > 127.5 else 0)
    if bits[:3] != [1, 0, 1]:
        return None
    seq = int("".join(map(str, bits[3:15])), 2)
    return seq if int("".join(map(str, bits[15:19])), 2) == checksum(seq) else None


def stats(values):
    v = sorted(values)
    if not v:
        return {"n": 0}
    p95 = v[min(len(v) - 1, int(round(0.95 * (len(v) - 1))))]
    return {
        "n": len(v),
        "median": round(statistics.median(v), 2),
        "p95": round(p95, 2),
        "max": round(v[-1], 2),
        "min": round(v[0], 2),
        "mean": round(statistics.fmean(v), 2),
    }


# ---------------------------------------------------------------- rig
class TrackingServer(ThreadingHTTPServer):
    """Harness-only: remembers connections so a test can drop them abruptly."""

    daemon_threads = True

    def __init__(self, *args, **kw):
        self.conns, self.conn_lock = set(), threading.Lock()
        super().__init__(*args, **kw)

    def get_request(self):
        sock, addr = super().get_request()
        with self.conn_lock:
            self.conns.add(sock)
        return sock, addr

    def shutdown_request(self, request):
        with self.conn_lock:
            self.conns.discard(request)
        super().shutdown_request(request)

    def drop_all(self):
        with self.conn_lock:
            socks = list(self.conns)
        for s in socks:
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        return len(socks)


phone_api._QuietServer = (
    TrackingServer  # harness: track sockets (pre-fix servers: ThreadingHTTPServer)
)


class StubScheduler:
    running = None

    def tasks(self):
        return []

    def submit(self, *a, **k):
        return "x"


class Rig:
    def __init__(self, token, web_root, rate=10.0):
        self.clock, self.token, self.web_root, self.rate = Clock(), token, web_root, rate
        self.src, self.enc, self.events = {}, {}, []
        self.seq = 0
        self.base, self.noise = make_base()
        self.run_source = threading.Event()
        self.alive = True
        self.port = 0
        self.slot_free_at = None
        self.stream = self._new_stream()
        self.api = self._new_api()
        self.server = self.api.serve(port=0)
        self.port = self.server.server_address[1]
        self.ctl = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.ctl.daemon_threads = True
        threading.Thread(target=self.ctl.serve_forever, daemon=True).start()
        self.ctl_port = self.ctl.server_address[1]
        self._thread = threading.Thread(target=self._source, daemon=True)
        self._thread.start()

    # -- construction
    def _new_stream(self):
        return CameraStream(on_rgb=self._on_rgb)

    def _new_api(self):
        api = PhoneAPI(
            StubScheduler(),
            self.token,
            status=lambda: {
                "robot_time_s": time.time() % 1000,
                "odometry": [0, 0, 0],
                "stop_reason": None,
            },
            snapshot=lambda: {
                "provider": "structured",
                "provider_label": "harness",
                "provider_choices": [],
                "robot_time_s": time.time() % 1000,
                "odometry": [0, 0, 0],
                "stop_reason": None,
                "camera": None,
                "map": None,
                "objects": [],
                "persistent_memory": None,
                "task_graph": None,
                "outcome": None,
                "history": [],
            },
            web_root=self.web_root,
        )
        api.camera = self.stream
        return api

    def _on_rgb(self, rgb, encoded):
        seq = decode_stamp(rgb)
        if seq is not None:
            self.enc[seq] = (encoded.encoded_unix_s, encoded.encode_ms, len(encoded.jpeg))

    # -- source
    def frame(self, seq):
        img = np.roll(self.base, seq * 2, axis=1) + self.noise[seq % 8]
        rgb = np.clip(img, 0, 255).astype(np.uint8)
        for k, bit in enumerate(stamp_bits(seq)):
            rgb[0:BS, k * BS : (k + 1) * BS] = 255 if bit else 0
        rgb[0:BS, BLOCKS * BS :] = 0
        return SimpleNamespace(time=seq / self.rate, rgb=rgb)

    def _source(self):
        nxt = time.perf_counter()
        while self.alive:
            if not self.run_source.is_set():
                time.sleep(0.01)
                nxt = time.perf_counter()
                continue
            now = time.perf_counter()
            if now < nxt:
                time.sleep(min(nxt - now, 0.005))
                continue
            self.seq += 1
            frame = self.frame(self.seq)
            t = self.clock.now()
            self.src[self.seq % 4096] = t  # keyed by the stamped (12-bit) number
            self.stream.submit(frame, captured_unix_s=t)
            nxt += 1.0 / self.rate
            if nxt < now - 1.0:
                nxt = now

    def start(self, rate):
        self.rate = float(rate)
        self.run_source.set()

    # -- faults
    def drop(self):
        return self.server.drop_all()

    def swap(self):
        old, self.stream = self.stream, self._new_stream()
        self.api.camera = self.stream
        old.close()

    def block(self, seconds):
        """Drop every connection, then hold both stream slots (the real 429 refusal path)."""
        self.drop()
        got = 0
        deadline = time.monotonic() + 10
        while got < 2 and time.monotonic() < deadline:
            if self.api._media_slots.acquire(timeout=0.05):
                got += 1

        def release():
            time.sleep(seconds)
            for _ in range(got):
                self.api._media_slots.release()

        threading.Thread(target=release, daemon=True).start()
        return got

    def restart(self, down_s):
        """Whole-server restart on the same port: new epoch, new slots, connections gone."""
        self.server.shutdown()
        self.server.server_close()
        self.drop()
        old = self.stream
        time.sleep(down_s)
        self.stream = self._new_stream()
        self.api = self._new_api()
        self.server = self.api.serve(port=self.port)
        old.close()

    def poll_slots_free(self, name):
        def run():
            t0 = self.clock.now()
            while self.clock.now() - t0 < 20:
                a = [self.api._media_slots.acquire(blocking=False) for _ in range(2)]
                for ok in a:
                    if ok:
                        self.api._media_slots.release()
                if all(a) and self.clock.now() - t0 > 0.002:
                    self.events.append({"name": name + "_slots_free", "t": self.clock.now()})
                    return
                time.sleep(0.01)

        threading.Thread(target=run, daemon=True).start()

    def _handler(self):
        rig = self

        class Ctl(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                u = urlparse(self.path)
                q = {k: v[0] for k, v in parse_qs(u.query).items()}
                name = u.path.removeprefix("/ctl/")
                t = rig.clock.now()
                out = {"t": t}
                if name == "cpu":
                    out["process_s"] = time.process_time()
                elif name == "source":
                    (rig.run_source.set if q.get("run") == "1" else rig.run_source.clear)()
                    rig.events.append({"name": "source_" + q.get("run", ""), "t": t})
                elif name == "drop":
                    out["dropped"] = rig.drop()
                elif name == "swap":
                    rig.swap()
                elif name == "block":
                    out["slots_held"] = rig.block(float(q.get("seconds", 5)))
                    out["t_released_after_s"] = float(q.get("seconds", 5))
                elif name == "restart":
                    rig.restart(float(q.get("down_s", 2)))
                    out["t_up"] = rig.clock.now()
                elif name == "mark":
                    rig.events.append({"name": q.get("name"), "t": t})
                    if q.get("name") == "chrome_kill":
                        rig.poll_slots_free("chrome_kill")
                body = json.dumps(out).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        return Ctl

    def close(self):
        self.alive = False
        self.run_source.clear()
        try:
            self.server.shutdown()
            self.server.server_close()
        except OSError:
            pass
        self.drop()
        self.ctl.shutdown()
        self.ctl.server_close()
        self.stream.close()


# ---------------------------------------------------------------- driver glue
def run_driver(rig, out_dir, name, **cfg):
    cfg_path, out_path = out_dir / f"_{name}.cfg.json", out_dir / f"_{name}.raw.json"
    cfg = {
        "chrome": CHROME,
        "token": rig.token,
        "base": f"http://127.0.0.1:{rig.port}",
        "ctl": f"http://127.0.0.1:{rig.ctl_port}",
        "out": str(out_path),
        **cfg,
    }
    cfg_path.write_text(json.dumps(cfg))
    env = {**os.environ, "DREAM_PLAYWRIGHT_CORE": PLAYWRIGHT}
    log = (out_dir / f"_{name}.driver.log").open("w")
    p = subprocess.run(
        ["node", str(ROOT / "experiments/camera_latency_driver.cjs"), str(cfg_path)],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        timeout=cfg.get("timeout_s", 180),
    )
    log.close()
    cfg_path.unlink(missing_ok=True)
    raw = json.loads(out_path.read_text()) if out_path.exists() else {}
    raw["driver_exit"] = p.returncode
    return raw


def offset_of(samples):
    """Server clock = page clock + offset. Best sample = smallest round trip."""
    best = min(samples, key=lambda s: s["t1"] - s["t0"])
    off = best["ts"] - (best["t0"] + best["t1"]) / 2
    return off, (best["t1"] - best["t0"]) / 2, [s["ts"] - (s["t0"] + s["t1"]) / 2 for s in samples]


def clock_offset(raw):
    a, ua, sa = offset_of(raw["sync_start"])
    b, ub, sb = offset_of(raw["sync_end"])
    allv = sa + sb
    return {
        "offset_start_ms": round(a, 3),
        "offset_end_ms": round(b, 3),
        "drift_ms": round(b - a, 3),
        "rtt_half_start_ms": round(ua, 3),
        "rtt_half_end_ms": round(ub, 3),
        "offset_spread_ms": round(max(allv) - min(allv), 3),
        # +-: best-sample RTT/2 plus observed drift plus 1 animation frame (watcher resolution)
        "uncertainty_ms": round(max(ua, ub) + abs(b - a) + 16.7, 2),
        "offset_used_ms": (a + b) / 2,
    }


def join_paints(rig, raw):
    """Per painted frame: seq, source time, paint time on the server clock, latency."""
    co = clock_offset(raw)
    rows = []
    for p in raw.get("paints", []):
        if not p["ok"]:
            rows.append(
                {"seq": -1, "decode_ok": 0, "t_paint": (p["t"] + co["offset_used_ms"]) / 1000}
            )
            continue
        t_paint = (p["t"] + co["offset_used_ms"]) / 1000
        t_src = rig.src.get(p["seq"])
        enc = rig.enc.get(p["seq"])
        rows.append(
            {
                "seq": p["seq"],
                "decode_ok": 1,
                "t_src_s": t_src,
                "t_paint_s": t_paint,
                "latency_ms": None if t_src is None else (t_paint - t_src) * 1000,
                "encode_ms": None if enc is None else enc[1],
                "jpeg_bytes": None if enc is None else enc[2],
                "t_encoded_s": None if enc is None else enc[0],
            }
        )
    return rows, co


def delivery_summary(rig, rows, rate):
    good = [r for r in rows if r.get("decode_ok") and r.get("latency_ms") is not None]
    if not good:
        return {"painted": 0}
    seqs = [r["seq"] for r in good]
    t = [r["t_paint_s"] for r in good]
    produced = set(range(min(seqs), max(seqs) + 1))
    painted = set(seqs)
    encoded_missing = [s for s in produced if s not in rig.enc]
    iv = np.diff(t) * 1000 if len(t) > 1 else np.array([])
    span = (t[-1] - t[0]) if len(t) > 1 else 0
    jitter = {}
    if len(iv):
        med = float(np.median(iv))
        jitter = {
            "interval_median_ms": round(med, 2),
            "interval_stdev_ms": round(float(np.std(iv)), 2),
            "interval_p95_ms": round(float(np.percentile(iv, 95)), 2),
            "abs_dev_from_median_p95_ms": round(float(np.percentile(np.abs(iv - med), 95)), 2),
            "interval_max_ms": round(float(iv.max()), 2),
        }
    return {
        "painted": len(painted),
        "produced_in_range": len(produced),
        "missing_seqs": len(produced - painted),
        "missing_not_encoded_by_server": len(encoded_missing),
        "missing_after_encode": len(produced - painted) - len(encoded_missing),
        "loss_pct": round(100 * len(produced - painted) / max(len(produced), 1), 2),
        "delivered_fps": round((len(painted) - 1) / span, 2) if span > 0 else 0,
        "decode_failures": sum(1 for r in rows if not r.get("decode_ok")),
        "latency_ms": stats([r["latency_ms"] for r in good]),
        "jitter": jitter,
        "rate_source_fps": rate,
    }


def write_csv(path, rows, cols):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})


CSV_COLS = [
    "seq",
    "decode_ok",
    "t_src_s",
    "t_encoded_s",
    "t_paint_s",
    "latency_ms",
    "encode_ms",
    "jpeg_bytes",
]


def new_rig(args, rate=10, web_root=None):
    return Rig(args.token, web_root or args.raw_root, rate)


# ---------------------------------------------------------------- scenarios
TOKEN_FOR_REDACTION = [""]


def raw_request(port, method, path, headers=None, body=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    c.request(method, path, body=body, headers=headers or {})
    r = c.getresponse()
    data = r.read(200)
    out = {
        "request": f"{method} {path}".replace(TOKEN_FOR_REDACTION[0], "<token>"),
        "sent_authorization": (headers or {}).get("Authorization", "<none>")[:11] + "...",
        "status": r.status,
        "reason": r.reason,
        "body": data.decode("latin1")[:120],
    }
    c.close()
    return out


def scn_auth(args, out):
    rig = new_rig(args)
    rig.start(10)
    time.sleep(1.5)
    tok, bad = args.token, "wrong-" + secrets.token_urlsafe(24)
    TOKEN_FOR_REDACTION[0] = tok
    cases = []
    for path in ("/camera/frame", "/camera/meta", "/camera/stream"):
        for label, headers, p in (
            ("no token", {}, path),
            ("wrong token", {"Authorization": f"Bearer {bad}"}, path),
            ("token in query only (leak check)", {}, path + f"?token={tok}"),
            ("access_token in query only", {}, path + f"?access_token={tok}"),
            ("Basic scheme with token", {"Authorization": f"Basic {tok}"}, path),
            ("correct bearer", {"Authorization": f"Bearer {tok}"}, path),
        ):
            if path == "/camera/stream" and label == "correct bearer":
                # streaming: read just the status line
                s = socket.create_connection(("127.0.0.1", rig.port), timeout=5)
                s.sendall(
                    f"GET {p} HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {tok}\r\n\r\n".encode()
                )
                line = s.recv(100).split(b"\r\n")[0].decode()
                s.close()
                cases.append(
                    {
                        "label": label,
                        "request": f"GET {p}".replace(tok, "<token>"),
                        "status_line": line,
                    }
                )
                continue
            r = raw_request(rig.port, "GET", p, headers)
            r["label"] = label
            cases.append(r)
    cases.append(
        {
            **raw_request(
                rig.port,
                "POST",
                "/camera/client-metrics",
                {"Content-Type": "application/json"},
                b"{}",
            ),
            "label": "POST client-metrics no token",
        }
    )
    time.sleep(0.5)
    echoed = any(tok in c.get("body", "") or tok in c.get("status_line", "") for c in cases)
    result = {
        "cases": cases,
        "token_echoed_in_any_response_text": echoed,
        "pass": all(
            c.get("status") == 401 or c.get("status_line", "").endswith("401 Unauthorized")
            for c in cases
            if c["label"] not in ("correct bearer", "POST client-metrics no token")
        )
        and not echoed,
    }
    rig.close()
    out["auth"] = result


def latency_run(args, out_dir, client, mode, rate, duration, scenario="latency", **extra):
    rig = new_rig(args, rate, web_root=args.app_root if client == "app" else args.raw_root)
    rig.start(rate)
    name = f"{scenario}_{client}_{mode}_{rate}fps"
    raw = run_driver(
        rig,
        out_dir,
        name,
        client=client,
        mode=mode,
        scenario=scenario,
        duration_s=duration,
        screenshot=str(out_dir / f"screenshot_{name}.png") if client == "app" else None,
        **extra,
    )
    meta = rig.stream.metadata()
    summary = {
        "client": client,
        "transport": mode,
        "source_fps": rate,
        "duration_s": duration,
        "driver_exit": raw.get("driver_exit"),
        "first_paint": raw.get("first_paint"),
        "chrome_version": raw.get("chrome_version"),
    }
    if "paints" in raw:
        rows, co = join_paints(rig, raw)
        write_csv(out_dir / f"{name}.csv", rows, CSV_COLS)
        summary.update(delivery_summary(rig, rows, rate))
        summary["clock"] = co
        summary["stream_metadata"] = {
            k: meta[k] for k in ("submitted_frames", "encoded_frames", "dropped_frames", "error")
        }
        if raw.get("cpu_start") and raw.get("cpu_end"):
            a, b = raw["cpu_start"], raw["cpu_end"]
            wall = (b["wall_ms"] - a["wall_ms"]) / 1000
            summary["server_process_cpu_pct_of_one_core"] = round(
                100 * (b["server_process_s"] - a["server_process_s"]) / wall, 1
            )
            summary["browser_cpu_pct_of_one_core_by_process_type"] = {
                k: round(
                    100
                    * (b["browser_by_type_s"].get(k, 0) - a["browser_by_type_s"].get(k, 0))
                    / wall,
                    1,
                )
                for k in b["browser_by_type_s"]
            }
        enc = [v[1] for s, v in rig.enc.items()]
        summary["encode_ms"] = stats(enc)
        summary["jpeg_bytes_median"] = (
            int(statistics.median([v[2] for v in rig.enc.values()])) if rig.enc else None
        )
        summary["urls_containing_token"] = raw.get("urls_with_token")
        summary["requests_made_by_page"] = raw.get("request_count")
        for k in ("rerender", "body_text", "console_errors", "app_img_missing", "responses"):
            if raw.get(k):
                summary[k] = raw[k]
    return rig, raw, summary


def scn_latency(args, out):
    results = []
    for client, modes in (("raw", ("poll", "push")), ("app", ("auto",))):
        for mode in modes:
            for rate in (5, 10, 15):
                rig, raw, summary = latency_run(args, args.out, client, mode, rate, args.duration)
                if client == "app":
                    summary["note"] = (
                        "real built web app; transport = its own policy (push first); "
                        "mode_in_use below"
                    )
                    summary["dom_labels_seen"] = sorted(
                        {label for d in raw.get("dom", []) for label in d["found"]}
                    )
                rig.close()
                print(
                    client,
                    mode,
                    rate,
                    json.dumps(summary.get("latency_ms")),
                    summary.get("delivered_fps"),
                    flush=True,
                )
                results.append(summary)
    out["latency"] = results


def scn_app_poll_only(args, out):
    """Force the app to poll by holding both stream slots for the whole run."""
    results = []
    for rate in (5, 10, 15):
        rig = new_rig(args, rate, web_root=args.app_root)
        rig.start(rate)
        held = [rig.api._media_slots.acquire(blocking=False) for _ in range(2)]
        name = f"latency_app_pollfallback_{rate}fps"
        raw = run_driver(
            rig,
            args.out,
            name,
            client="app",
            mode="auto",
            scenario="latency",
            duration_s=args.duration,
            first_paint_timeout_ms=30000,
        )
        s = {
            "client": "app",
            "transport": "poll (stream refused with 429 by holding both slots)",
            "source_fps": rate,
            "slots_held": sum(held),
        }
        if "paints" in raw:
            rows, co = join_paints(rig, raw)
            write_csv(args.out / f"{name}.csv", rows, CSV_COLS)
            s.update(delivery_summary(rig, rows, rate))
            s["clock"] = co
        results.append(s)
        print("app poll", rate, json.dumps(s.get("latency_ms")), s.get("delivered_fps"), flush=True)
        rig.close()
    out["latency_app_poll"] = results


def scn_rerender(args, out):
    rig, raw, s = latency_run(args, args.out, "app", "auto", 10, args.duration, scenario="rerender")
    rig.close()
    rr = s.get("rerender") or {}
    inside = sum(v for k, v in (rr.get("mutations") or {}).items() if k.startswith("INSIDE"))
    outside = sum(v for k, v in (rr.get("mutations") or {}).items() if k.startswith("OUTSIDE"))
    ri = sum(v for k, v in (rr.get("renders") or {}).items() if k.startswith("INSIDE"))
    ro = sum(v for k, v in (rr.get("renders") or {}).items() if not k.startswith("INSIDE"))
    s["rerender_summary"] = {
        "window_s": args.duration,
        "frames_painted": rr.get("paints"),
        "react_commits": rr.get("commits"),
        "component_renders_inside_camera_element": ri,
        "component_renders_outside_camera_element": ro,
        "dom_mutations_inside_camera_element": inside,
        "dom_mutations_outside_camera_element": outside,
        "top_outside_mutations": sorted(
            ((v, k) for k, v in (rr.get("mutations") or {}).items() if k.startswith("OUTSIDE")),
            reverse=True,
        )[:8],
        "top_outside_renders": sorted(
            ((v, k) for k, v in (rr.get("renders") or {}).items() if not k.startswith("INSIDE")),
            reverse=True,
        )[:8],
        "caveat": (
            "includes the app's own 1.2 s snapshot poll and 1 s metadata poll; "
            "not isolated to camera frames"
        ),
    }
    out["rerender"] = s


def first_after(rows, t):
    for r in rows:
        if r["t_paint_s"] > t and r.get("decode_ok"):
            return r
    return None


def last_before(rows, t):
    prev = None
    for r in rows:
        if r["t_paint_s"] > t:
            break
        if r.get("decode_ok"):
            prev = r
    return prev


def to_server(raw, co, page_ms):
    return (page_ms + co["offset_used_ms"]) / 1000


def scn_fallback(args, out):
    block_s, after_s, warm = 12, 20, 6
    results = {}
    for client in ("app", "raw"):
        rig = new_rig(args, 10, web_root=args.app_root if client == "app" else args.raw_root)
        rig.start(10)
        raw = run_driver(
            rig,
            args.out,
            f"fallback_{client}",
            client=client,
            mode="auto",
            scenario="block_push",
            warm_s=warm,
            block_s=block_s,
            after_s=after_s,
            timeout_s=240,
        )
        rows, co = join_paints(rig, raw)
        write_csv(args.out / f"fallback_{client}.csv", rows, CSV_COLS)
        act = next(a for a in raw["actions"] if a["name"] == "block")
        t_kill = to_server(raw, co, act["t0"])
        t_ret = t_kill + block_s
        before, after = last_before(rows, t_kill), first_after(rows, t_kill)
        recv = raw.get("recv", [])
        # first polled receive after the kill = the switch to polling
        poll_recv = [r for r in recv if r["mode"] == "poll" and to_server(raw, co, r["t"]) > t_kill]
        dom = [
            {"t": to_server(raw, co, d["t"]) - t_kill, "found": d["found"], "fps": d.get("fps")}
            for d in raw.get("dom", [])
        ]
        push_back = [d for d in dom if "MJPEG push" in d["found"] and d["t"] > 0.5]
        res = {
            "client": client,
            "slots_held_by_harness": act["resp"].get("slots_held"),
            "block_s": block_s,
            "last_paint_before_kill_to_kill_ms": None
            if before is None
            else round((t_kill - before["t_paint_s"]) * 1000, 1),
            "kill_to_next_painted_frame_s": None
            if after is None
            else round(after["t_paint_s"] - t_kill, 3),
            "kill_to_first_polled_receive_s": round(
                to_server(raw, co, poll_recv[0]["t"]) - t_kill, 3
            )
            if poll_recv
            else None,
            "frames_painted_during_block_window": sum(
                1 for r in rows if t_kill < r["t_paint_s"] < t_ret and r.get("decode_ok")
            ),
            "block_window_delivered_fps": None,
            "dom_label_transitions_s_since_kill": dom
            if client == "app"
            else "raw page has no labels",
            "page_events": [
                {**e, "t": round(to_server(raw, co, e["t"]) - t_kill, 3)}
                for e in raw.get("page_events", [])
            ]
            if client == "raw"
            else None,
            "clock_uncertainty_ms": co["uncertainty_ms"],
        }
        win = [r for r in rows if t_kill + 1 < r["t_paint_s"] < t_ret and r.get("decode_ok")]
        if len(win) > 1:
            res["block_window_delivered_fps"] = round(
                (len(win) - 1) / (win[-1]["t_paint_s"] - win[0]["t_paint_s"]), 2
            )
        # push return: first MJPEG-mode receive after the slots were released
        if client == "raw":
            back = [
                e
                for e in raw.get("page_events", [])
                if e["e"] == "stream_status"
                and e.get("status") == 200
                and to_server(raw, co, e["t"]) > t_kill + 1
            ]
            res["slots_released_to_push_resumed_s"] = (
                round(to_server(raw, co, back[0]["t"]) - t_ret, 3) if back else None
            )
            res["stream_status_sequence"] = [
                (round(to_server(raw, co, e["t"]) - t_kill, 2), e.get("status"))
                for e in raw.get("page_events", [])
                if e["e"] == "stream_status"
            ]
        else:
            res["slots_released_to_push_label_s"] = (
                round(push_back[0]["t"] - block_s, 3) if push_back else None
            )
        results[client] = res
        rig.close()
    out["fallback"] = results


def scn_reconnect(args, out):
    results = {}
    for client in ("app", "raw"):
        for kind, extra in (("drop", {}), ("swap", {}), ("restart", {"down_s": 2})):
            rig = new_rig(args, 10, web_root=args.app_root if client == "app" else args.raw_root)
            rig.start(10)
            raw = run_driver(
                rig,
                args.out,
                f"reconnect_{client}_{kind}",
                client=client,
                mode="auto",
                scenario=kind,
                warm_s=6,
                repeats=3,
                gap_s=14,
                timeout_s=240,
                **extra,
            )
            if "paints" not in raw:
                results[f"{client}_{kind}"] = {
                    "error": "no paints",
                    "driver_exit": raw.get("driver_exit"),
                }
                rig.close()
                continue
            rows, co = join_paints(rig, raw)
            write_csv(args.out / f"reconnect_{client}_{kind}.csv", rows, CSV_COLS)
            reps = []
            for a in [a for a in raw["actions"] if a["name"] == kind]:
                t0 = to_server(raw, co, a["t0"])
                t_done = to_server(raw, co, a["t1"])
                before, after = last_before(rows, t0), first_after(rows, t_done)
                reps.append(
                    {
                        "fault_to_next_painted_frame_s": None
                        if after is None
                        else round(after["t_paint_s"] - t0, 3),
                        "fault_call_returned_to_next_frame_s": None
                        if after is None
                        else round(after["t_paint_s"] - t_done, 3),
                        "picture_gap_s": None
                        if (after is None or before is None)
                        else round(after["t_paint_s"] - before["t_paint_s"], 3),
                        "fault_call_duration_s": round(t_done - t0, 3),
                    }
                )
            results[f"{client}_{kind}"] = {
                "repeats": reps,
                "clock_uncertainty_ms": co["uncertainty_ms"],
                "dom_labels": sorted({label for d in raw.get("dom", []) for label in d["found"]}),
            }
            rig.close()
    out["reconnect"] = results


def abrupt_close(sock):
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))  # close sends RST
    sock.close()


def open_stream(port, token):
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    s.sendall(
        f"GET /camera/stream HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {token}\r\n\r\n".encode()
    )
    return s, s.recv(64).split(b" ")[1]


def scn_leak(args, out):
    """Check that an abrupt disconnect returns the slot, with and without frames."""
    results = {}
    for label, source_on in (("frames_flowing", True), ("source_idle", False)):
        rig = new_rig(args, 10)
        if source_on:
            rig.start(10)
        time.sleep(1)
        reps = []
        for _ in range(5):
            a, ca = open_stream(rig.port, args.token)
            b, cb = open_stream(rig.port, args.token)
            c, cc = open_stream(rig.port, args.token)  # third must be refused
            c.close()
            if source_on:
                a.settimeout(2)
                got_frame = len(a.recv(4096)) > 100
            else:
                got_frame = None
            t0 = time.perf_counter()
            abrupt_close(a)
            abrupt_close(b)
            freed = None
            while time.perf_counter() - t0 < 10:
                x, y = [rig.api._media_slots.acquire(blocking=False) for _ in range(2)]
                if x:
                    rig.api._media_slots.release()
                if y:
                    rig.api._media_slots.release()
                if x and y:
                    freed = time.perf_counter() - t0
                    break
                time.sleep(0.01)
            s1, c1 = open_stream(rig.port, args.token)
            s2, c2 = open_stream(rig.port, args.token)
            s1.close()
            s2.close()
            t1 = time.perf_counter()
            while time.perf_counter() - t1 < 10:  # let a polite close finish before the next repeat
                got = [rig.api._media_slots.acquire(blocking=False) for _ in range(2)]
                for ok in got:
                    if ok:
                        rig.api._media_slots.release()
                if all(got):
                    break
                time.sleep(0.01)
            polite_free_s = time.perf_counter() - t1
            reps.append(
                {
                    "first_two": [ca.decode(), cb.decode()],
                    "third": cc.decode(),
                    "frame_seen_before_close": got_frame,
                    "slots_both_free_after_s": None if freed is None else round(freed, 3),
                    "reopen_two_after_abrupt_close": [c1.decode(), c2.decode()],
                    "polite_close_both_slots_free_after_s": round(polite_free_s, 3),
                }
            )
        results[label] = {
            "repeats": reps,
            "leaks": sum(
                1
                for r in reps
                if r["slots_both_free_after_s"] is None
                or r["reopen_two_after_abrupt_close"] != ["200", "200"]
            ),
        }
        rig.close()
    # a real browser killed abruptly (taskkill-equivalent SIGKILL) while streaming
    chrome = []
    for i in range(5):
        rig = new_rig(args, 10, web_root=args.raw_root)
        rig.start(10)
        run_driver(
            rig,
            args.out,
            f"leak_chrome_kill_{i}",
            client="raw",
            mode="push",
            scenario="crash",
            warm_s=3,
            timeout_s=120,
        )
        time.sleep(2.0)
        ev = {e["name"]: e["t"] for e in rig.events}
        mark, free = ev.get("chrome_kill"), ev.get("chrome_kill_slots_free")
        chrome.append(
            {
                "mark_to_both_slots_free_s": None
                if (mark is None or free is None)
                else round(free - mark, 3)
            }
        )
        rig.close()
    results["chrome_sigkill"] = {
        "repeats": chrome,
        "leaks": sum(1 for r in chrome if r["mark_to_both_slots_free_s"] is None),
    }
    out["slot_leak"] = results


def scn_stale(args, out):
    rig = new_rig(args, 10, web_root=args.app_root)
    rig.start(10)
    raw = run_driver(
        rig,
        args.out,
        "stale_app",
        client="app",
        mode="auto",
        scenario="stale",
        warm_s=6,
        stale_s=14,
        after_s=8,
        timeout_s=200,
    )
    rows, co = join_paints(rig, raw)
    write_csv(args.out / "stale_app.csv", rows, CSV_COLS)
    stop = next(a for a in raw["actions"] if a["name"] == "source" and "run=0" in a["query"])
    resume = next(a for a in raw["actions"] if a["name"] == "source" and "run=1" in a["query"])
    t_stop_server = stop["resp"]["t"]
    t_resume = resume["resp"]["t"]
    last_src = max(
        k for k, v in rig.src.items() if v <= t_stop_server
    )  # last stamped frame before stop
    dom = [
        {
            "t": round(to_server(raw, co, d["t"]) - t_stop_server, 3),
            "found": d["found"],
            "fps": d.get("fps"),
        }
        for d in raw.get("dom", [])
    ]
    during = [
        r for r in rows if t_stop_server + 0.5 < r["t_paint_s"] < t_resume and r.get("decode_ok")
    ]
    last_before_stop = last_before(rows, t_stop_server)
    ind = [d for d in dom if "Simulation paused or not sensing" in d["found"] and d["t"] > 0]
    live_after = [
        d
        for d in dom
        if d["t"] > t_resume - t_stop_server
        and not (set(d["found"]) & {"Simulation paused or not sensing"})
    ]
    first_new = first_after(rows, t_resume)
    out["stale"] = {
        "client": "real app",
        "source_stopped_at": "t=0",
        "stale_window_s": round(t_resume - t_stop_server, 2),
        "last_painted_frame_before_stop_to_stop_ms": None
        if last_before_stop is None
        else round((t_stop_server - last_before_stop["t_paint_s"]) * 1000, 1),
        "stop_to_indicator_s": ind[0]["t"] if ind else None,
        "indicator_text": "Simulation paused or not sensing",
        "new_frames_painted_during_stale_window": len(during),
        "fps_label_values_seen_during_window": [
            d["fps"] for d in dom if 0 < d["t"] < t_resume - t_stop_server
        ],
        "dom_transitions": dom,
        "age_readout_every_500ms_s_since_stop": [
            (round(to_server(raw, co, a["t"]) - t_stop_server, 1), a["age"], a["fps"])
            for a in raw.get("age", [])
        ][::2],
        "resume_to_first_new_frame_s": None
        if first_new is None
        else round(first_new["t_paint_s"] - t_resume, 3),
        "resume_to_indicator_cleared_s": round(live_after[0]["t"] - (t_resume - t_stop_server), 3)
        if live_after
        else None,
        "clock_uncertainty_ms": co["uncertainty_ms"],
        "last_source_seq_before_stop": last_src,
    }
    rig.close()


def scn_baseline_cpu(args, out):
    """Server-side cost with no client attached, per source rate."""
    res = []
    for rate in (5, 10, 15):
        rig = new_rig(args, rate)
        rig.start(rate)
        time.sleep(1)
        c0, w0 = time.process_time(), time.perf_counter()
        time.sleep(15)
        c1, w1 = time.process_time(), time.perf_counter()
        res.append(
            {
                "source_fps": rate,
                "server_process_cpu_pct_of_one_core_no_client": round(
                    100 * (c1 - c0) / (w1 - w0), 1
                ),
                "includes": (
                    "numpy frame synthesis + stamp + JPEG encode (Pillow q80) in this process"
                ),
                "encode_ms": stats([v[1] for v in rig.enc.values()]),
            }
        )
        rig.close()
    out["baseline_cpu_no_client"] = res


def git_sha():
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()


def machine():
    import multiprocessing

    return {
        "platform": platform.platform(),
        "cpu_count": multiprocessing.cpu_count(),
        "python": sys.version.split()[0],
        "node": subprocess.run(
            ["node", "--version"], capture_output=True, text=True
        ).stdout.strip(),
        "processor": platform.processor(),
    }


SCENARIOS = {
    "auth": scn_auth,
    "latency": scn_latency,
    "app_poll": scn_app_poll_only,
    "rerender": scn_rerender,
    "fallback": scn_fallback,
    "reconnect": scn_reconnect,
    "leak": scn_leak,
    "stale": scn_stale,
    "baseline": scn_baseline_cpu,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", default=",".join(SCENARIOS))
    ap.add_argument("--duration", type=float, default=25)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.token = secrets.token_urlsafe(32)  # minted here; never read from the phone token file
    args.app_root = APP_ROOT
    args.raw_root = Path(tempfile.mkdtemp(prefix="dream_camera_raw_"))
    parser_src = (ROOT / "web/app/camera-client.mjs").read_text().replace("export class", "class")
    (args.raw_root / "index.html").write_text(
        (ROOT / "experiments/camera_latency_raw.html")
        .read_text()
        .replace("/*PARSER*/", parser_src),
        encoding="utf-8",
    )
    for name in args.only.split(","):
        out = {}
        t0 = time.time()
        try:
            SCENARIOS[name](args, out)
        except Exception:  # report, do not hide
            import traceback

            out["error"] = traceback.format_exc()
        out["elapsed_s"] = round(time.time() - t0, 1)
        (args.out / f"{name}.json").write_text(json.dumps(out, indent=2, default=str))
        print("done", name, out.get("error", "ok")[:200] if "error" in out else "ok", flush=True)
    meta = {
        "git_sha": git_sha(),
        "machine": machine(),
        "chrome": CHROME,
        "network": "LOOPBACK 127.0.0.1 only; not a tailnet, WAN or IAP measurement",
        "clock": (
            "single machine; browser epoch clock mapped to the server clock by "
            "round-trip handshake (NTP-style, min-RTT sample)"
        ),
        "frame": {"height": H, "width": W, "source": "HEAD_CAMERA in dream/sim/g1_mobile.py"},
        "app_build": str(APP_ROOT),
        "conditions": "headless Chrome, other MuJoCo jobs may share this laptop",
        "run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
