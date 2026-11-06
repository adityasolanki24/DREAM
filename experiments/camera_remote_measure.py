"""Measure live camera delivery from this laptop to a real Isaac session over IAP tunnels.

    python experiments/camera_remote_measure.py --manifest <manifest.json> --out <dir>

Run while an `isaac_g1_live.py --media` job is serving on the session VM (see
cloud/isaac_session.py submit). It opens two IAP tunnels (API port and the MediaMTX ICE/TCP
port 18189), pairs through the one-time /_bootstrap route with a token minted here (kept in
memory, handed to the browser driver through its environment, never printed or written),
submits a Qwen-controlled task through the real phone API so the simulator ticks, and then
measures each transport in its own window with headless Chrome running the web client's
logic (experiments/camera_remote_page.html): JPEG polling, MJPEG push, WHEP, and the
push-with-polling-fallback policy through a tunnel outage.

Everything reported is a client-side measurement over IAP. Clock mapping between the VM and
this laptop is a round-trip handshake whose uncertainty is half the best round trip, which
over IAP is large; it is reported next to every latency. Capture rate, NVENC and MediaMTX
numbers come from the VM's own evidence, not from here.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import secrets
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHROME = os.environ.get("DREAM_CHROME", r"C:\Program Files\Google\Chrome\Application\chrome.exe")
NODE = shutil.which("node") or "node"
PLAYWRIGHT = os.environ.get(
    "DREAM_PLAYWRIGHT_CORE", r"C:\Users\24adi\OneDrive\DREAM\web\node_modules\playwright-core"
)
API_LOCAL, ICE_LOCAL = (
    18765,
    18189,
)  # the ICE/TCP candidate the server advertises is 127.0.0.1:18189


def stats(values):
    v = sorted(values)
    if not v:
        return {"n": 0}
    p95 = v[min(len(v) - 1, int(round(0.95 * (len(v) - 1))))]
    return {
        "n": len(v),
        "min": round(v[0], 1),
        "median": round(statistics.median(v), 1),
        "mean": round(statistics.fmean(v), 1),
        "p95": round(p95, 1),
        "max": round(v[-1], 1),
    }


# -- pure analysis (unit-tested) ---------------------------------------------------------------


def clock_offset(sync: list[dict]) -> dict:
    """VM clock minus page clock, in seconds, from the minimum-round-trip handshake. The VM
    timestamp is taken somewhere inside the round trip, so the error is at most half of it."""
    rows = [s for s in sync if s.get("server_unix_s") is not None]
    if not rows:
        return {"offset_s": None, "uncertainty_ms": None, "samples": 0}
    best = min(rows, key=lambda s: s["t1"] - s["t0"])
    rtt = best["t1"] - best["t0"]
    mid = (best["t0"] + best["t1"]) / 2000.0
    return {
        "offset_s": best["server_unix_s"] - mid,
        "uncertainty_ms": rtt / 2,
        "min_rtt_ms": rtt,
        "median_rtt_ms": statistics.median(s["t1"] - s["t0"] for s in rows),
        "samples": len(rows),
    }


def jpeg_summary(result: dict, window_s: float) -> dict:
    """Delivered rate, capture-to-screen latency, bytes, sequence gaps for a push/poll run."""
    shown = [s for s in result.get("shown", []) if s.get("captured")]
    off = clock_offset(result.get("sync", []))
    out = {"clock": off, "frames_shown": len(shown), "window_s": window_s}
    if not shown:
        return out
    o = off["offset_s"] or 0.0
    first, last = min(s["t_shown"] for s in shown), max(s["t_shown"] for s in shown)
    span = max((last - first) / 1000.0, 1e-6)
    seqs = sorted({s["seq"] for s in shown})
    out.update(
        delivered_fps_over_window=round(len(shown) / window_s, 2),
        delivered_fps_while_frames_flowing=round((len(shown) - 1) / span, 2),
        first_to_last_s=round(span, 1),
        capture_to_receive_ms=stats(
            [(s["t_recv"] / 1000 + o - s["captured"]) * 1000 for s in shown]
        ),
        capture_to_screen_ms=stats(
            [(s["t_shown"] / 1000 + o - s["captured"]) * 1000 for s in shown]
        ),
        client_decode_to_screen_ms=stats([s["t_shown"] - s["t_recv"] for s in shown]),
        server_residence_ms=stats(
            [(s["server"] - s["captured"]) * 1000 for s in shown if s["server"]]
        ),
        bytes_per_frame=stats([s["bytes"] for s in shown]),
        mean_kbit_s_while_flowing=round(sum(s["bytes"] for s in shown) * 8 / span / 1000, 1),
        sequence_span=seqs[-1] - seqs[0] + 1,
        sequence_frames_never_shown=(seqs[-1] - seqs[0] + 1) - len(seqs),
        modes=sorted({s["mode"] for s in shown}),
    )
    return out


def whep_summary(result: dict, window_s: float) -> dict:
    rows = [r for r in result.get("stats", []) if r.get("inbound")]
    off = clock_offset(result.get("sync", []))
    events = result.get("events", [])
    out = {
        "clock": off,
        "connection_events": [
            {k: v for k, v in e.items() if k != "status"}
            for e in events
            if "state" in e or "status" in e
        ],
        "answer_lines": result.get("answer"),
        "stats_rows": len(result.get("stats", [])),
        "video_frame_callbacks": len(result.get("video", [])),
    }
    if rows:
        a, b = rows[0], rows[-1]
        dt = max((b["t"] - a["t"]) / 1000.0, 1e-6)
        ia, ib = a["inbound"], b["inbound"]
        out.update(
            decoded_fps=round((ib["framesDecoded"] - ia["framesDecoded"]) / dt, 2),
            frames_decoded=ib["framesDecoded"],
            frames_dropped=ib.get("framesDropped"),
            packets_lost=ib.get("packetsLost"),
            freeze_count=ib.get("freezeCount"),
            bytes_received=ib.get("bytesReceived"),
            mean_jitter_buffer_delay_ms=(
                None
                if not ib.get("jitterBufferEmittedCount")
                else round(ib["jitterBufferDelay"] / ib["jitterBufferEmittedCount"] * 1000, 1)
            ),
            mean_decode_ms=(
                None
                if not ib.get("framesDecoded")
                else round(ib["totalDecodeTime"] / ib["framesDecoded"] * 1000, 2)
            ),
            frame_size=[ib.get("frameWidth"), ib.get("frameHeight")],
            round_trip_ms=[r["pair"].get("currentRoundTripTime") for r in rows if r.get("pair")][
                -1:
            ],
        )
        o = off["offset_s"]
        playouts = [
            (r["t"] / 1000 + o) * 1000 - r["inbound"]["estimatedPlayoutTimestamp"]
            for r in rows
            if o is not None and r["inbound"].get("estimatedPlayoutTimestamp")
        ]
        out["estimated_playout_lag_ms"] = stats(playouts) if playouts else {"n": 0}
        out["estimated_playout_note"] = (
            "page time mapped to the VM clock minus RTCP-derived estimatedPlayoutTimestamp; "
            "only meaningful if MediaMTX forwards the publisher's NTP mapping"
        )
    cb = result.get("video", [])
    if len(cb) > 1:
        span = (cb[-1]["t"] - cb[0]["t"]) / 1000
        out["presented_fps"] = round((len(cb) - 1) / max(span, 1e-6), 2)
        recv = [c["t"] - c["receiveTime"] for c in cb if c.get("receiveTime")]
        out["frame_callback_minus_receive_ms"] = stats(recv) if recv else {"n": 0}
    out["window_s"] = window_s
    return out


# -- tunnels and API ---------------------------------------------------------------------------


def listening(port: int) -> bool:
    try:
        socket.create_connection(("127.0.0.1", port), timeout=1).close()
        return True
    except OSError:
        return False


class Tunnel:
    def __init__(self, manifest: dict, remote: int, local: int, log: pathlib.Path):
        self.args = [
            shutil.which("gcloud.cmd") or shutil.which("gcloud"),
            "compute", "start-iap-tunnel", manifest["name"], str(remote),
            f"--local-host-port=localhost:{local}", f"--zone={manifest['zone']}",
            f"--project={manifest['project']}",
        ]  # fmt: skip
        self.local, self.log, self.proc = local, log, None

    def start(self, timeout_s=60) -> float:
        t0 = time.monotonic()
        self.proc = subprocess.Popen(
            self.args, stdout=self.log.open("a"), stderr=subprocess.STDOUT, text=True
        )
        while time.monotonic() - t0 < timeout_s:
            if listening(self.local):
                return time.monotonic() - t0
            if self.proc.poll() is not None:
                raise RuntimeError("IAP tunnel exited; see " + str(self.log))
            time.sleep(0.5)
        raise TimeoutError("IAP tunnel did not listen")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            if os.name == "nt":
                # gcloud.cmd spawns the Python tunnel as a child; terminate() would leave it
                # serving (the first outage test never cut the connection).
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(self.proc.pid)], capture_output=True
                )
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def start_when_serving(tunnel: Tunnel, timeout_s: float) -> float:
    """The VM port only opens once the live job (and its media runtime) is serving; IAP refuses
    the tunnel's connectivity check until then (error 4003), so retry."""
    t0 = time.monotonic()
    while True:
        try:
            tunnel.start()
            return time.monotonic() - t0
        except RuntimeError:
            if time.monotonic() - t0 > timeout_s:
                raise
            time.sleep(15)


class Api:
    def __init__(self, port: int, token: str):
        self.base, self._token = f"http://127.0.0.1:{port}", token

    def call(self, method, path, body=None, timeout=30.0, auth=True):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, method=method)
        if auth:
            req.add_header("Authorization", f"Bearer {self._token}")
        if data:
            req.add_header("Content-Type", "application/json")
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, json.loads(r.read() or b"null"), (time.perf_counter() - t0) * 1000
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"null"), (time.perf_counter() - t0) * 1000


def wait_pair(api: Api, token: str, timeout_s: float) -> dict:
    deadline, attempts = time.monotonic() + timeout_s, 0
    while time.monotonic() < deadline:
        attempts += 1
        try:
            status, body, ms = api.call("POST", "/_bootstrap", {"token": token}, 20, auth=False)
            if status == 200:
                return {"paired": True, "attempts": attempts, "ms": round(ms)}
            if status == 403:
                return {"paired": False, "reason": "already paired by another client"}
        except (OSError, ValueError):
            pass
        time.sleep(10)
    return {"paired": False, "reason": "timeout", "attempts": attempts}


def ensure_task(api: Api, log: list, text: str) -> None:
    _, tasks, _ = api.call("GET", "/tasks")
    if any(t["state"] in ("running", "queued", "pending", "suspended") for t in tasks):
        return
    status, body, _ = api.call("POST", "/tasks", {"text": text, "provider": "qwen"})
    log.append({"t": time.time(), "submitted": text, "status": status, "n_previous": len(tasks)})


def run_page(mode, seconds, out_dir, token, page_html, extra=None, during=None) -> dict:
    cfg = {"chrome": CHROME, "page": str(page_html), "mode": mode, "seconds": seconds,
           "base": f"http://127.0.0.1:{API_LOCAL}",
           "out": str(out_dir / f"{mode}.json")}  # fmt: skip
    cfg.update(extra or {})
    cfg_path = out_dir / f"{mode}_cfg.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    env = {**os.environ, "DREAM_PLAYWRIGHT_CORE": PLAYWRIGHT, "DREAM_MEASURE_TOKEN": token}
    proc = subprocess.Popen(
        [NODE, str(ROOT / "experiments/camera_remote_driver.cjs"), str(cfg_path)],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )  # fmt: skip
    marks = during(proc) if during else None
    stdout, _ = proc.communicate(timeout=seconds + 240)
    data = json.loads(pathlib.Path(cfg["out"]).read_text(encoding="utf-8"))
    data["driver_stdout"] = stdout[-600:]
    data["marks"] = marks
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--pair-timeout-s", type=float, default=2700)
    ap.add_argument("--phase-s", type=float, default=45)
    ap.add_argument("--auto-s", type=float, default=75)
    ap.add_argument("--task", default="Inspect the cabinet")
    ap.add_argument("--only", default="poll,push,whep,auto")
    args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    parser_src = (ROOT / "web/app/camera-client.mjs").read_text().replace("export class", "class")
    work = pathlib.Path(tempfile.mkdtemp(prefix="dream_remote_cam_"))
    page = work / "index.html"
    page.write_text(
        (ROOT / "experiments/camera_remote_page.html")
        .read_text(encoding="utf-8")
        .replace("/*PARSER*/", parser_src),
        encoding="utf-8",
    )
    tunnel_log = out / "iap_tunnels.log"
    api_tunnel, ice_tunnel = (
        Tunnel(manifest, 8765, API_LOCAL, tunnel_log),
        Tunnel(manifest, 18189, ICE_LOCAL, tunnel_log),
    )
    report = {
        "manifest_name": manifest["name"],
        "started_utc": time.strftime("%FT%TZ", time.gmtime()),
    }
    submitted: list = []
    try:
        report["api_tunnel_up_s"] = round(start_when_serving(api_tunnel, args.pair_timeout_s), 1)
        report["ice_tunnel_up_s"] = round(start_when_serving(ice_tunnel, args.pair_timeout_s), 1)
        api = Api(API_LOCAL, token)
        report["pairing"] = wait_pair(api, token, args.pair_timeout_s)
        (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        if not report["pairing"]["paired"]:
            return
        _, status, _ = api.call("GET", "/status")
        _, media0, _ = api.call("GET", "/media/status")
        _, meta0, _ = api.call("GET", "/camera/meta")
        report["status_at_pairing"] = {"status": status, "media": media0, "camera_meta": meta0}
        # Idle round trips through IAP (keep-alive is not used by urllib: each is a new connection).
        report["api_rtt_new_connection_ms"] = stats(
            [api.call("GET", "/status")[2] for _ in range(15)]
        )
        ensure_task(api, submitted, args.task)
        phases = {}
        for name in args.only.split(","):
            ensure_task(api, submitted, args.task)
            before = {
                "media": api.call("GET", "/media/status")[1],
                "meta": api.call("GET", "/camera/meta")[1],
            }
            if name == "auto":
                seconds = args.auto_s

                def outage(proc):
                    time.sleep(25)
                    t_kill = time.time()
                    api_tunnel.stop()
                    time.sleep(12)
                    t_back = time.time()
                    up = api_tunnel.start()
                    return {"kill_unix_s": t_kill, "restart_unix_s": t_back,
                            "tunnel_listening_after_s": round(up, 1),
                            "restored_unix_s": time.time()}  # fmt: skip

                data = run_page("auto", seconds, out, token, page, during=outage)
            else:
                seconds = args.phase_s
                data = run_page(name, seconds, out, token, page)
            after = {
                "media": api.call("GET", "/media/status")[1],
                "meta": api.call("GET", "/camera/meta")[1],
            }
            res = data.get("result") or {}
            summary = whep_summary(res, seconds) if name == "whep" else jpeg_summary(res, seconds)
            phases[name] = {"summary": summary, "server_before": before, "server_after": after,
                            "run_error": data.get("run_error"), "console": data.get("console"),
                            "failed_requests": data.get("failed_requests"),
                            "marks": data.get("marks"),
                            "chrome": data.get("chrome_version")}  # fmt: skip
            (out / f"{name}_summary.json").write_text(
                json.dumps(phases[name], indent=2, default=str)
            )
            report["phases"] = phases
            (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
        report["tasks_submitted"] = submitted
        report["final_tasks"] = api.call("GET", "/tasks")[1]
        report["final_media"] = api.call("GET", "/media/status")[1]
        report["final_camera_meta"] = api.call("GET", "/camera/meta")[1]
    finally:
        api_tunnel.stop()
        ice_tunnel.stop()
        report["ended_utc"] = time.strftime("%FT%TZ", time.gmtime())
        (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
        shutil.rmtree(work, ignore_errors=True)
    print(
        json.dumps({k: report.get(k) for k in ("pairing", "api_rtt_new_connection_ms")}, indent=2)
    )


if __name__ == "__main__":
    sys.exit(main())
