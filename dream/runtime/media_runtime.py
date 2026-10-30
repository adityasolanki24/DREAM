"""Optional NVENC/MediaMTX publisher, separate from physics and control.

MediaMTX v1.21.3 (MIT) is checksum-pinned outside Git. Signalling is proxied
through the bearer-authenticated PhoneAPI. The private IAP deployment uses
ICE/TCP on port 18189; this is a congestion-sensitive fallback, not UDP WebRTC.
Only the simulation owner captures RGB. Both encoder queues are latest-only.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import shutil
import subprocess
import tarfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


def media_config():
    return {
        "logLevel": "info",
        "api": False,
        "metrics": False,
        "pprof": False,
        "playback": False,
        "rtmp": False,
        "hls": False,
        "srt": False,
        "moq": False,
        "rtsp": True,
        "rtspAddress": "127.0.0.1:8554",
        "rtspTransports": ["tcp"],
        "webrtc": True,
        "webrtcAddress": "127.0.0.1:8889",
        "webrtcAllowOrigins": ["http://127.0.0.1:3000", "http://localhost:3000"],
        "webrtcLocalUDPAddress": "",
        "webrtcLocalTCPAddress": ":18189",
        "webrtcIPsFromInterfaces": False,
        "webrtcAdditionalHosts": ["127.0.0.1"],
        "webrtcICEServers2": [],
        "authInternalUsers": [
            {
                "user": "any",
                "pass": "",
                "ips": ["127.0.0.1", "::1"],
                "permissions": [
                    {"action": "publish", "path": "dream"},
                    {"action": "read", "path": "dream"},
                ],
            }
        ],
        "paths": {"dream": {"source": "publisher"}},
    }


def encoder_command(width, height, recording, *, codec="h264_nvenc", modern_ffmpeg=False):
    """Software codec is an explicitly labelled local plumbing test only."""
    codec_args = (
        ["-preset", "p1", "-tune", "ull", "-rc", "cbr"]
        if codec == "h264_nvenc"
        else ["-preset", "ultrafast", "-tune", "zerolatency"]
    )
    return [
        shutil.which("ffmpeg") or "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-use_wallclock_as_timestamps",
        "1",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        "10",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        codec,
        *codec_args,
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "baseline",
        "-bf",
        "0",
        "-g",
        "10",
        "-b:v",
        "1000k",
        "-maxrate",
        "1000k",
        "-bufsize",
        "100k",
        *(["-fps_mode", "passthrough"] if modern_ffmpeg else ["-vsync", "0"]),
        "-flags:v",
        "+global_header",
        "-map",
        "0:v",
        "-f",
        "tee",
        f"[f=rtsp:rtsp_transport=tcp]rtsp://127.0.0.1:8554/dream|[f=matroska]{Path(recording).as_posix()}",
    ]


def install_binary(lock_path, cache):
    """Fetch one pinned upstream executable; no weights/binaries enter the repository."""
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    asset = lock["archives"][platform.system()]
    root = Path(cache) / lock["version"]
    root.mkdir(parents=True, exist_ok=True)
    archive = root / asset["name"]
    if not archive.is_file():
        url = f"https://github.com/bluenviron/mediamtx/releases/download/{lock['version']}/{asset['name']}"
        with urllib.request.urlopen(url, timeout=60) as response, archive.open("wb") as stream:
            shutil.copyfileobj(response, stream)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != asset["sha256"]:
        raise ValueError("MediaMTX archive checksum mismatch")
    name = "mediamtx.exe" if platform.system() == "Windows" else "mediamtx"
    binary = root / name
    # Extract only the known executable, never arbitrary archive paths or links.
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as bundle:
            binary.write_bytes(bundle.read(name))
    else:
        with tarfile.open(archive) as bundle:
            member = bundle.getmember(name)
            if not member.isfile():
                raise ValueError("MediaMTX binary is not a regular file")
            binary.write_bytes(bundle.extractfile(member).read())
        binary.chmod(0o755)
    return binary, lock


class MediaRuntime:
    def __init__(self, out, binary, *, codec="h264_nvenc"):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.codec = codec
        self.ffmpeg_version = subprocess.check_output(
            [shutil.which("ffmpeg") or "ffmpeg", "-version"], text=True
        ).splitlines()[0]
        self.modern_ffmpeg = int(re.search(r"version (\d+)", self.ffmpeg_version)[1]) >= 5
        self._condition = threading.Condition()
        self._pending = None
        self._closed = False
        self._encoder = None
        self._error = None
        self._submitted = self._written = self._dropped = 0
        self._last = None
        self._started = time.monotonic()
        config = self.out / "mediamtx.json"
        config.write_text(json.dumps(media_config(), indent=2), encoding="utf-8")
        self._router_log = (self.out / "mediamtx.log").open("wb")
        self._encoder_log = (self.out / "nvenc.log").open("wb")
        self._router = subprocess.Popen(
            [str(binary), str(config)], stdout=self._router_log, stderr=subprocess.STDOUT
        )
        self._thread = threading.Thread(target=self._run, daemon=True, name="nvenc-publisher")
        self._thread.start()

    def submit(self, rgb, frame):
        with self._condition:
            if self._closed:
                return
            self._submitted += 1
            self._dropped += self._pending is not None
            self._pending = (rgb, frame)
            self._condition.notify_all()

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._pending is not None or self._closed)
                if self._closed:
                    return
                rgb, frame = self._pending
                self._pending = None
            try:
                if self._router.poll() is not None:
                    raise RuntimeError("MediaMTX exited; see mediamtx.log")
                if self._encoder is None:
                    height, width = rgb.shape[:2]
                    cmd = encoder_command(
                        width,
                        height,
                        self.out / "head-camera.mkv",
                        codec=self.codec,
                        modern_ffmpeg=self.modern_ffmpeg,
                    )
                    (self.out / "encoder_command.json").write_text(
                        json.dumps(cmd), encoding="utf-8"
                    )
                    self._encoder = subprocess.Popen(
                        cmd,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=self._encoder_log,
                    )
                start = time.monotonic()
                self._encoder.stdin.write(rgb.tobytes())
                self._encoder.stdin.flush()
                self._written += 1
                self._last = {
                    "frame_id": frame.id,
                    "sim_time_s": frame.sim_time_s,
                    "captured_unix_s": frame.captured_unix_s,
                    "pipe_write_ms": (time.monotonic() - start) * 1000,
                }
            except Exception as error:
                self._error = str(error)
                return  # Do not silently restart/drop a failed recording.

    def status(self):
        return {
            "available": self._written > 0
            and self._error is None
            and self._router.poll() is None
            and self._encoder is not None
            and self._encoder.poll() is None,
            "codec": self.codec,
            "transport": "WebRTC ICE/TCP through private IAP (congestion-sensitive fallback)",
            "whep_url": "/media/whep",
            "ice_tcp_port": 18189,
            "submitted_frames": self._submitted,
            "written_frames": self._written,
            "dropped_frames": self._dropped,
            "average_fps": self._written / max(time.monotonic() - self._started, 0.001),
            "latest": self._last,
            "error": self._error,
            "closed": self._closed,
        }

    def proxy(self, method, path, body):
        suffix = path.removeprefix("/media/whep")
        if (
            not path.startswith("/media/whep")
            or not re.fullmatch(r"(?:/[a-fA-F0-9-]{1,80})?", suffix)
            or method not in ("POST", "DELETE")
        ):
            return 404, {"error": "unknown media operation"}
        url = "http://127.0.0.1:8889/dream/whep" + suffix
        request = urllib.request.Request(
            url,
            data=body if method == "POST" else None,
            method=method,
            headers={"Content-Type": "application/sdp"},
        )
        try:
            with urllib.request.urlopen(request, timeout=8) as response:
                headers = {}
                location = response.headers.get("Location")
                if location:
                    # MediaMTX's session location never exposes its loopback server.
                    session = location.rstrip("/").split("/")[-1]
                    if re.fullmatch(r"[a-fA-F0-9-]{1,80}", session):
                        headers["Location"] = "/media/whep/" + session
                return response.status, (
                    response.headers.get_content_type(),
                    response.read(),
                    headers,
                )
        except urllib.error.HTTPError as error:
            return error.code, {"error": "media negotiation failed", "status": error.code}
        except (OSError, TimeoutError):
            return 503, {"error": "media publisher unavailable; use JPEG fallback"}

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        # A blocked pipe is only in this worker. Termination releases it before joining.
        self._thread.join(timeout=2)
        if self._encoder is not None:
            try:
                if self._thread.is_alive():
                    self._encoder.terminate()
                else:
                    self._encoder.stdin.close()
                self._encoder.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                self._encoder.kill()
                self._encoder.wait(timeout=3)
        self._thread.join(timeout=2)
        self._router.terminate()
        try:
            self._router.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self._router.kill()
            self._router.wait(timeout=3)
        self._router_log.close()
        self._encoder_log.close()
