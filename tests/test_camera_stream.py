"""Authenticated bounded image delivery, independent of simulator and network speed."""

import io
import threading
import time
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest
from PIL import Image

from dream.runtime.camera_stream import CameraStream
from dream.runtime.phone_api import PhoneAPI


def frame(t=1):
    return SimpleNamespace(time=t, rgb=np.zeros((240, 320, 3), np.uint8))


def wait_for(test):
    deadline = time.monotonic() + 3
    while not test():
        assert time.monotonic() < deadline
        time.sleep(0.005)


def test_same_frame_is_not_reencoded_and_metadata_is_light():
    stream = CameraStream()
    f = frame()
    stream.submit(f)
    wait_for(lambda: stream.latest() is not None)
    first = stream.latest()
    stream.submit(f)
    time.sleep(0.03)
    assert stream.latest() is first
    meta = stream.metadata()
    assert meta["encoded_frames"] == 1 and "data_url" not in str(meta)
    assert Image.open(io.BytesIO(first.jpeg)).size == (320, 240)
    stream.close()


def test_slow_encoder_discards_pending_frames_without_blocking_producer():
    gate = threading.Event()

    def encode(rgb):
        gate.wait(1)
        return b"jpeg"

    stream = CameraStream(encode=encode)
    stream.submit(frame(1))
    wait_for(lambda: stream.metadata()["submitted_frames"] == 1)
    start = time.monotonic()
    for t in range(2, 100):
        stream.submit(frame(t))
    assert time.monotonic() - start < 0.2
    gate.set()
    wait_for(lambda: stream.latest() is not None and stream.latest().sim_time_s == 99)
    assert stream.metadata()["dropped_frames"] > 0
    stream.close()


def test_auth_changed_frame_and_reconnect():
    stream = CameraStream()
    stream.submit(frame())
    wait_for(lambda: stream.latest() is not None)
    api = PhoneAPI(SimpleNamespace(running=None), "t" * 32, camera=stream)
    assert api.handle("GET", "/camera/meta", None, b"")[0] == 401
    auth = "Bearer " + "t" * 32
    status, out = api.handle("GET", "/camera/frame", auth, b"")
    assert status == 200 and out[0] == "image/jpeg"
    current = stream.latest().id
    assert api.handle("GET", "/camera/frame?after=" + current, auth, b"")[0] == 204
    stream.submit(frame(2))
    wait_for(lambda: stream.latest().id != current)
    assert api.handle("GET", "/camera/frame?after=" + current, auth, b"")[0] == 200
    # A reconnected client may fetch the current latest frame, never an old queue.
    assert api.handle("GET", "/camera/frame", auth, b"")[1][1] == stream.latest().jpeg
    server = api.serve(port=0)
    try:
        url = f"http://127.0.0.1:{server.server_port}/camera/frame"
        with urlopen(Request(url, headers={"Authorization": auth})) as r:
            assert r.headers["Content-Type"] == "image/jpeg"
            assert r.headers["X-Frame-Id"] == stream.latest().id
        with pytest.raises(HTTPError):
            urlopen(url)
    finally:
        server.shutdown()
        server.server_close()
        stream.close()
