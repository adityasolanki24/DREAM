"""Media security and latency contracts without requiring a paid GPU."""

from dream.runtime.media_runtime import MediaRuntime, encoder_command, media_config


def test_media_listeners_and_auth_do_not_expose_robot_controls():
    cfg = media_config()
    assert cfg["rtspAddress"].startswith("127.0.0.1:")
    assert cfg["webrtcAddress"].startswith("127.0.0.1:")
    assert not cfg["webrtcLocalUDPAddress"]
    assert cfg["webrtcLocalTCPAddress"] == ":18189"
    assert all(user["ips"] == ["127.0.0.1", "::1"] for user in cfg["authInternalUsers"])
    assert not cfg["api"] and not cfg["rtmp"] and not cfg["hls"]


def test_nvenc_is_low_latency_and_recorded_without_scripted_frames(tmp_path):
    cmd = encoder_command(640, 360, tmp_path / "camera.mkv")
    assert "h264_nvenc" in cmd
    assert cmd[cmd.index("-bf") + 1] == "0"
    assert cmd[cmd.index("-tune") + 1] == "ull"
    assert "127.0.0.1:8554/dream" in cmd[-1]


def test_proxy_cannot_fetch_arbitrary_destinations():
    media = MediaRuntime.__new__(MediaRuntime)
    assert media.proxy("GET", "/media/whep/http://example.com", b"")[0] == 404


def test_raw_sdp_signalling_requires_bearer_auth_before_proxy():
    from types import SimpleNamespace

    from dream.runtime.phone_api import PhoneAPI

    seen = []
    media = SimpleNamespace(
        proxy=lambda *args: seen.append(args) or (201, ("application/sdp", b"answer", {}))
    )
    api = PhoneAPI(None, "t" * 32, media=media)
    assert api.handle("POST", "/media/whep", None, b"v=0")[0] == 401
    assert seen == []
    assert api.handle("POST", "/media/whep", "Bearer " + "t" * 32, b"v=0")[0] == 201
    assert seen == [("POST", "/media/whep", b"v=0")]
