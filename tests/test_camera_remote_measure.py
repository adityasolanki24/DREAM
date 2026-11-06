"""Analysis helpers of the remote (IAP) camera measurement; no network, no browser."""

from experiments import camera_remote_measure as m


def test_clock_offset_uses_the_minimum_round_trip_and_reports_its_uncertainty():
    sync = [
        {"t0": 1_000_000, "t1": 1_002_000, "server_unix_s": 1005.0},  # rtt 2000 ms
        {"t0": 2_000_000, "t1": 2_000_400, "server_unix_s": 2005.2},  # rtt 400 ms, offset 5.0
    ]
    off = m.clock_offset(sync)
    assert abs(off["offset_s"] - 5.0) < 1e-9
    assert off["uncertainty_ms"] == 200 and off["min_rtt_ms"] == 400
    assert m.clock_offset([])["offset_s"] is None


def test_jpeg_summary_latency_rate_and_sequence_gaps():
    # Page clock 0; VM clock = page clock + 10 s. Frames captured 0.5 s before they are received.
    sync = [{"t0": 0, "t1": 100, "server_unix_s": 10.05}]
    shown = [
        {"mode": "push", "seq": s, "bytes": 1000, "captured": 10.0 + i * 0.1,
         "server": 10.0 + i * 0.1 + 0.01, "t_recv": 500 + i * 100, "t_shown": 520 + i * 100}
        for i, s in enumerate([1, 2, 4, 5])
    ]  # fmt: skip
    out = m.jpeg_summary({"shown": shown, "sync": sync}, window_s=2.0)
    assert out["frames_shown"] == 4 and out["delivered_fps_over_window"] == 2.0
    assert out["sequence_span"] == 5 and out["sequence_frames_never_shown"] == 1
    assert abs(out["capture_to_screen_ms"]["median"] - 520.0) < 1e-6
    assert out["clock"]["uncertainty_ms"] == 50


def test_whep_summary_reports_decoded_rate_loss_and_playout_lag():
    def row(t, decoded):
        return {"t": t, "inbound": {"framesDecoded": decoded, "framesDropped": 1, "packetsLost": 3,
                "jitterBufferDelay": 0.5, "jitterBufferEmittedCount": 10, "totalDecodeTime": 0.1,
                "estimatedPlayoutTimestamp": 10_000.0 + t - 300}}  # fmt: skip

    result = {
        "stats": [row(0, 0), row(10_000, 100)],
        "sync": [{"t0": 0, "t1": 10, "server_unix_s": 10.0}],
    }
    out = m.whep_summary(result, 10.0)
    assert out["decoded_fps"] == 10.0 and out["packets_lost"] == 3
    assert out["mean_jitter_buffer_delay_ms"] == 50.0
    assert abs(out["estimated_playout_lag_ms"]["median"] - 295.0) < 1e-6  # offset 9.995 s


def test_empty_runs_are_reported_as_empty_not_fabricated():
    assert m.jpeg_summary({"shown": [], "sync": []}, 10)["frames_shown"] == 0
    assert m.whep_summary({"stats": [], "sync": []}, 10)["stats_rows"] == 0
