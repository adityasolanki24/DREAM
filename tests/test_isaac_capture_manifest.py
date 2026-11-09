import hashlib
import json

from experiments import isaac_capture_manifest as m


def test_summary_counts_frames_bytes_and_digest(tmp_path):
    raw = tmp_path / "raw" / "home_a"
    (raw / "frames").mkdir(parents=True)
    rows = [{"i": 0, "gt": {"x": {}}, "qwen": {"raw": "[]"}}, {"i": 1, "gt": {}}]
    (raw / "frames.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (raw / "frames" / "00000.png").write_bytes(b"abc")
    lines = []
    for p in sorted(tmp_path.rglob("*")):
        if p.is_file():
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            lines.append(f"{digest}  {p.relative_to(tmp_path).as_posix()}")
    (tmp_path / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    files = m.read_checksums(tmp_path / "MANIFEST.sha256")
    s = m.summarise_item("home_a", files, tmp_path / "raw", "raw")
    assert (s["frames"], s["frames_with_qwen"], s["frames_with_gt_instance"]) == (2, 1, 1)
    assert s["files"] == 2 and s["bytes"] > 3 and len(s["all_file_checksums_digest"]) == 64
