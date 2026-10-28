from copy import deepcopy

import pytest

from experiments.summarize_vlm_comparison import compare


def runs():
    rows = [
        {
            "id": f"case{i}",
            "category": "navigation",
            "valid_tool_call": True,
            "correct": i < 6,
            "tool": "look_around",
            "args": {},
            "error": None,
            "latency_s": 1.0,
            "messages_sha256": f"message{i}",
            "tools_sha256": "tools",
        }
        for i in range(11)
    ]
    base = {
        "rows": rows,
        "selected_ids": [row["id"] for row in rows],
        "summary": {"backend": "model", "input_tokens": 100, "output_tokens": 20},
        "sha256": {
            "results/phase1_vlm_compare/decisions.json": "decisions",
            "results/phase1_vlm_compare/frames/test.png": "image",
        },
    }
    reconstruction = deepcopy(base)
    reconstruction["summary"]["backend"] = "offline-provenance-reconstruction"
    qwen = deepcopy(base)
    qwen["sha256"] = {key.replace("/", "\\"): value for key, value in base["sha256"].items()}
    return base, reconstruction, qwen


def test_comparison_normalizes_paths_and_reports_decisions():
    result = compare(*runs())
    assert result["gpt"]["correct"] == 6
    assert result["qwen"]["by_category"]["navigation"] == {"n": 11, "correct": 6}
    assert all(row["agreement"] for row in result["decisions"])
    assert "reconstruction" in result["matching"]["prompt_and_tool_hashes"]


@pytest.mark.parametrize("field", ["messages_sha256", "tools_sha256"])
def test_comparison_rejects_prompt_and_tool_mismatch(field):
    gpt, reconstruction, qwen = runs()
    qwen["rows"][3][field] = "different"
    with pytest.raises(ValueError, match=field):
        compare(gpt, reconstruction, qwen)


def test_comparison_rejects_image_mismatch():
    gpt, reconstruction, qwen = runs()
    qwen["sha256"]["results\\phase1_vlm_compare\\frames\\test.png"] = "different"
    with pytest.raises(ValueError, match="image hashes differ"):
        compare(gpt, reconstruction, qwen)
