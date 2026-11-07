from experiments.g1_explore_room import episode


def test_seed_without_a_valid_room_is_recorded_not_raised():
    # nominal 1313 of the frozen closeout set: "no free start position" (found in the cloud run).
    row = episode(("nominal", 1313, "gain", {}, None, "sha"))
    assert row["scene_error"] and row["success"] is False and row["claimed"] is False
    assert row["refusal_category"] == "scene_generation_failed"
    assert "no free start position" in row["reason"]
