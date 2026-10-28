"""Exercise the unmodified pinned NVIDIA episode loader on an exported dataset.

Run in a separate CPU environment with pandas 2.3.3, pyarrow 22.0.0, torch 2.9.0,
torchcodec 0.8.1 and compatible FFmpeg shared libraries. No model weights are used.
Missing video support is recorded as a failed full-loader gate, never patched out.
The source manifest permits a minimal namespace checkout or a full upstream tree.
"""

import argparse
import hashlib
import json
import runpy
import sys
from pathlib import Path

import numpy as np


def validate(upstream: Path, dataset: Path, out: Path) -> dict:
    manifest = json.loads(Path("experiments/configs/g1_loader_sources.json").read_text())
    for name, expected in manifest["files"].items():
        if hashlib.sha256((upstream / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"upstream source differs from pinned revision: {name}")
    sys.path.insert(0, str(upstream.resolve()))
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.embodiment_tags import EmbodimentTag

    runpy.run_path("experiments/g1_modality.py")
    cfg = MODALITY_CONFIGS[EmbodimentTag.NEW_EMBODIMENT.value]
    action_config = cfg["action"].action_configs[0]
    assert action_config.rep.value == "absolute"
    assert cfg["action"].delta_indices == list(range(16))
    loader = LeRobotEpisodeLoader(dataset, cfg)
    stats = loader.get_dataset_statistics()
    import pandas as pd

    lengths = []
    for index, episode in enumerate(loader.episodes_metadata):
        eid = episode["episode_index"]
        raw = pd.read_parquet(
            dataset
            / loader.data_path_pattern.format(
                episode_chunk=eid // loader.chunk_size, episode_index=eid
            )
        )
        decoded = loader._load_parquet_data(eid)
        for source, loaded in (("observation.state", "state.joints"), ("action", "action.joints")):
            np.testing.assert_array_equal(np.stack(raw[source]), np.stack(decoded[loaded]))
        assert all(
            isinstance(x, str) and x for x in decoded["language.annotation.human.task_description"]
        )
        assert loader.get_episode_length(index) == len(raw)
        np.testing.assert_allclose(np.diff(raw["timestamp"]), 0.05, atol=2e-6)
        lengths.append(len(raw))
    result = {
        "upstream": manifest,
        "dataset": str(dataset.resolve()),
        "metadata_parquet_language_passed": True,
        "normalization_loaded": stats,
        "lengths": lengths,
        "absolute_action": True,
        "action_horizon": 16,
        "sampling_hz": loader.fps,
        "new_embodiment_registered": True,
        "full_video_loader_passed": False,
        "training_ready": False,
        "model_calls": 0,
        "scope": "episode decoding; not processor/training sampler or motor validation",
    }
    try:
        for index, length in enumerate(lengths):
            frames = loader[index]["video.external_scene"]
            assert len(frames) == length and frames.iloc[0].shape == (720, 1280, 3)
        result["full_video_loader_passed"] = True
    except (ImportError, RuntimeError, OSError) as error:
        result["video_error"] = f"{type(error).__name__}: {error}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.upstream, args.dataset, args.out)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "metadata_parquet_language_passed",
                    "full_video_loader_passed",
                    "lengths",
                )
            },
            indent=2,
        )
    )
    sys.exit(0 if result["full_video_loader_passed"] else 2)
