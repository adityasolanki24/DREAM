"""GR00T NEW_EMBODIMENT config for explicitly recorded DREAM G1 joint references.

Load only in the pinned isolated upstream training environment. No REAL_G1 weights
are asserted to have these semantics. Training data defines named 31D joints in SI
radians, fixed pelvis, 20 Hz, and current external RGB. This needs fine-tuning.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

register_modality_config(
    {
        "video": ModalityConfig(delta_indices=[0], modality_keys=["external_scene"]),
        "state": ModalityConfig(delta_indices=[0], modality_keys=["joints"]),
        "action": ModalityConfig(
            delta_indices=list(
                range(16)
            ),  # Initial design from the official custom-embodiment example.
            modality_keys=["joints"],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="joints",
                )
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.task_description"]
        ),
    },
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)
