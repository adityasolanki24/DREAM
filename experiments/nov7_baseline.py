"""Comparison baselines that run WITHOUT the original-request guard. Measured, never shipped.

Production code refuses to act on a goal that would not establish what the user asked:
``TaskAgent.run`` requires the original-request record and enforces it, and
``interpret_mobile_goal`` checks every model proposal against it. The two pieces here exist
only so that the evaluations which measure the unguarded behaviour can still run it:

- ``interpret_mobile_goal_nov7``: the 7 Nov 2026 interpretation, the model's goal taken as is.
  This is the behaviour behind the Qwen request-to-goal substitutions (README closeout).
- ``UngatedTaskAgent``: ``TaskAgent`` with the binding check, the finish-time re-check and the
  record's digest check switched off. It is the ``abl_substitution_ungated`` ablation in
  experiments/semantic_search_eval.py and the ``nov7`` arm of experiments/isaac_g1_live.py.

Each use is an arm of a comparison against the guarded path (Rule 4); neither is a fallback.
"""

from __future__ import annotations

from dream.runtime.agent import TaskAgent
from dream.runtime.mobile_model import propose_mobile_goal


def interpret_mobile_goal_nov7(backend, utterance, places, scene, camera):
    """The unguarded 7 Nov 2026 interpretation: one model proposal, no binding check."""
    return propose_mobile_goal(backend, utterance, places, scene, camera)


class UngatedTaskAgent(TaskAgent):
    """``TaskAgent`` without the original-request guard (a measured ablation, never production).

    ``spec`` is accepted for call compatibility and ignored: no binding check before acting,
    no digest or binding check at finish, no record on the episode."""

    def run(self, goal, on_event=None, history=None, preempt=None, *, spec=None):
        return self._run(goal, on_event, history, preempt, None)
