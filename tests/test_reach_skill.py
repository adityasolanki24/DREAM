"""Decision flow of the goal-conditioned reach skill (dream/tools/reach.py) with fakes.

No simulator and no robot code: a fake ``ReachBody`` stands in for the G1, so what is tested is the
skill's decisions (what it does when the target is not seen, when the planner refuses, when the
correction is limited, when a stop happens, when it may claim) and that the flow contains no height
rules. The scripted baseline's refusal band is tested the same way.
"""

import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dream.perception.detect import Detection
from dream.perception.reach_target import HandleEstimator
from dream.tools import reach
from dream.tools.reach import ReachSkill, Sight, parse_reach_instruction

HANDLE = np.array([1.5, 0.0, 0.75])


def handle_det(color="red", at=HANDLE, occluded=False) -> Detection:
    return Detection(
        "handle", color, np.array(at, float), 0.0, np.array([0.03, 0.09, 0.03]), 0.002, occluded,
        False, pixels=60, top_centre=np.array(at, float),
    )  # fmt: skip


def fixture_det(x: float = 1.53) -> Detection:
    return Detection(
        "box", "grey", np.array([x, 0.0, 0.6]), 0.0, np.array([0.02, 0.6, 1.2]), 0.003, True,
        True, pixels=900,
    )  # fmt: skip


@dataclass
class Plan:
    stance: tuple = (1.0, 0.0, 0.0)
    arm: str = "right"
    score: float = 1.0
    metrics: dict = field(default_factory=lambda: {"walk": 0.0})
    pelvis_height: float = 0.7
    torso_rpy: tuple = (0.0, 0.1, 0.0)
    target: object = None


class PlanResult:
    def __init__(self, plans, reasons=None):
        self.plans, self.failures = plans, []
        self.n_stances_blocked = 0
        self._reasons = reasons or {}

    @property
    def feasible(self):
        return bool(self.plans)

    def reasons(self):
        return self._reasons


class FakeBody:
    """A scripted robot. ``visible`` decides when the handle is detected; ``hand_error`` is where
    the hand ends relative to the handle."""

    def __init__(
        self,
        visible=lambda n, g: True,
        plans=None,
        hand_error=0.0,
        rollout_ok=True,
        stop_in_execute=False,
        correct_actions=("within_tolerance",),
        pre_ok=True,
        touch_s=0.5,
    ):
        self.visible, self.n_look = visible, 0
        self.plans = [Plan()] if plans is None else plans
        self.hand_error, self.rollout_ok = hand_error, rollout_ok
        self.stop_in_execute = stop_in_execute
        self.correct_actions, self.pre_ok = correct_actions, pre_ok
        self.calls, self.pose_xy = [], np.zeros(3)
        self.stop = None
        self.fixture_x = 1.53
        self.approaches = []
        self.n_ring = 0
        self.with_fixture = True
        # seconds of sustained tactile contact the hand reports (or an iterator of them)
        self.touch_s = touch_s

    # sensing
    def pose(self):
        return self.pose_xy.copy()

    def gaze_range(self):
        return (-0.3, 0.4)

    def now(self):
        return float(len(self.calls))

    def stopped(self):
        return self.stop

    def look(self, gaze):
        self.n_look += 1
        self.calls.append(("look", gaze))
        dets = [fixture_det(self.fixture_x)] if self.with_fixture else []
        if self.visible(self.n_look, gaze):
            dets.append(handle_det())
        return Sight(dets, np.array([0.0, 0.0, 1.2]), float(self.n_look), gaze)

    # motion
    def turn(self, angle):
        self.calls.append(("turn", angle))
        return True

    def go_to(self, x, y, heading):
        self.calls.append(("go_to", round(x, 2), round(y, 2)))
        self.pose_xy = np.array([x, y, heading])
        return True, "ok"

    # planning
    def plan(self, target, covariance, obstacles, stance=None, arm=None, approach=None):
        self.calls.append(("plan", None if stance is None else "explicit"))
        self.approaches.append(approach)
        if stance is None:  # a ring plan: the best stance moves a little every time
            k, self.n_ring = self.n_ring, self.n_ring + 1
            plans = [replace(p, stance=(1.0 + 0.3 * k, 0.0, 0.0)) for p in self.plans]
        else:
            plans = [replace(p, stance=tuple(float(v) for v in stance)) for p in self.plans]
        return PlanResult(plans, {} if plans else {"hand_residual": 16})

    def pre_contact(self, plan, point, obstacles, approach=None):
        self.calls.append(("pre_contact",))
        return Plan(arm=plan.arm) if self.pre_ok else None

    def verify(self, plans, obstacles, pre=None, approach=None):
        self.calls.append(("verify", len(plans)))
        r = SimpleNamespace(passed=self.rollout_ok, reasons=[] if self.rollout_ok else ["com"],
                            metrics={"hand_error": 0.01}, seconds=0.1, simulated=2.0)  # fmt: skip
        return (plans[0] if self.rollout_ok else None), [(plans[0], r)]

    # execution
    def execute(self, plan):
        self.calls.append(("execute",))
        stops = [SimpleNamespace(t=1.0, reason="pelvis tilt")] if self.stop_in_execute else []
        return SimpleNamespace(completed=not stops, refused=None, stops=stops)

    def correct(self, plan, provider, obstacles, executed=None, tolerance=None, approach=None):
        self.calls.append(("correct", tolerance))
        provider()
        its = [{"iteration": i, "error": 0.02, "action": a, "hand": (0, 0, 0)}
               for i, a in enumerate(self.correct_actions)]  # fmt: skip
        return SimpleNamespace(iterations=its, stops=[], refused=None, final_target=None)

    def hand_position(self, arm):
        return HANDLE + np.array([self.hand_error, 0.0, 0.0])

    def touch(self, arm, seconds):
        self.calls.append(("touch",))
        s = next(self.touch_s) if hasattr(self.touch_s, "__next__") else self.touch_s
        return s, (2.0 if s else 0.0)

    def recover(self):
        self.calls.append(("recover",))
        return True


def names(body):
    return [c[0] for c in body.calls]


# -- structured goal parsing ---------------------------------------------------------------------
def test_instruction_parser_accepts_the_fixed_phrase_only():
    g = parse_reach_instruction("Touch the red handle.")
    assert (g.verb, g.color, g.noun, g.tolerance) == ("touch", "red", "handle", 0.03)
    for bad in (
        "touch the red knob",
        "open the cabinet",
        "touch the plaid handle",
        "",
        "touch red",
    ):
        with pytest.raises(ValueError):
            parse_reach_instruction(bad)


def test_unsupported_instruction_is_refused_without_moving():
    body = FakeBody()
    rep = ReachSkill(body).run("open the cabinet")
    assert rep.verdict == "refused" and body.calls == []


# -- observing -----------------------------------------------------------------------------------
def test_gaze_sweep_finds_a_handle_the_level_camera_cannot_see_before_any_turning():
    body = FakeBody(visible=lambda n, g: g == -0.3)
    rep = ReachSkill(body).run("touch the red handle")
    assert rep.verdict == "reached", rep.reason
    assert "turn" not in names(body)
    assert ("look", -0.3) in body.calls and body.calls[0] == ("look", 0.0)
    assert body.calls.count(("look", 0.0)) >= 1  # nominal gaze restored for walking


def test_target_never_seen_is_a_failure_after_turning_all_around_not_a_refusal():
    body = FakeBody(visible=lambda n, g: False)
    rep = ReachSkill(body).run("touch the red handle")
    assert rep.verdict == "failed" and "not found" in rep.reason
    assert names(body).count("turn") == reach.MAX_TURNS - 1
    assert "go_to" not in names(
        body
    )  # a flat unseen wall: nothing to approach (the only box in view is close)


def test_wrong_colour_is_not_the_target():
    body = FakeBody(visible=lambda n, g: True)
    rep = ReachSkill(body).run("touch the green handle")  # only a red handle is ever in view
    assert rep.verdict == "failed"


def test_unresolved_box_is_approached_before_turning_on():
    seen_after_move = {"moved": False}

    class Body(FakeBody):
        def go_to(self, x, y, heading):
            seen_after_move["moved"] = True
            return super().go_to(x, y, heading)

    body = Body(visible=lambda n, g: seen_after_move["moved"])
    body.fixture_x = 2.6  # a box far ahead whose handle is too small to resolve from here
    rep = ReachSkill(body).run("touch the red handle")
    assert rep.verdict == "reached"
    first_move = names(body).index("go_to")
    assert "turn" not in names(body)[:first_move]  # the box in view is approached first


# -- planning, refusal, the stance ----------------------------------------------------------------
def test_planner_infeasible_is_a_refusal_with_reasons_and_the_robot_does_not_move():
    body = FakeBody(plans=[])
    rep = ReachSkill(body).run("touch the red handle")
    assert rep.verdict == "refused"
    assert "no feasible plan within the measured envelope" in rep.reason
    assert "hand_residual" in rep.reason
    assert "go_to" not in names(body) and "execute" not in names(body)
    assert rep.plans[0]["failure_reasons"] == {"hand_residual": 16}


def test_flow_walks_replans_from_the_measured_pose_verifies_executes_then_corrects():
    body = FakeBody()
    rep = ReachSkill(body).run("touch the red handle")
    n = names(body)
    assert rep.verdict == "reached" and rep.measured_error == pytest.approx(0.0, abs=1e-9)
    i_walk = n.index("go_to")
    assert n.index("plan") < i_walk  # planned before walking
    assert "plan" in n[i_walk:] and ("plan", "explicit") in body.calls  # re-planned from the pose
    verifies = [i for i, c in enumerate(n) if c == "verify"]
    assert verifies[0] < i_walk < verifies[-1]  # a stance is chosen on a verified plan, and the
    assert n.index("execute") > verifies[-1]  # gate is repeated at the stance actually reached
    assert n.index("correct") > n.index("execute")
    assert rep.restances == 0


def test_rollout_failure_at_the_stance_triggers_one_bounded_restance_then_refusal():
    body = FakeBody(rollout_ok=False)
    rep = ReachSkill(body).run("touch the red handle")
    assert rep.verdict == "refused" and rep.restances == reach.MAX_RESTANCES
    assert names(body).count("go_to") == 2 and "execute" not in names(body)
    assert "recover" in names(body)  # back to nominal before walking to the new stance


def test_a_restance_must_be_a_different_stance():
    class Same(FakeBody):
        def plan(self, target, covariance, obstacles, stance=None, arm=None, approach=None):
            self.calls.append(("plan", None if stance is None else "explicit"))
            return PlanResult([Plan(stance=(1.0, 0.0, 0.0))])

    body = Same(rollout_ok=False)
    rep = ReachSkill(body).run("touch the red handle")
    assert rep.verdict == "refused" and "not already tried" in rep.reason
    assert names(body).count("go_to") == 1  # no second walk to the stance that just failed


def test_fingers_are_asked_to_point_into_the_face_the_handle_protrudes_from():
    body = FakeBody()
    ReachSkill(body).run("touch the red handle")
    assert body.approaches and all(a == pytest.approx((1.0, 0.0, 0.0)) for a in body.approaches)


def test_pre_contact_stage_runs_first_when_the_face_normal_is_known_and_is_skipped_otherwise():
    body = FakeBody()
    ReachSkill(body).run("touch the red handle")
    cs = [c for c in body.calls if c[0] == "correct"]
    assert [c[1] for c in cs] == [reach.PRE_CONTACT_TOLERANCE, None]  # free air, then the contact
    bare = FakeBody()
    bare.with_fixture = False  # no face seen: no normal to approach along
    ReachSkill(bare).run("touch the red handle")
    assert [c[1] for c in bare.calls if c[0] == "correct"] == [None]  # direct approach only


def test_a_plan_whose_pre_contact_posture_is_unreachable_is_not_chosen():
    body = FakeBody(pre_ok=False)
    rep = ReachSkill(body).run("touch the red handle")
    assert rep.verdict == "refused" and "pre_contact_unreachable" in rep.reason
    assert "execute" not in names(body)


# -- execution, claims ----------------------------------------------------------------------------
def test_protective_stop_during_execution_is_a_failure_that_recovers_and_never_claims():
    body = FakeBody(stop_in_execute=True)
    rep = ReachSkill(body).run("touch the red handle")
    assert rep.verdict == "failed" and "execution stopped" in rep.reason
    assert "recover" in names(body) and "correct" not in names(body)


def test_claim_requires_sustained_tactile_contact_where_the_handle_was_perceived():
    ok = ReachSkill(FakeBody(hand_error=0.010)).run("touch the red handle")
    no_touch = ReachSkill(
        FakeBody(hand_error=0.045, touch_s=0.0, correct_actions=("no_improvement",))
    ).run("touch the red handle")
    brief = ReachSkill(FakeBody(hand_error=0.010, touch_s=0.1)).run("touch the red handle")
    # touching something, but not the handle that was perceived
    elsewhere = ReachSkill(FakeBody(hand_error=0.4)).run("touch the red handle")
    assert ok.verdict == "reached" and ok.execution["touch"]["sustained_s"] >= reach.TOUCH_SUSTAIN_S
    assert no_touch.verdict == "failed" and no_touch.restances == reach.MAX_RESTANCES
    assert brief.verdict == "failed" and "no claim" in brief.reason
    assert elsewhere.verdict == "failed" and elsewhere.execution["touch"]["consistent"] is False


def test_without_contact_the_hand_presses_a_little_deeper_before_giving_up():
    body = FakeBody(hand_error=0.010, touch_s=iter([0.0, 0.0, 0.5]))
    rep = ReachSkill(body).run("touch the red handle")
    assert rep.verdict == "reached" and rep.execution["touch"]["seek_attempts"] == 2
    assert [c["stage"] for c in rep.execution["correction"]].count("seek") >= 2


def test_inside_the_tolerance_without_contact_is_reported_without_a_pointless_restance():
    body = FakeBody(hand_error=0.026, touch_s=0.0, correct_actions=("no_improvement",))
    rep = ReachSkill(body).run("touch the red handle")
    assert rep.verdict == "failed" and rep.restances == 0
    assert "no claim" in rep.reason
    assert names(body).count("go_to") == 1


def test_limited_correction_restances_once_and_the_second_attempt_can_succeed():
    errs = iter([0.05, 0.005])

    class Body(FakeBody):
        def hand_position(self, arm):
            return HANDLE + np.array([next(errs), 0.0, 0.0])

    body = Body(correct_actions=("no_improvement",), touch_s=iter([0.0, 0.0, 0.0, 0.5]))
    rep = ReachSkill(body).run("touch the red handle")
    assert rep.verdict == "reached" and rep.restances == 1
    assert "recover" in names(body)  # back to the nominal posture before walking again


def test_report_records_the_perceived_target_plans_rollouts_and_events():
    rep = ReachSkill(FakeBody()).run("touch the red handle")
    assert rep.target["color"] == "red" and rep.target["sigma"] > 0
    assert rep.target["fixture"] is not None
    assert [p["stage"] for p in rep.plans][:2] == ["initial", "at_stance"]
    assert rep.rollouts and rep.events and rep.posture["arm"] == "right"


# -- no height rules ------------------------------------------------------------------------------
def test_skill_contains_no_height_rules_or_scene_access():
    src = Path(reach.__file__).read_text()
    code = "\n".join(line.split("#")[0] for line in src.splitlines())
    assert not re.search(
        r"position\[2\]|\.z\b|height\s*[<>]|[<>]\s*(0\.\d+|1\.\d+)\s*(#|$)", code
    ), "the skill must not branch on a height"
    for banned in ("mujoco", "dream.sim", "dream.robot", "dream.eval", "truth", "_odom"):
        assert banned not in code.replace("dream/robot", ""), banned


def test_skill_flow_is_identical_for_low_and_high_targets():
    """The same decisions are made whatever the height: only the planner's answer differs."""
    traces = []
    for z in (0.25, 1.35):
        body = FakeBody()
        orig = body.look

        def look(g, orig=orig, z=z):
            s = orig(g)
            s.detections[-1] = handle_det(at=(1.5, 0.0, z))
            return s

        body.look = look
        ReachSkill(body).run("touch the red handle")
        traces.append(names(body))
    assert traces[0] == traces[1]


# -- the estimator -------------------------------------------------------------------------------
def test_estimator_fuses_views_reports_covariance_and_builds_perceived_boxes():
    est = HandleEstimator("red")
    cam = np.array([0.0, 0.0, 1.2])
    for k in range(3):
        d = handle_det(at=HANDLE + np.array([0.002 * k, 0.0, 0.0]))
        assert est.ingest([fixture_det(), d, handle_det("green", (2.0, 1.0, 0.4))], cam, k)
    e = est.estimate()
    assert e.n_views == 3 and np.allclose(e.position, HANDLE + [0.002, 0, 0], atol=0.002)
    assert e.covariance.shape == (3, 3) and np.all(np.linalg.eigvalsh(e.covariance) > 0)
    assert e.fixture is not None and e.fixture.hand_may_touch
    assert e.face_normal == pytest.approx((-1.0, 0.0), abs=1e-6)  # the face looks at the robot
    colors = {o.color for o in e.obstacles}
    assert "green" in colors and "red" not in colors  # the target is not an obstacle
    # the unseen volume behind a visible face is occupied: the slab is extruded away from the camera
    assert 2 * max(e.fixture.half_extents[:2]) >= 0.5 or e.fixture.half_extents[0] >= 0.25
    # a handle protrudes from the face: the fixture's front lies behind the handle centre
    front = e.fixture.center[0] - e.fixture.half_extents[0]
    assert front > e.position[0]


def test_pieces_of_the_fixture_face_all_carry_the_target_and_other_boxes_do_not():
    """Regression (development episode dev_h0.75_L1_s101): the detector split one face into two
    boxes; the piece the handle was not nearest to became an obstacle that enclosed the target,
    and every plan failed."""
    est = HandleEstimator("red")
    a = fixture_det()
    b = Detection(
        "box", "grey", np.array([1.53, 0.45, 0.6]), 0.0, np.array([0.02, 0.3, 1.2]), 0.003, True,
        True, pixels=400,
    )  # fmt: skip
    far = Detection(
        "box", "grey", np.array([1.6, -1.5, 0.3]), 0.0, np.array([0.3, 0.3, 0.6]), 0.003, True,
        True, pixels=400,
    )  # fmt: skip
    est.ingest([a, b, far, handle_det()], np.array([0.0, 0.0, 1.2]))
    e = est.estimate()
    touch = [o for o in e.obstacles if o.hand_may_touch]
    assert len(touch) == 2 and len(e.obstacles) == 3  # the face and its piece; the far box is not
    assert all(o.center[0] - o.half_extents[0] > e.position[0] for o in touch)  # behind the handle


def test_small_object_flush_against_a_thin_face_is_a_handle():
    """Regression (development episodes dev_h1.10_L0_s101, _L1_s103, _L1_s105): the detector
    labels a component a handle only if its top-centre lies OUTSIDE the box footprint rectangle.
    A face seen head-on has a rectangle as deep as the depth noise, so a handle a centimetre or
    two in front of it fell inside the rectangle, stayed an "object", and was never found."""
    box = Detection(
        "box", "grey", np.array([1.53, 0.0, 0.6]), 0.0, np.array([0.04, 0.6, 1.2]), 0.003, True,
        True, pixels=900,
    )  # fmt: skip

    def small(color, at, z=None):
        top = np.array([at[0], at[1], at[2] if z is None else z])
        return Detection(
            "object", color, np.array(at, float), 0.0, np.array([0.02, 0.09, 0.02]), 0.002,
            False, False, pixels=60, top_centre=top,
        )  # fmt: skip

    est = HandleEstimator("red")
    assert est.ingest([box, small("red", (1.52, 0.0, 0.75))], np.zeros(3))  # inside the rectangle
    assert est.estimate().position == pytest.approx([1.52, 0.0, 0.75])
    assert est.estimate().candidates == 1
    for bad in (
        small("red", (1.0, 0.8, 0.75)),  # not against any box
        small("red", (1.52, 0.0, 1.3)),  # above the box top: sitting on it, not a handle
        small("green", (1.52, 0.0, 0.75)),  # another colour
    ):
        assert not HandleEstimator("red").ingest([box, bad], np.zeros(3))
    big = small("red", (1.52, 0.0, 0.75))
    big = replace(big, extent=np.array([0.4, 0.4, 0.4]))  # a crate, not a handle
    assert not HandleEstimator("red").ingest([box, big], np.zeros(3))


def test_estimator_ignores_occluded_fragments_and_other_colours_and_resets():
    est = HandleEstimator("red")
    cam = np.zeros(3)
    assert not est.ingest([handle_det(occluded=True), handle_det("blue")], cam)
    assert est.estimate() is None
    est.ingest([handle_det()], cam)
    est.reset()
    assert est.estimate() is None


# -- the scripted baseline's refusal band ---------------------------------------------------------
def baseline_refusal(z: float) -> str:
    from experiments.g1_scripted_baseline import ScriptedBaselineReach

    class Body(FakeBody):
        def look(self, gaze):
            s = super().look(gaze)
            s.detections[-1] = handle_det(at=(1.5, 0.0, z))
            return s

    body = Body()
    rep = ScriptedBaselineReach(body).run("touch the red handle")
    return rep.verdict if "go_to" not in names(body) else "attempted"


@pytest.mark.parametrize("z", [0.10, 0.35, 0.5499, 1.1501, 1.40])
def test_scripted_baseline_refuses_outside_the_upright_band(z):
    assert baseline_refusal(z) == "refused"


def test_scripted_baseline_band_constants_match_the_isaac_task_gate():
    from experiments import g1_scripted_baseline as b

    assert b.HEIGHT_BAND == (0.55, 1.15) and b.STAND_OFF == 0.42 and b.ARM == "right"
    src = Path("experiments/isaac_task.py").read_text()
    assert "0.55 <= f[1][2] <= 1.15" in src  # the gate the baseline copies
