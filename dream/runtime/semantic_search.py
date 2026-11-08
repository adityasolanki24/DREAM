"""Planners for object search ("Find the refrigerator") under TaskAgent's Planner protocol.

Every arm shares perception, the object tracker, the verifier (dream/semantics/verify.py),
candidate verification and navigation (Dijkstra on the robot's map, the toolbox's follower).
They differ only in how they choose where to look next:

- ``utility`` (proposed): receding-horizon choice of the next viewpoint by expected value
  per second,
      U(v) = (alpha * detection(v) + lam * gain(v) / A_REF) / (T_LOOK + cost(v) / SPEED)
             - gamma * risk(v)
  where detection(v) is the share of the current target belief a full turn at v would
  explain, gain(v) the unknown area it would reach (dream/semantics/belief.py), cost(v) the
  walking distance and risk(v) a clearance penalty at v. Walk there, turn in place, update,
  choose again. No exact POMDP: one step of lookahead over a few hundred candidates.
  The additive form alpha*detection + lam*gain - beta*cost first tried mixes units: on
  development home 100-B every useful view scored below a worthless one beside the robot,
  and the search stopped early. A rate is the classical greedy rule for search with travel
  cost (value per unit effort), and needs no exchange rate between metres and probability.
- ``frontier_gain`` (Baseline A: DREAM's existing explorer, dream/runtime/search.py): the
  frontier with the largest gain - FRONTIER_COST_WEIGHT * cost; no semantic knowledge.
- ``frontier_nearest`` (Baseline B: classical nearest-frontier exploration, Yamauchi 1997).
- ``room_priority`` (Baseline C): visit regions in decreasing semantic factor (how strongly
  their inferred room category suggests the target), covering each before the next;
  nearest frontier while no region points to the target.

When a track carries the target label but is not verified, every arm goes to look at it
from a new position (``MAX_VERIFY`` attempts), so all arms verify alike. Stopping is typed
(dream/tools/intent.TERMINAL_OUTCOMES); the time budget is applied outside the planner.
None of this code knows the layout, the room types, the target's location or the seed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from dream.navigation.planning import frontiers
from dream.navigation.rpp import line_of_sight
from dream.perception.occupancy import CELL
from dream.perception.semantic import Track
from dream.runtime.agent import Finish, Step, ToolCall
from dream.tools.goals import FindObject
from dream.tools.mobile import FRONTIER_COST_WEIGHT, PLACE_TOLERANCE

POLICIES = ("utility", "frontier_gain", "frontier_nearest", "room_priority")
A_REF = 10.0  # m^2: information gain normaliser (about a small room's floor)
LATTICE = 0.6  # m between candidate viewpoints
MAX_VERIFY = 3  # look attempts per candidate track
VERIFY_RING = (1.2, 2.0)  # m from a candidate's centre: inside the coverage model's good band
REACH_RING = (0.7, 1.0)  # m from a verified object's footprint, to stand at it
MAX_REACH = 2
DROP_RADIUS = 0.5  # m: viewpoints this close to two failed go_to targets are skipped
LOOK_HERE_RADIUS = 0.5  # m: a full turn within this of an earlier one adds little
EXHAUSTED_DETECTION = 0.01  # share of belief: below this, no reachable look is worth taking
EXHAUSTED_GAIN = 0.5  # m^2 of unknown area
T_LOOK = 10.4  # s: one full turn in place (8 x 45 degrees at 0.5 rad/s plus settling)
SPEED = 0.35  # m/s: typical walking speed under the navigator's 0.4 m/s cruise
REJECT_VIEWS = 6  # views of a candidate before its label purity can refute it
REJECT_PURITY = 0.3


@dataclass(frozen=True)
class UtilityWeights:
    """Design starting point; values were chosen on development homes (configs record them)."""

    alpha: float = 1.0
    lam: float = 0.3
    gamma: float = 0.2


@dataclass
class SemanticSearchPlanner:
    toolbox: object  # MobileToolbox with a SemanticWorld (the robot's own maps only)
    policy: str = "utility"
    weights: UtilityWeights = field(default_factory=UtilityWeights)
    look_after_move: bool | None = None  # default: True for utility/room_priority
    memory_hints: list = field(default_factory=list)
    trace: list = field(default_factory=list)  # decision records for evaluation
    _purpose: dict = field(default_factory=dict)
    _verify_tries: dict = field(default_factory=dict)
    _reach_tries: int = 0
    _looked: list = field(default_factory=list)  # odometry xy of every full turn issued

    def __post_init__(self) -> None:
        if self.policy not in POLICIES:
            raise ValueError(f"unknown policy {self.policy!r}")
        if self.look_after_move is None:
            self.look_after_move = self.policy in ("utility", "room_priority")

    # ----- protocol ----------------------------------------------------------------------------
    def next_action(self, goal, scene: dict, verdict: dict, history: list[Step]):
        if not isinstance(goal, FindObject):
            return Finish(False, f"this planner searches for objects, not {goal!r}")
        world = self.toolbox.semantic
        world.set_target(goal.target, self.memory_hints)
        if verdict.get("satisfied"):
            return Finish(True, f"verified: {verdict['reason']}", "target_found_verified")
        if not any(s.call.name == "look_around" for s in history):
            return self._act(ToolCall("look_around"), "initial_look")
        last = history[-1] if history else None
        follow = self._follow_up(last)
        if follow is not None:
            return follow
        world.refresh()
        passable, start, dist = self.toolbox.reachability()
        pose = self.toolbox.base.odometry()
        target = goal.target
        if verdict.get("status") == "target_found_verified":  # outcome "reached" pending
            return self._reach(world, verdict, passable, dist)
        cand = self._candidate(world, target, passable, dist, history)
        if cand is not None:
            return cand
        return self._explore(world, target, passable, dist, pose, history)

    # ----- steps -------------------------------------------------------------------------------
    def _act(self, call: ToolCall, purpose: str, **info):
        if call.name == "look_around":
            self._looked.append(np.asarray(self.toolbox.base.odometry()[:2], float))
        self._purpose = {"purpose": purpose, **info}
        self.trace.append({"call": call.name, "args": call.args, **self._purpose})
        return call

    def _follow_up(self, last: Step | None):
        """After arriving: turn in place (exploration) or face the object (verification)."""
        if last is None or last.call.name != "go_to" or not self._purpose:
            return None
        purpose = self._purpose.get("purpose")
        if not last.result.ok:
            return None
        if purpose in ("verify", "reach"):
            x, y = self._purpose["object_xy"]
            return self._act(ToolCall("look_at", {"x": x, "y": y}), purpose + "_look",
                             track=self._purpose.get("track"))  # fmt: skip
        if purpose == "explore" and self.look_after_move:
            return self._act(ToolCall("look_around"), "explore_look")
        return None

    def _candidate(self, world, target, passable, dist, history):
        tracks = [t for t in world.map.candidates(target) if not self._rejected(t, target)]
        best = None
        for tr in tracks:
            vp = self._verification_view(world, tr, target, passable, dist, VERIFY_RING)
            if vp is None:  # not reachable yet; exploring may open a way
                continue
            key = (-tr.support(target), vp[1])
            if best is None or key < best[0]:
                best = (key, tr, vp)
        if best is None:
            return None
        _, tr, (xy, cost, heading) = best
        self._verify_tries[tr.id] = self._verify_tries.get(tr.id, 0) + 1
        return self._act(
            ToolCall("go_to", {"x": float(xy[0]), "y": float(xy[1]), "heading": heading}),
            "verify", track=tr.id, object_xy=[float(tr.position[0]), float(tr.position[1])],
            cost_m=round(cost, 2),
        )  # fmt: skip

    def _rejected(self, tr: Track, target: str) -> bool:
        """Given up on: verification attempts spent, or the evidence refutes the label (the
        object is plainly there but rarely called the target, or no longer detected)."""
        tries = self._verify_tries.get(tr.id, 0)
        views = sum(tr.labels.values())
        refuted = (views >= REJECT_VIEWS and tr.purity(target) < REJECT_PURITY) or (
            tr.misses >= 3 and tr.purity(target) < 0.5
        )
        return tries >= MAX_VERIFY or refuted

    def _verification_view(self, world, tr, target, passable, dist, ring, footprint=False):
        """Cheapest reachable standing cell on ``ring`` around the object with line of sight,
        at least 0.5 m from every camera position that already labelled it the target."""
        grid = world.grid
        cells = np.argwhere(passable & np.isfinite(dist))
        if not len(cells):
            return None
        pts = grid.centre(cells[:, 0], cells[:, 1])
        if footprint:
            lo, hi = tr.lo[:2], tr.hi[:2]
            d = np.linalg.norm(np.maximum(np.maximum(lo - pts, pts - hi), 0.0), axis=1)
        else:
            d = np.linalg.norm(pts - tr.position[:2], axis=1)
        ok = (d >= ring[0]) & (d <= ring[1])
        if not footprint:  # a new position, or the same views would just repeat
            for c in (c for views in tr.label_views.values() for c in views):
                ok &= np.linalg.norm(pts - c, axis=1) >= 0.5
        idx = np.flatnonzero(ok)
        if not len(idx):
            return None
        order = idx[np.argsort(dist[cells[idx, 0], cells[idx, 1]])]
        occ = grid.occupied()
        own = np.zeros_like(occ)
        i0, j0 = grid.cell(tr.lo[:2] - 0.1)
        i1, j1 = grid.cell(tr.hi[:2] + 0.1)
        own[max(i0, 0) : i1 + 1, max(j0, 0) : j1 + 1] = True
        clear = ~(occ & ~own)
        goal_cell = tuple(int(v) for v in grid.cell(tr.position[:2]))
        for k in order[:200]:
            cell = tuple(int(v) for v in cells[k])
            if line_of_sight(clear, cell, goal_cell):
                xy = pts[k]
                heading = math.atan2(tr.position[1] - xy[1], tr.position[0] - xy[0])
                return xy, float(dist[cell]), heading
        return None

    def _reach(self, world, verdict, passable, dist):
        tr = next(t for t in world.map.tracks if t.id == verdict["track"])
        if self._reach_tries >= MAX_REACH:
            return Finish(False, f"{verdict['target']} verified but no reachable place to stand "
                          "at it", "target_unreachable")  # fmt: skip
        vp = self._verification_view(world, tr, verdict["target"], passable, dist, REACH_RING,
                                     footprint=True)  # fmt: skip
        self._reach_tries += 1
        if vp is None:
            return Finish(False, f"{verdict['target']} verified but no reachable place to stand "
                          "at it", "target_unreachable")  # fmt: skip
        xy, cost, heading = vp
        return self._act(
            ToolCall("go_to", {"x": float(xy[0]), "y": float(xy[1]), "heading": heading}),
            "reach", track=tr.id, object_xy=[float(tr.position[0]), float(tr.position[1])],
        )  # fmt: skip

    # ----- exploration -------------------------------------------------------------------------
    def _explore(self, world, target, passable, dist, pose, history):
        grid = world.grid
        dead = [
            np.array([s.call.args["x"], s.call.args["y"]])
            for s in history
            if s.call.name == "go_to" and not s.result.ok
        ]

        def alive(xy) -> bool:
            return sum(np.linalg.norm(np.asarray(xy) - d) < DROP_RADIUS for d in dead) < 2

        fr = [f for f in frontiers(grid, passable, dist)
              if f.cost > PLACE_TOLERANCE and alive(grid.centre(*f.cell))]  # fmt: skip
        if self.policy in ("frontier_gain", "frontier_nearest"):
            if not fr:
                return self._stop(world, target, "explored the reachable space")
            if self.policy == "frontier_nearest":
                f = min(fr, key=lambda f: f.cost)
            else:
                f = max(fr, key=lambda f: f.gain - FRONTIER_COST_WEIGHT * f.cost)
            xy = grid.centre(*f.cell)
            return self._act(ToolCall("go_to", {"x": float(xy[0]), "y": float(xy[1])}),
                             "explore", kind="frontier", gain_m2=round(f.gain, 2),
                             cost_m=round(f.cost, 2))  # fmt: skip
        views = [
            v for v in self._views(world, passable, dist, fr, pose, history, alive)
            if v["detection"] >= EXHAUSTED_DETECTION or v["gain_m2"] >= EXHAUSTED_GAIN
        ]  # fmt: skip
        if not views:
            return self._stop(world, target, "no reachable look is expected to reveal more")
        if self.policy == "room_priority":
            choice = self._room_priority(world, views, fr, grid)
        else:
            w = self.weights
            for v in views:
                value = w.alpha * v["detection"] + w.lam * v["gain_m2"] / A_REF
                v["utility"] = value / (T_LOOK + v["cost_m"] / SPEED) - w.gamma * v["risk"]
            choice = max(views, key=lambda v: v["utility"])
        if choice is None:
            return self._stop(world, target, "no reachable look is expected to reveal more")
        info = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in choice.items()
                if k != "xy"}  # fmt: skip
        if choice["here"]:
            return self._act(ToolCall("look_around"), "explore_look", **info)
        xy = choice["xy"]
        return self._act(ToolCall("go_to", {"x": float(xy[0]), "y": float(xy[1])}),
                         "explore", **info)  # fmt: skip

    def _views(self, world, passable, dist, fr, pose, history, alive) -> list[dict]:
        grid = world.grid
        step = int(LATTICE / CELL)
        ok = passable & np.isfinite(dist)
        lattice = np.zeros_like(ok)
        lattice[::step, ::step] = True
        cells = [tuple(c) for c in np.argwhere(ok & lattice)]
        cells += [f.cell for f in fr]
        looked = self._looked
        here = tuple(int(v) for v in grid.cell(pose[:2]))
        clearance = ndimage.distance_transform_edt(~grid.occupied()) * CELL
        out = []
        for cell in set(cells) | {here}:
            xy = grid.centre(*cell)
            is_here = cell == here
            if not is_here and (not alive(xy) or dist[cell] <= PLACE_TOLERANCE):
                continue
            if is_here and any(np.linalg.norm(xy - p) < LOOK_HERE_RADIUS for p in looked):
                continue
            v = world.target.view(grid, world.map, xy)
            out.append({
                "xy": xy, "here": is_here, "detection": v.detection, "gain_m2": v.gain_m2,
                "cost_m": 0.0 if is_here else float(dist[cell]),
                "risk": float(np.clip((0.6 - clearance[cell]) / 0.25, 0.0, 1.0)),
                "region": world.segmenter.region_of(grid, xy),
            })  # fmt: skip
        return out

    def _room_priority(self, world, views, fr, grid):
        factor = world.target.region_factor
        mass = world.target.region_mass(world.segmenter)
        ranked = sorted(
            (r for r, f in factor.items() if f > 1.0 and mass.get(r, 0.0) > 0.02),
            key=lambda r: -factor[r],
        )
        for r in ranked:
            inside = [v for v in views if v["region"] == r and v["detection"] > 0]
            if inside:
                return max(inside, key=lambda v: v["detection"])
        if fr:
            f = min(fr, key=lambda f: f.cost)
            xy = grid.centre(*f.cell)
            near = min(views, key=lambda v: float(np.linalg.norm(v["xy"] - xy)))
            return {**near, "xy": xy, "here": False, "cost_m": float(f.cost)}
        rest = sorted((v for v in views if v["detection"] > 0),
                      key=lambda v: (-factor.get(v["region"], 1.0), -v["detection"]))  # fmt: skip
        return rest[0] if rest else None

    def _stop(self, world, target, why: str) -> Finish:
        pending = [t for t in world.map.candidates(target) if not self._rejected(t, target)]
        unconfirmed = world.map.candidates(target)
        if pending or unconfirmed:
            return Finish(False, f"{why}; a possible {target} was seen but not verified",
                          "possible_target_unconfirmed")  # fmt: skip
        return Finish(False, f"{why}; no {target} found", "search_exhausted")


def stop_outcome(toolbox, target: str, episode_completion: str | None, reason: str) -> str:
    """Typed outcome of an episode that ended without verified success, from the robot's own
    state (budget expiry has no completion of its own)."""
    if episode_completion:
        return episode_completion
    if "cancel" in reason:
        return "cancelled"
    if toolbox.base.stop_reason:
        return "safety_interruption"
    world = toolbox.semantic
    if world is not None and world.map.candidates(target):
        return "possible_target_unconfirmed"
    return "target_not_yet_found"
