"""Unfamiliar-height reach test (W5): "touch the <colour> handle" at heights never seen.

    python experiments/g1_reach_heights.py experiments/configs/g1_reach_heights.json --set dev
    python experiments/g1_reach_heights.py experiments/configs/g1_reach_heights.json --set heldout
    python experiments/g1_reach_heights.py experiments/configs/g1_reach_heights.json --set dev \
        --summarize

Two systems on identical scenes: ``agent`` (the goal-conditioned reach skill with the whole-body
posture planner, dream/tools/reach.py) and ``scripted_baseline`` (the fixed upright right-arm
reach, experiments/g1_scripted_baseline.py). Each episode is one JSON file under
results/<set directory>/<system>/, written when the episode finishes, so a rerun resumes.
Ground truth is read only in the evaluator (dream/eval/reach.py) and in scene construction.
Odometry is ORACLE-ASSISTED (dream/sim/g1_mobile.py); perception is the classical baseline.
The held-out set is run once, after the code is frozen (AGENTS.md Rule 7).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import platform
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

if os.name != "nt" and "DISPLAY" not in os.environ:
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dream.eval.reach import (  # noqa: E402
    ReachProbe,
    judge,
    reference_feasibility,
    scene_geom_categories,
    wilson,
)
from dream.perception.spatial import SpatialMemory  # noqa: E402
from dream.robot.g1_posture import PlannerConfig, PosturePlanner  # noqa: E402
from dream.robot.g1_reach_ports import G1ReachBody  # noqa: E402
from dream.robot.g1_wbc import WBCLocomotion  # noqa: E402
from dream.robot.wholebody import ControllerCapabilities, WBCBackend  # noqa: E402
from dream.sim.g1_mobile import G1MobileBase, G1MobileEnv, build_mobile_g1  # noqa: E402
from dream.sim.reach_scenes import add_reach_scene, sample_reach_scene  # noqa: E402
from dream.tools.mobile import MobileToolbox  # noqa: E402
from dream.tools.reach import ReachSkill  # noqa: E402

SYSTEMS = ("agent", "scripted_baseline")


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# -- episode specifications -----------------------------------------------------------------------
def dev_specs(cfg: dict) -> list[dict]:
    d = cfg["dev"]
    out = []
    for h in d["heights"]:
        for li, layout in enumerate(d["layouts"]):
            for seed in d["seeds"]:
                out.append(
                    {
                        "id": f"dev_h{h:.2f}_L{li}_s{seed}", "set": "dev", "seed": seed,
                        "height": float(h), "layout": li,
                        "distance": layout["distance"], "lateral": layout["lateral"],
                        "yaw_jitter": math.radians(layout["yaw_jitter_deg"]),
                        "n_crates": layout["crates"], "second_fixture": layout["second_fixture"],
                    }
                )  # fmt: skip
    return out


def heldout_specs(cfg: dict) -> list[dict]:
    d = cfg["heldout"]
    out = []
    for seed in d["seeds"]:
        r = np.random.default_rng([seed, 99])
        j = math.radians(d["yaw_jitter_deg"])
        out.append(
            {
                "id": f"heldout_s{seed}", "set": "heldout", "seed": seed,
                "height": float(r.uniform(*d["height_m"])),
                "distance": float(r.uniform(*d["distance_m"])),
                "lateral": float(r.uniform(*d["lateral_m"])),
                "yaw_jitter": float(r.uniform(-j, j)),
                "n_crates": int(r.integers(0, d["max_crates"] + 1)),
                "second_fixture": bool(r.random() < d["second_fixture_prob"]),
            }
        )  # fmt: skip
    return out


def specs(cfg: dict, which: str) -> list[dict]:
    return dev_specs(cfg) if which == "dev" else heldout_specs(cfg)


# -- one episode ----------------------------------------------------------------------------------
def to_jsonable(x):
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return to_jsonable(x.tolist())
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if hasattr(x, "__dataclass_fields__"):
        return to_jsonable({k: getattr(x, k) for k in x.__dataclass_fields__})
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def make_rollout_factory(caps):
    def factory(model, stance):
        env = G1MobileEnv(model, np.random.default_rng(0), head_camera=None)
        env.reset(*stance)
        return WBCBackend(env, WBCLocomotion.from_cache(), caps)

    return factory


def reference_for(spec: dict, scene, caps, out: pathlib.Path) -> dict:
    """Reference feasibility belongs to the scene: cached per episode id, shared by both systems."""
    path = out / "reference" / f"{spec['id']}.json"
    if path.exists():
        return json.loads(path.read_text())
    ref = reference_feasibility(scene, PosturePlanner(caps), make_rollout_factory(caps))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(ref), indent=1))
    return ref


def trace_summary(body, probe) -> dict:
    rows = [r for s in body.supervisors for r in s.trace]
    cmd = body.backend.command()
    last = rows[-1] if rows else {}

    def peak(k, fn=max):
        v = [r[k] for r in rows if k in r and r[k] is not None]
        return float(fn(v)) if v else None

    return {
        "commanded_final": {
            "pelvis_height": cmd.height, "torso_rpy": list(cmd.rpy),
            "arm_max_abs": float(np.max(np.abs(cmd.arm))),
        },
        "achieved_final": {k: last.get(k) for k in ("ach_height", "ach_rpy", "com_margin")},
        "height_error_final": None if not rows else float(last["ach_height"] - cmd.height),
        "peak_torque_ratio": peak("torque_ratio"), "peak_tilt": peak("tilt"),
        "peak_tilt_rate": peak("tilt_rate"), "min_com_margin": peak("com_margin", min),
        "monitored_ticks": len(rows),
    }  # fmt: skip


def run_episode(args: tuple) -> dict:
    spec, system, cfg, out_dir, sha, video = args
    out = pathlib.Path(out_dir)
    t_wall = time.perf_counter()
    path = out / system / f"{spec['id']}.json"
    result = {"spec": spec, "system": system, "git_sha": sha}
    try:
        scene = sample_reach_scene(
            spec["seed"], spec["height"], spec["distance"], spec["lateral"], spec["yaw_jitter"],
            spec["n_crates"], spec["second_fixture"],
        )  # fmt: skip
        caps = ControllerCapabilities.from_envelope_v2(ROOT / cfg["envelope"])
        env = G1MobileEnv(
            build_mobile_g1(scene=lambda s: add_reach_scene(s, scene), hands=True),
            np.random.default_rng([spec["seed"], 1]),
        )
        env.reset(*scene.start)
        rng = np.random.default_rng([spec["seed"], 3])
        env.data.qpos[env.qadr] += rng.normal(scale=0.02, size=len(env.qadr))  # scene setup
        mujoco.mj_forward(env.model, env.data)
        ctrl = WBCLocomotion.from_cache()
        for _ in range(50):  # one second to settle the stance (setup, before the agent acts)
            env.step(ctrl)
        base = G1MobileBase(env, ctrl)
        tb = MobileToolbox(base, SpatialMemory(np.random.default_rng([spec["seed"], 2])),
                           navigator="rpp")  # fmt: skip
        planner = PosturePlanner(caps, PlannerConfig())
        body = G1ReachBody(env, ctrl, tb, planner)
        probe = ReachProbe(env, ctrl, planner.pm.anatomy.grip, scene_geom_categories(scene))
        probe.attach(scene.handle_center)
        recorder = None
        if video:
            recorder = attach_video(env, video, f"{spec['id']}_{system}")
        instruction = cfg["instruction"].format(color=scene.handle_color)
        if system == "agent":
            skill = ReachSkill(body)
        else:
            from experiments.g1_scripted_baseline import ScriptedBaselineReach

            skill = ScriptedBaselineReach(body)
        t_run = time.perf_counter()
        report = skill.run(instruction)
        run_wall = time.perf_counter() - t_run
        if recorder:
            recorder.close()
        stops = [(s.t, s.reason) for sup in body.supervisors for s in sup.stops]
        verdict = judge(probe, scene, report.verdict, body.backend.held, stops)
        ref = reference_for(spec, scene, caps, out)
        refused = report.verdict == "refused"
        verdict["reference_feasible"] = ref["feasible"]
        verdict["refusal_correct"] = bool(refused and not ref["feasible"])
        verdict["missed_opportunity"] = bool(refused and ref["feasible"])
        verdict["height_band"] = (
            "low" if spec["height"] < 0.55 else "high" if spec["height"] > 1.15 else "mid"
        )
        result.update(
            instruction=instruction,
            agent=to_jsonable(report),
            evaluator=to_jsonable(verdict),
            reference=to_jsonable(ref),
            execution_trace=to_jsonable(trace_summary(body, probe)),
            truth={  # evaluator only
                "handle_center": scene.handle_center,
                "handle_color": scene.handle_color,
                "fixture": to_jsonable(scene.fixture),
                "distractors": to_jsonable(scene.distractors),
                "params": to_jsonable(scene.params),
            },
            timing={
                "sim_s": float(env.time),
                "wall_s": time.perf_counter() - t_wall,
                "skill_wall_s": run_wall,
            },  # fmt: skip
            error=None,
        )
    except Exception:  # noqa: BLE001 - recorded as a failed episode, never silently dropped
        result.update(error=traceback.format_exc(),
                      timing={"wall_s": time.perf_counter() - t_wall})  # fmt: skip
    (out / system).mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(result), indent=1))
    return result


def attach_video(env, root: str, name: str):
    """Spectator-camera MP4 from true simulator state (evidence only); outside Git."""
    width, height = 640, 360
    r = mujoco.Renderer(env.model, height, width)
    pathlib.Path(root).mkdir(parents=True, exist_ok=True)
    path = pathlib.Path(root) / f"{name}.mp4"
    proc = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{width}x{height}", "-r", "25", "-i", "-", "-pix_fmt", "yuv420p",
         "-vcodec", "libx264", str(path)],
        stdin=subprocess.PIPE,
    )  # fmt: skip
    step = env.step
    n = [0]

    def stepped(controller):
        step(controller)
        n[0] += 1
        if n[0] % 2 == 0:  # 25 fps of the 50 Hz policy steps
            r.update_scene(env.data, camera="spectator")
            proc.stdin.write(r.render().tobytes())

    env.step = stepped

    class Rec:
        def close(self):
            proc.stdin.close()
            proc.wait()
            r.close()

    return Rec()


# -- summary --------------------------------------------------------------------------------------
def rate(k: int, n: int) -> dict:
    return {"k": k, "n": n, "rate": (k / n if n else None), "wilson95": wilson(k, n)}


def summarise(rows: list[dict], gates: dict) -> dict:
    ok = [r for r in rows if r.get("error") is None]
    errors = [r["spec"]["id"] for r in rows if r.get("error")]
    ev = lambda r: r["evaluator"]  # noqa: E731
    feas = [r for r in ok if ev(r)["reference_feasible"]]
    infeas = [r for r in ok if not ev(r)["reference_feasible"]]
    edge = [r for r in feas if r["spec"]["height"] < 0.55 or r["spec"]["height"] > 1.15]
    band = lambda r: ev(r)["height_band"]  # noqa: E731
    succ = lambda rs: sum(ev(r)["physical_success"] for r in rs)  # noqa: E731
    out = {
        "n_episodes": len(rows), "n_errors": len(errors), "error_ids": errors,
        "n_reference_feasible": len(feas), "n_reference_infeasible": len(infeas),
        "collapses": rate(sum(ev(r)["collapse"]["collapsed"] for r in ok), len(ok)),
        "collapse_by_criterion": {
            k: sum(bool(ev(r)["collapse"][k]) for r in ok)
            for k in ("tilt_latch", "undesired_ground_contact", "foot_support_loss",
                      "joint_limit_contact")  # fmt: skip
        },
        "joint_limit_contact_episodes": sum(ev(r)["joint_limit_contact_s"] > 0.2 for r in ok),
        "false_success_claims": rate(sum(ev(r)["false_success_claim"] for r in ok), len(ok)),
        "unresolved_protective_stops": sum(ev(r)["unresolved_protective_stop"] for r in ok),
        "episodes_with_protective_stop": sum(bool(ev(r)["protective_stops"]) for r in ok),
        "unintended_contact_episodes": sum(ev(r)["contacts"]["unintended_contact"] for r in ok),
        "success_reference_feasible": rate(succ(feas), len(feas)),
        "claimed_success_reference_feasible": rate(
            sum(ev(r)["claimed_success"] for r in feas), len(feas)),
        "success_reference_feasible_off_band": rate(succ(edge), len(edge)),
        "success_all": rate(succ(ok), len(ok)),
        "success_by_band": {
            b: rate(succ([r for r in ok if band(r) == b]), len([r for r in ok if band(r) == b]))
            for b in ("low", "mid", "high")
        },
        "success_feasible_by_band": {
            b: rate(succ([r for r in feas if band(r) == b]),
                    len([r for r in feas if band(r) == b]))
            for b in ("low", "mid", "high")
        },
        "refusals": sum(r["agent"]["verdict"] == "refused" for r in ok),
        "correct_refusals_on_infeasible": rate(
            sum(ev(r)["refusal_correct"] for r in infeas), len(infeas)),
        "missed_opportunities": sum(ev(r)["missed_opportunity"] for r in ok),
        "verdicts": {
            v: sum(r["agent"]["verdict"] == v for r in ok) for v in ("reached", "refused", "failed")
        },
        "median_sim_s": float(np.median([r["timing"]["sim_s"] for r in ok])) if ok else None,
        "median_wall_s": float(np.median([r["timing"]["wall_s"] for r in ok])) if ok else None,
    }  # fmt: skip
    g = gates

    def at_least(name: str, value: dict, threshold: float) -> dict:
        """A rate gate; with no cases (n = 0) it cannot be evaluated and is reported as such."""
        if value["n"] == 0:
            return {"threshold": threshold, "pass": None, "applicable": False, "value": value}
        return {"threshold": threshold, "pass": value["rate"] >= threshold, "applicable": True,
                "value": value}  # fmt: skip

    out["gates"] = {
        "zero_collapses": {"pass": out["collapses"]["k"] == 0, "applicable": True,
                           "value": out["collapses"]},  # fmt: skip
        "zero_false_success_claims": {
            "pass": out["false_success_claims"]["k"] == 0, "applicable": True,
            "value": out["false_success_claims"]},  # fmt: skip
        "success_on_reference_feasible_ge": at_least(
            "success", out["success_reference_feasible"], g["success_feasible_min"]),
        "success_off_band_feasible_ge": at_least(
            "off_band", out["success_reference_feasible_off_band"],
            g["success_feasible_off_band_min"]),
        "correct_refusal_on_infeasible_ge": at_least(
            "refusal", out["correct_refusals_on_infeasible"], g["correct_refusal_min"]),
    }  # fmt: skip
    return out


def summarize_set(cfg: dict, which: str, out: pathlib.Path) -> None:
    summary = {}
    for system in SYSTEMS:
        rows = [json.loads(p.read_text()) for p in sorted((out / system).glob("*.json"))]
        summary[system] = summarise(rows, cfg["gates"]) if rows else None
    (out / "summary.json").write_text(json.dumps(to_jsonable(summary), indent=1))
    for system, s in summary.items():
        if s is None:
            continue
        print(f"== {system}: {s['n_episodes']} episodes, {s['n_errors']} errors")
        for name, gate in s["gates"].items():
            tag = "n/a " if gate["pass"] is None else "PASS" if gate["pass"] else "FAIL"
            print(f"   {tag} {name} {gate['value']}")


def is_finished(path: pathlib.Path) -> bool:
    """A checkpoint counts only if the episode ran to the end: a crashed one is rerun."""
    try:
        return path.exists() and json.loads(path.read_text()).get("error") is None
    except (OSError, ValueError):
        return False


def git(*a) -> str:
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--set", choices=("dev", "heldout"), required=True)
    ap.add_argument("--system", choices=("agent", "scripted_baseline", "both"), default="both")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--only", nargs="*", help="episode ids")
    ap.add_argument("--out", help="override the results directory (smoke runs)")
    ap.add_argument("--video-dir", help="record spectator videos here (outside Git)")
    ap.add_argument("--video-ids", nargs="*", default=[])
    args = ap.parse_args()
    cfg = json.loads(pathlib.Path(args.config).read_text())
    out = ROOT / (args.out or cfg[args.set]["results_dir"])
    envelope = ROOT / cfg["envelope"]
    if sha256(envelope) != cfg["envelope_sha256"]:
        raise SystemExit(f"envelope {envelope} does not match the pinned SHA-256")
    if args.summarize:
        summarize_set(cfg, args.set, out)
        return
    sha = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain", "dream", "experiments", "tests")
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.json"
    resumes = out / "resumes.jsonl"
    launch = {
        "git_sha": sha, "dirty": dirty.splitlines(), "set": args.set, "system": args.system,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }  # fmt: skip
    with resumes.open("a") as f:  # every launch is recorded: the first run and each resume
        f.write(json.dumps(launch) + "\n")
    if not meta_path.exists():
        meta_path.write_text(json.dumps({
            "git_sha_at_start": sha, "dirty_paths_at_start": dirty.splitlines(),
            "config": cfg, "envelope": cfg["envelope"], "envelope_sha256": sha256(envelope),
            "python": platform.python_version(), "mujoco": mujoco.__version__,
            "numpy": np.__version__, "machine": platform.platform(),
            "odometry": "oracle-assisted (true increments + noise), dream/sim/g1_mobile.py",
            "perception": "classical colour-and-depth baseline, dream/perception/detect.py",
            "instruction_parsing": "structured goal parsing, not language understanding",
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, indent=1))  # fmt: skip
    systems = SYSTEMS if args.system == "both" else (args.system,)
    jobs = []
    for spec in specs(cfg, args.set):
        if args.only and spec["id"] not in args.only:
            continue
        for system in systems:
            if is_finished(out / system / f"{spec['id']}.json"):
                continue
            video = args.video_dir if spec["id"] in args.video_ids else None
            jobs.append((spec, system, cfg, str(out), sha, video))
    print(f"{len(jobs)} episodes to run -> {out}", flush=True)
    t0 = time.perf_counter()
    # One episode per worker process: each builds MuJoCo models and renderers, and memory on the
    # laptop is the limit (a long-lived worker grew to 3 GB and an allocation failed).
    kw = {"max_tasks_per_child": 1} if sys.version_info >= (3, 11) else {}
    with ProcessPoolExecutor(args.workers, **kw) as pool:
        for i, r in enumerate(pool.map(run_episode, jobs), 1):
            e = r.get("evaluator")
            line = (f"[{i}/{len(jobs)} {time.perf_counter() - t0:.0f}s] {r['system']} "
                    f"{r['spec']['id']} h={r['spec']['height']:.2f}: ")  # fmt: skip
            if r.get("error"):
                line += "ERROR " + r["error"].strip().splitlines()[-1]
            else:
                line += (f"{r['agent']['verdict']} true_d={e['true_distance_m'] * 1000:.0f}mm "
                         f"success={e['physical_success']} feasible={e['reference_feasible']} "
                         f"collapse={e['collapse']['collapsed']}")  # fmt: skip
            print(line, flush=True)
    summarize_set(cfg, args.set, out)


if __name__ == "__main__":
    main()
