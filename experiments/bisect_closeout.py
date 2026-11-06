"""One-off diagnostic runner for the room-explorer bisect (not part of the product).

    python experiments/bisect_closeout.py JOBS.json --out DIR [--workers N]

Each job {tree, kind, seed, variant} runs one gain-arm episode in its own subprocess, using the
dream/ tree and the room runner copied byte-for-byte from commit <tree> into
experiments/bisect_trees/<tree>/ (MANIFEST.json holds each copy's git tree hash, compared with
``git rev-parse <tree>:dream``). The runner's own ``episode`` function is called unchanged.
This file only (a) puts the tree on sys.path, (b) links third_party/ into it, (c) gives
onnxruntime sessions an explicit CPU provider where the tested commit predates that (cf47597
did this inside dream/; the guest may carry onnxruntime-gpu), (d) captures the tool-call
steps, toolbox events and candidate memory for the record, and (e) for variant jx / jy shifts
the start pose by 1 mm. It never edits anything under dream/.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
TREES = HERE / "bisect_trees"
CFG = {"time_budget_s": 240, "max_steps": 60}
VARIANTS = {"base": (0.0, 0.0), "jx": (0.001, 0.0), "jy": (0.0, 0.001)}


def jsonable(v):
    import numpy as np

    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return jsonable(v.tolist())
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, (np.integer, np.bool_)):
        return v.item()
    if isinstance(v, (str, int, bool)) or v is None:
        return v
    return repr(v)[:200]


def link_third_party(root: pathlib.Path) -> None:
    link = root / "third_party"
    if link.exists():
        return
    try:
        os.symlink(ROOT / "third_party", link, target_is_directory=True)
    except FileExistsError:
        pass
    except OSError:  # Windows without symlink rights: a directory junction
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(ROOT / "third_party")],
                       check=True, stdout=subprocess.DEVNULL)  # fmt: skip


def child(tree: str, kind: str, seed: int, variant: str, out: pathlib.Path) -> None:
    root = TREES / tree
    link_third_party(root)
    sys.path[:] = [str(root)] + [
        p for p in sys.path if p and not pathlib.Path(p).resolve().is_relative_to(HERE)
    ]
    sys.modules.pop("experiments", None)

    import onnxruntime as ort

    class Session(ort.InferenceSession):
        def __init__(self, path, sess_options=None, providers=None, **kw):
            super().__init__(path, sess_options, providers=providers or ["CPUExecutionProvider"],
                             **kw)  # fmt: skip

    ort.InferenceSession = Session

    spec = importlib.util.spec_from_file_location(
        "runner_under_test", root / "experiments" / "g1_explore_room.py"
    )
    runner = importlib.util.module_from_spec(spec)
    sys.modules["runner_under_test"] = runner
    spec.loader.exec_module(runner)
    import dream

    assert pathlib.Path(dream.__file__).resolve().is_relative_to(root.resolve()), dream.__file__

    dx, dy = VARIANTS[variant]
    if dx or dy:
        env_cls = runner.G1MobileEnv
        orig_reset = env_cls.reset

        def reset(self, x=0.0, y=0.0, yaw=0.0):
            return orig_reset(self, x + dx, y + dy, yaw)

        env_cls.reset = reset

    cap: dict = {"events": []}
    tb_cls, agent_cls = runner.MobileToolbox, runner.TaskAgent

    class TB(tb_cls):
        def __init__(self, base, memory, on_event=None, **kw):
            cap["tb"] = self

            def wrapped(kind_, text, **data):
                cap["events"].append({"t": round(float(base.time), 2), "kind": kind_,
                                      "text": str(text)[:160], "data": jsonable(data)})  # fmt: skip
                if on_event:
                    on_event(kind_, text, **data)

            super().__init__(base, memory, on_event=wrapped, **kw)

    class Agent(agent_cls):
        def run(self, *a, **k):
            ep = super().run(*a, **k)
            cap["ep"] = ep
            return ep

    runner.MobileToolbox, runner.TaskAgent = TB, Agent
    t0 = time.perf_counter()
    row = runner.episode((kind, seed, "gain", CFG, None, tree))
    ep, tb = cap.get("ep"), cap.get("tb")
    steps = []
    for s in (ep.steps if ep else []):
        res = s.result
        data = jsonable(res.data) if isinstance(res.data, dict) else {}
        keep = {k: data[k] for k in ("error_m", "replans", "handle_observed",
                                     "handle_confirmed_close", "inspection_complete", "pose_note",
                                     "pose") if k in data}  # fmt: skip
        steps.append({"tool": s.call.name, "args": jsonable(s.call.args), "status": res.status,
                      "message": str(res.message)[:200], "data": keep})  # fmt: skip
    cands = []
    for c in (tb.memory.candidates if tb else []):
        cands.append({"id": c.id, "position": jsonable(c.position),
                      "handle": jsonable(c.handle), "handle_views": c.handle_views,
                      "sides_close": jsonable(sorted(c.sides_close))})  # fmt: skip
    truth = {}
    if tb is not None:
        env = tb.base.env
        x, y, yaw = env.true_pose()
        truth = {"final_true_pose": [float(x), float(y), float(yaw)],
                 "final_odometry": jsonable(env.odometry()),
                 "sim_time": float(env.time)}  # fmt: skip
        try:
            truth["true_handle_xy"] = jsonable(env.data.site("cabinet_handle_site").xpos[:2])
        except Exception:  # noqa: BLE001 - no cabinet in this room
            truth["true_handle_xy"] = None
    import mujoco

    rec = {"tree": tree, "kind": kind, "seed": seed, "variant": variant,
           "mujoco": mujoco.__version__, "onnxruntime": ort.__version__,
           "child_wall_s": time.perf_counter() - t0, "row": jsonable(row), "steps": steps,
           "events": cap["events"], "candidates": cands, "truth": truth}  # fmt: skip
    out.write_text(json.dumps(rec))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--child", nargs=5, metavar=("TREE", "KIND", "SEED", "VARIANT", "OUT"))
    args = ap.parse_args()
    if args.child:
        t, k, s, v, o = args.child
        child(t, k, int(s), v, pathlib.Path(o))
        return
    jobs = json.loads(pathlib.Path(args.jobs).read_text())
    for tree in {j["tree"] for j in jobs}:
        link_third_party(TREES / tree)
    out = pathlib.Path(args.out)
    (out / "episodes").mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    done = []

    def run(job: dict) -> dict:
        name = f"{job['tree']}__{job['kind']}_{job['seed']}__{job['variant']}"
        path = out / "episodes" / f"{name}.json"
        log = out / "episodes" / f"{name}.log"
        with open(log, "w") as f:
            proc = subprocess.run(
                [sys.executable, str(pathlib.Path(__file__).resolve()), args.jobs, "--out",
                 args.out, "--child", job["tree"], job["kind"], str(job["seed"]),
                 job["variant"], str(path)],
                stdout=f, stderr=subprocess.STDOUT, timeout=1800, cwd=str(ROOT),
            )  # fmt: skip
        ok = proc.returncode == 0 and path.exists()
        summary = {"name": name, "exit": proc.returncode}
        if ok:
            r = json.loads(path.read_text())["row"]
            summary.update(success=r["success"], claimed=r["claimed"], reason=r["reason"][:80],
                           sim_s=r["sim_s"], path_m=r["path_m"])  # fmt: skip
        done.append(summary)
        print(f"[{len(done)}/{len(jobs)} {time.perf_counter() - t0:.0f}s] {json.dumps(summary)}",
              flush=True)
        return summary

    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(run, jobs))
    (out / "index.json").write_text(json.dumps(results, indent=2))
    print("finished", sum(r["exit"] == 0 for r in results), "of", len(results), flush=True)


if __name__ == "__main__":
    main()
