"""Watch DREAM experiment with a mechanism it has never seen.

    python experiments/demo.py --seed 3            # hidden mechanism family drawn from the seed
    python experiments/demo.py --family slider     # or choose the hidden family yourself

The agent is never told the family. Each step prints what the agent believes, what it predicts
the chosen probe will do, and what actually happened. Outputs in results/demo/<family>_<seed>/:
    demo.gif      rendered episode; translucent lines/arrows are sampled hypotheses
                  (blue: hinge axes, orange: slide directions)
    summary.png   type posterior, predicted vs observed motion, held-out prediction error
    log.json      the full episode log
Headless rendering uses OSMesa (apt install libosmesa6). --viewer opens MuJoCo's live viewer
on a machine with a display. --no-render skips the GIF.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

if "DISPLAY" not in os.environ:  # headless: render offscreen with OSMesa
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dream.agent.exploration import EIGSelector, RandomSelector, candidate_probes  # noqa: E402
from dream.agent.model import PRISMATIC, REVOLUTE, TYPE_NAMES  # noqa: E402
from dream.eval.episode import run_episode  # noqa: E402
from dream.sim.mechanisms import FAMILIES, sample_mechanism  # noqa: E402

TYPE_COLORS = {"fixed": "#1baf7a", "prismatic": "#eb6834", "revolute": "#2a78d6"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _hex_rgba(h: str, alpha: float) -> np.ndarray:
    return np.array([int(h[i : i + 2], 16) / 255 for i in (1, 3, 5)] + [alpha], dtype=np.float32)


class Recorder:
    """Renders frames and/or drives a live viewer, with the agent's hypotheses drawn in."""

    def __init__(
        self,
        frames: bool = True,
        live: bool = False,
        width: int = 480,
        height: int = 360,
        every: int = 50,
    ) -> None:
        self.record_frames = frames
        self.live = live
        self.viewer = None
        self.size = (height, width)
        self.every = every
        self.renderer = None
        self.n = 0
        self.frames: list[tuple[np.ndarray, dict]] = []
        self.hyps = None
        self.h0 = None
        self.caption = {"step": 0, "probs": np.full(3, 1 / 3), "text": "grasping the handle"}

    def hook(self, model, data) -> None:
        self.n += 1
        if self.n % self.every:
            return
        handle = data.site("handle").xpos.copy()
        if self.live:
            self._sync_viewer(model, data, handle)
        if not self.record_frames:
            return
        if self.renderer is None:
            self.renderer = mujoco.Renderer(model, *self.size)
        self.renderer.update_scene(data, camera="front")
        if self.hyps is not None:
            self._draw_hypotheses(self.renderer.scene, handle)
        self.frames.append((self.renderer.render().copy(), dict(self.caption)))

    def _sync_viewer(self, model, data, handle: np.ndarray) -> None:
        import time

        import mujoco.viewer

        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(model, data)
        with self.viewer.lock():
            self.viewer.user_scn.ngeom = 0
            if self.hyps is not None:
                self._draw_hypotheses(self.viewer.user_scn, handle)
        self.viewer.sync()
        time.sleep(self.every * model.opt.timestep)  # roughly real time

    def _draw_hypotheses(self, scn, handle: np.ndarray) -> None:
        h = self.hyps
        for i in range(len(h)):
            if scn.ngeom >= scn.maxgeom:
                return
            kind = int(h.kind[i])
            if kind == PRISMATIC:
                a, b = handle, handle + 0.12 * h.axis[i]
                gtype, width, color = mujoco.mjtGeom.mjGEOM_ARROW, 0.006, TYPE_COLORS["prismatic"]
            elif kind == REVOLUTE:
                rel = h.origin[i] - self.h0
                foot = self.h0 + rel - np.dot(rel, h.axis[i]) * h.axis[i]
                a, b = foot - 0.3 * h.axis[i], foot + 0.3 * h.axis[i]
                gtype, width, color = mujoco.mjtGeom.mjGEOM_CAPSULE, 0.003, TYPE_COLORS["revolute"]
            else:
                continue
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(
                g, gtype, np.zeros(3), np.zeros(3), np.eye(3).ravel(), _hex_rgba(color, 0.35)
            )
            mujoco.mjv_connector(g, gtype, width, a, b)
            scn.ngeom += 1


def _frame_figure(img: np.ndarray, cap: dict) -> np.ndarray:
    fig = plt.figure(figsize=(9, 3.6), dpi=90)
    ax_img = fig.add_axes([0.0, 0.0, 0.53, 1.0])
    ax_img.imshow(img)
    ax_img.axis("off")
    ax = fig.add_axes([0.64, 0.5, 0.33, 0.4])
    ax.barh(TYPE_NAMES, cap["probs"], color=[TYPE_COLORS[t] for t in TYPE_NAMES], height=0.6)
    ax.set_xlim(0, 1)
    ax.set_title(f"belief after {cap['step']} probes", fontsize=10, loc="left", color=INK)
    ax.tick_params(colors=MUTED, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.text(0.545, 0.03, cap["text"], fontsize=6.3, color=INK, va="bottom", family="monospace")
    fig.canvas.draw()
    out = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return out


def _narrate(step: dict, info: dict, h_before: np.ndarray) -> str:
    pred = info["pre_action_prediction"]
    disp = np.linalg.norm(pred[:, :3] - h_before, axis=1)
    p_move = float(np.mean(disp > 0.005))
    probe = step["probe"]
    lines = [
        f"probe {step['step']}: dir {np.round(probe['direction'], 2)} "
        f"{probe['distance'] * 100:.0f} cm, <= {probe['max_force']:.0f} N"
        + (f"  (EIG {info['eig']:.2f} nats)" if "eig" in info else ""),
        f"  predicted: P(moves > 5 mm) = {p_move:.2f}, median {1000 * np.median(disp):.0f} mm "
        f"[10-90%: {1000 * np.percentile(disp, 10):.0f}-{1000 * np.percentile(disp, 90):.0f}]",
        f"  observed:  moved {1000 * step['observed_displacement_m']:.0f} mm, "
        f"now {np.degrees(step['observed_rotation_rad']):.1f} deg from closed orientation",
        "  belief:    "
        + ", ".join(f"{t} {p:.3f}" for t, p in zip(TYPE_NAMES, step["type_probs"], strict=True)),
    ]
    s = step["belief"]
    if "axis" in s:
        lines.append(
            f"  best explanation: {s['map_type']} axis {np.round(s['axis'], 2)}"
            + (f" through {np.round(s['origin'], 3)}" if "origin" in s else "")
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--family", choices=sorted(FAMILIES), default=None)
    ap.add_argument("--strategy", choices=["eig", "random"], default="eig")
    ap.add_argument("--budget", type=int, default=6)
    ap.add_argument("--no-render", action="store_true", help="skip the GIF")
    ap.add_argument("--viewer", action="store_true", help="live MuJoCo viewer (needs a display)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    family = args.family or str(rng.choice(sorted(FAMILIES)))
    spec = sample_mechanism(family, rng)
    out = ROOT / "results" / "demo" / f"{family}_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    probes = candidate_probes()
    selector = EIGSelector(probes) if args.strategy == "eig" else RandomSelector(probes)
    rec = None
    if not args.no_render or args.viewer:
        rec = Recorder(frames=not args.no_render, live=args.viewer)

    def on_step(step, belief, info):
        text = _narrate(step, info, belief.history[-1].handle_before)
        print(text, flush=True)
        if rec is not None:
            rec.h0 = belief.h0
            rec.hyps = belief.sample(40)
            rec.caption = {
                "step": step["step"],
                "probs": np.array(step["type_probs"]),
                "text": text,
            }

    print(f"Hidden mechanism: (secret) {family}. The agent sees only a panel with a handle.\n")
    log = run_episode(
        spec,
        selector,
        args.budget,
        seed=args.seed,
        on_step=on_step,
        env_hook=None if rec is None else rec.hook,
    )
    (out / "log.json").write_text(json.dumps(log, indent=2))
    final = log["steps"][-1]
    print(
        f"\nTruth: {spec.joint_type} ({family}), axis {np.round(spec.axis, 2)}"
        + (f", hinge through {np.round(spec.origin, 3)}" if spec.joint_type == "revolute" else "")
    )
    print(
        f"Scored after {args.budget} probes: identified={final['identified']}, "
        f"axis error={final['axis_err_deg'] and round(final['axis_err_deg'], 1)} deg, "
        f"hinge-line error={final['line_err_m'] and round(1000 * final['line_err_m'], 1)} mm, "
        f"opening task success={final['task_success']}"
    )
    _plot_summary(log, out)
    if rec is not None and rec.frames:
        from PIL import Image

        imgs = [Image.fromarray(_frame_figure(f, c)) for f, c in rec.frames]
        imgs[0].save(out / "demo.gif", save_all=True, append_images=imgs[1:], duration=100, loop=0)
    print(f"\nWrote {out.relative_to(ROOT)}/")


def _plot_summary(log: dict, out: pathlib.Path) -> None:
    steps = log["steps"]
    t = np.arange(len(steps))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    probs = np.array([s["type_probs"] for s in steps])
    for k, name in enumerate(TYPE_NAMES):
        axes[0].plot(
            t,
            probs[:, k],
            color=TYPE_COLORS[name],
            linewidth=2,
            marker="o",
            markersize=4,
            label=name,
        )
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].set_title("Belief over joint type", loc="left", fontsize=11)
    obs = [1000 * s["observed_displacement_m"] for s in steps[1:]]
    pred_err = [1000 * s["pre_action_pred_err_m"] for s in steps[1:]]
    axes[1].bar(t[1:] - 0.2, obs, 0.38, color="#52514e", label="observed motion")
    axes[1].bar(
        t[1:] + 0.2, pred_err, 0.38, color="#2a78d6", label="prediction error (before acting)"
    )
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].set_title("Chosen probe: motion vs prediction error (mm)", loc="left", fontsize=11)
    axes[2].plot(
        t,
        [1000 * s["heldout_pos_err_m"] for s in steps],
        color="#2a78d6",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    axes[2].set_title("Held-out probe prediction error (mm)", loc="left", fontsize=11)
    for ax in axes:
        ax.set_xlabel("probes executed", color=MUTED)
        ax.grid(True, color=GRID, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.savefig(out / "summary.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
