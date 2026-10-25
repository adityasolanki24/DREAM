"""Benchmark figures (matplotlib, static PNG)."""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Categorical slots 1-4 of the validated reference palette, in fixed order per strategy.
STRATEGY_COLORS = {
    "eig": "#2a78d6",
    "random": "#eb6834",
    "goal_directed": "#1baf7a",
    "axis_sweep": "#eda100",
}
STRATEGY_LABELS = {
    "eig": "EIG (active)",
    "random": "Random",
    "goal_directed": "Goal-directed pull",
    "axis_sweep": "Axis sweep",
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel("probes executed", color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def plot_summary(logs: list[dict], summary: dict, budget: int, out: pathlib.Path) -> None:
    overall = summary["overall"]
    panels = [
        ("id_rate_curve", "Mechanism identified", "fraction of episodes"),
        ("task_success_curve", "Opening task succeeds", "fraction of episodes"),
        ("heldout_err_mm_curve", "Held-out prediction error", "mean handle error (mm)"),
        ("type_entropy_curve", "Uncertainty over joint type", "entropy (nats)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    steps = np.arange(budget + 1)
    for ax, (key, title, ylabel) in zip(axes.flat, panels, strict=True):
        for strat in STRATEGY_COLORS:
            if strat not in overall:
                continue
            y = np.array([np.nan if v is None else v for v in overall[strat][key]])
            ax.plot(
                steps,
                y,
                color=STRATEGY_COLORS[strat],
                linewidth=2,
                marker="o",
                markersize=4,
                label=STRATEGY_LABELS[strat],
            )
        _style(ax, title, ylabel)
    axes[0, 0].legend(frameon=False, fontsize=9, labelcolor=INK)
    n = overall[next(iter(overall))]["n_episodes"]
    fig.suptitle(
        f"DREAM benchmark: {n} mechanisms per strategy, budget {budget}",
        x=0.01,
        ha="left",
        fontsize=12,
        color=INK,
    )
    fig.savefig(out / "curves.png", dpi=130)
    plt.close(fig)

    fams = list(summary["per_family"])
    strats = [s for s in STRATEGY_COLORS if s in overall]
    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    width = 0.8 / len(strats)
    for i, s in enumerate(strats):
        vals = [summary["per_family"][f][s]["mean_probes_to_id"] for f in fams]
        ax.bar(
            np.arange(len(fams)) + (i - (len(strats) - 1) / 2) * width,
            vals,
            width * 0.92,
            color=STRATEGY_COLORS[s],
            label=STRATEGY_LABELS[s],
        )
    ax.set_xticks(np.arange(len(fams)), fams)
    ax.axhline(budget + 1, color=MUTED, linewidth=1, linestyle="--")
    ax.text(len(fams) - 0.5, budget + 1.05, "never identified", color=MUTED, ha="right", fontsize=9)
    _style(ax, "Probes to stable identification (lower is better)", "mean probes")
    ax.set_xlabel("mechanism family", color=MUTED)
    ax.legend(frameon=False, fontsize=9, ncols=len(strats), labelcolor=INK)
    fig.savefig(out / "per_family.png", dpi=130)
    plt.close(fig)
