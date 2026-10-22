import numpy as np
import pytest

from dream.agent.exploration import (
    EIGSelector,
    RandomSelector,
    axis_sweep_selector,
    candidate_probes,
    pairwise_mixture_mi,
)
from dream.eval.episode import run_episode
from dream.eval.summary import paired_permutation_test, probes_to_identification, summarise
from dream.sim.mechanisms import sample_mechanism


def _log(ids, strategy="eig", family="drawer", seed=0):
    return {
        "strategy": strategy,
        "family": family,
        "mechanism_seed": seed,
        "steps": [
            {
                "identified": i,
                "task_success": i,
                "heldout_pos_err_m": 0.01,
                "type_entropy": 0.5,
                "p_true_type": 0.5,
            }
            for i in ids
        ],
    }


def test_probes_to_identification_requires_staying_identified():
    assert probes_to_identification(_log([False, True, True])) == 1
    assert probes_to_identification(_log([False, True, False, True])) == 3
    assert probes_to_identification(_log([False, True, False])) is None


def test_summary_censors_unidentified_and_pairs_by_mechanism():
    logs = [
        _log([False, True, True], "eig", seed=0),
        _log([False, False, False], "random", seed=0),
        _log([False, False, True], "eig", seed=1),
        _log([False, False, True], "random", seed=1),
    ]
    s = summarise(logs, budget=2)
    assert s["overall"]["random"]["mean_probes_to_id"] == pytest.approx((3 + 2) / 2)
    cmp = s["comparisons"]["eig_vs_random"]
    assert cmp["n_pairs"] == 2 and cmp["eig_better"] == 1 and cmp["ties"] == 1


def test_permutation_test_detects_consistent_difference_only():
    rng = np.random.default_rng(0)
    assert paired_permutation_test(np.zeros(30), np.ones(30)) < 0.001
    assert paired_permutation_test(rng.normal(size=30), rng.normal(size=30)) > 0.05


def test_mixture_mi_zero_when_predictions_agree_and_capped_at_log_m():
    sigma = np.ones((1, 50, 2))
    assert pairwise_mixture_mi(np.zeros((1, 50, 2)), sigma)[0] == pytest.approx(0, abs=1e-9)
    far = np.arange(50.0)[None, :, None] * 100 * np.ones((1, 50, 2))
    assert pairwise_mixture_mi(far, sigma)[0] == pytest.approx(np.log(50), abs=1e-6)


def test_eig_prefers_probes_whose_outcomes_differ_across_hypotheses():
    """Toy check: two hypotheses identical under probe 0 and different under probe 1."""
    y = np.zeros((2, 64, 6))
    y[1, 32:, 0] = 0.05
    sigma = np.full((2, 64, 6), 0.003)
    eig = pairwise_mixture_mi(y, sigma)
    assert eig[0] == pytest.approx(0, abs=1e-9)
    assert eig[1] == pytest.approx(np.log(2), abs=1e-3)


def test_baselines_share_the_eig_candidate_set():
    probes = candidate_probes()
    assert len(probes) == 104
    assert RandomSelector(probes).probes is EIGSelector(probes).probes


@pytest.mark.slow
@pytest.mark.parametrize("family", ["fixed", "slider", "top_hinge"])
def test_closed_loop_episode_identifies_and_operates(family):
    spec = sample_mechanism(family, np.random.default_rng(11))
    log = run_episode(spec, EIGSelector(candidate_probes()), budget=6, seed=11, n_particles=400)
    final = log["steps"][-1]
    assert final["identified"], final
    assert final["task_success"]
    assert log["steps"][0]["type_entropy"] > final["type_entropy"]


@pytest.mark.slow
def test_scripted_baseline_runs_closed_loop():
    spec = sample_mechanism("drawer", np.random.default_rng(1))
    log = run_episode(spec, axis_sweep_selector(), budget=2, seed=1, n_particles=300)
    assert log["steps"][1]["probe"]["direction"] == [0.0, -1.0, 0.0]
