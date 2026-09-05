"""Tests for the nested-sampling reference.

Small nlive and loose dlogz keep these quick; the full-quality run lives in
``nested_reference.main``.
"""

import numpy as np

import zulf_infer as zi
import nested_reference as nr


def _problem():
    prob = zi.InferenceProblem(seed=0)
    theta_true = np.array([prob.J_center + 0.7, 1.0, 55.0, 12.0])
    return prob, theta_true


def test_observation_matches_the_one_evaluate_builds():
    """The reference must run on the same data the network is scored on."""
    prob, tt = _problem()
    a = nr.make_observation(prob, tt, seed=0)
    b = nr.make_observation(prob, tt, seed=0)
    assert np.allclose(a, b), "observation must be reproducible"
    # and it must not disturb the problem's own rng stream
    prob2, _ = _problem()
    before = prob2.rng.normal()
    prob3, tt3 = _problem()
    nr.make_observation(prob3, tt3, seed=0)
    assert np.isclose(prob3.rng.normal(), before)


def test_nested_sampling_recovers_J_and_reports_evidence():
    prob, tt = _problem()
    x = nr.make_observation(prob, tt, seed=0)
    s, w, logz, logzerr, _ = nr.run_nested(prob, x, nlive=150, dlogz=0.5, seed=0)
    assert len(s) > 200 and np.isfinite(logz) and logzerr > 0
    mean_J = np.average(s[:, 0], weights=w)
    assert abs(mean_J - tt[0]) < 0.01, f"J mean {mean_J} vs {tt[0]}"


def test_nested_posterior_is_near_the_information_floor():
    prob, tt = _problem()
    x = nr.make_observation(prob, tt, seed=0)
    s, w, *_ = nr.run_nested(prob, x, nlive=150, dlogz=0.5, seed=0)
    lo, hi = zi.weighted_quantile(s[:, 0], [0.025, 0.975], w)
    width = (hi - lo) * 1e3
    floor = 2 * 1.96 * prob.sigma_f / np.sqrt(3) * 1e3
    assert 0.5 * floor < width < 2.0 * floor, f"{width:.2f} mHz vs floor {floor:.2f}"


def test_nested_sampling_shrinks_the_prior_a_lot():
    prob, tt = _problem()
    x = nr.make_observation(prob, tt, seed=0)
    s, w, *_ = nr.run_nested(prob, x, nlive=150, dlogz=0.5, seed=0)
    lo, hi = zi.weighted_quantile(s[:, 0], [0.025, 0.975], w)
    prior_width = prob.high[0] - prob.low[0]
    assert prior_width / (hi - lo) > 100
