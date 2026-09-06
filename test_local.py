"""Tests for the local least-squares baseline.

The baseline exists to be beaten, which makes it easy to under-implement by
accident. These tests pin the properties that make it a fair opponent: it
converges to the true minimum from a good start, its curvature error bars
reproduce the analytically known information floor, and the box penalty does
not distort the interior of the prior.
"""

import numpy as np
import pytest

import zulf_infer as zi
import local_baseline as lb


@pytest.fixture(scope="module")
def problem():
    return zi.InferenceProblem(seed=0)


@pytest.fixture(scope="module")
def noiseless(problem):
    """A noiseless observation, so the minimum sits exactly at the truth."""
    theta = np.array([problem.J_center + 0.7, 1.0, 55.0, 12.0])
    return theta, problem.simulate_one(theta, noisy=False)


# -- objective ---------------------------------------------------------------
def test_nll_is_minimized_at_the_truth_for_noiseless_data(problem, noiseless):
    theta, x = noiseless
    u = (theta - problem.low) / (problem.high - problem.low)
    f0 = lb._nll_u(problem, x, u)
    rng = np.random.default_rng(0)
    for _ in range(20):
        du = rng.normal(0, 1e-3, size=len(u))
        assert lb._nll_u(problem, x, u + du) >= f0 - 1e-9


def test_box_penalty_is_zero_inside_and_grows_outside(problem, noiseless):
    _, x = noiseless
    u = np.full(len(problem.low), 0.5)
    inside = lb._nll_u(problem, x, u)
    u_out = u.copy()
    u_out[0] = 1.1                                  # 0.1 outside the box
    assert lb._nll_u(problem, x, u_out) > inside + 0.5 * lb._BOX_PENALTY * 0.01


def test_simplex_reflects_off_the_far_wall():
    s = lb._simplex(np.array([0.99, 0.5]), 0.05)
    assert s.shape == (3, 2)
    assert np.all(s >= 0.0) and np.all(s <= 1.0)
    assert s[1, 0] == pytest.approx(0.94)           # stepped inward, not out


# -- convergence -------------------------------------------------------------
def test_fit_recovers_a_noiseless_truth_from_the_prior_centre(problem, noiseless):
    theta, x = noiseless
    centre = 0.5 * (problem.low + problem.high)
    fit = lb.local_fit(problem, x, centre)
    # J to well under a millihertz; the nuisances to a few percent of their prior
    assert abs(fit["theta"][0] - theta[0]) < 1e-3
    span = problem.high - problem.low
    assert np.all(np.abs(fit["theta"] - theta) < 0.05 * span)


def test_restarts_actually_help(problem, noiseless):
    theta, x = noiseless
    centre = 0.5 * (problem.low + problem.high)
    one = lb.local_fit(problem, x, centre, n_restarts=1)
    many = lb.local_fit(problem, x, centre, n_restarts=6)
    assert many["nll"] <= one["nll"] + 1e-9
    assert abs(many["theta"][0] - theta[0]) <= abs(one["theta"][0] - theta[0]) + 1e-9


def test_fit_reports_its_own_cost(problem, noiseless):
    _, x = noiseless
    fit = lb.local_fit(problem, x, 0.5 * (problem.low + problem.high))
    assert fit["nfev"] > 100
    assert fit["seconds"] > 0
    assert 1 <= fit["restarts"] <= 6


# -- error bars --------------------------------------------------------------
def test_curvature_sigma_on_J_matches_the_information_floor(problem, noiseless):
    """Three multiplet lines all move 1:1 with J, so sigma_J = sigma_f/sqrt(3).

    This is the one place where the right answer is known in closed form, so it
    is the strongest available check that the numerical Hessian is right.
    """
    theta, x = noiseless
    sigma, _, _, _ = lb.curvature_errors(problem, x, theta)
    expected = problem.sigma_f / np.sqrt(problem.n_high)
    assert sigma[0] == pytest.approx(expected, rel=0.10)


def test_hessian_is_symmetric_and_positive_definite_at_a_minimum(problem, noiseless):
    theta, x = noiseless
    _, cov, hess, _ = lb.curvature_errors(problem, x, theta)
    assert np.allclose(hess, hess.T, rtol=1e-6, atol=1e-6 * np.abs(hess).max())
    assert np.all(np.linalg.eigvalsh(hess) > 0)
    assert np.all(np.isfinite(cov))


def test_steps_are_auto_scaled_not_uniform(problem, noiseless):
    """A single fixed step cannot resolve both J and B; the tuner must adapt."""
    theta, x = noiseless
    _, _, _, steps = lb.curvature_errors(problem, x, theta)
    frac = steps / (problem.high - problem.low)
    assert frac.max() / frac.min() > 5


def test_sigma_is_finite_for_every_parameter(problem, noiseless):
    theta, x = noiseless
    sigma, _, _, _ = lb.curvature_errors(problem, x, theta)
    assert np.all(np.isfinite(sigma)) and np.all(sigma > 0)


# -- multistart and clustering ----------------------------------------------
def test_multistart_returns_one_fit_per_start(problem, noiseless):
    _, x = noiseless
    fits = lb.multistart(problem, x, n_starts=4, seed=1, verbose=False,
                         n_restarts=2)
    assert len(fits) == 4
    assert all(problem.in_prior(f["theta"][None, :])[0] for f in fits)
    starts = np.array([f["theta0"] for f in fits])
    assert len(np.unique(starts[:, 0])) == 4       # genuinely different starts


def test_cluster_minima_groups_by_value_and_orders_by_depth():
    fits = [dict(theta=np.array([100.0]), nll=5.0),
            dict(theta=np.array([100.0002]), nll=5.1),
            dict(theta=np.array([102.0]), nll=1.0),
            dict(theta=np.array([102.0001]), nll=1.2)]
    cl = lb.cluster_minima(fits, index=0, tol=1e-3)
    assert len(cl) == 2
    assert cl[0]["value"] == pytest.approx(102.0, abs=1e-3)   # deepest first
    assert cl[0]["delta_nll"] == 0.0
    assert cl[1]["delta_nll"] == pytest.approx(4.0)
    assert cl[0]["n"] == 2 and cl[1]["n"] == 2
    assert sum(c["frac"] for c in cl) == pytest.approx(1.0)


def test_cluster_minima_keeps_one_cluster_when_all_agree():
    fits = [dict(theta=np.array([7.0 + 1e-6 * k]), nll=1.0 + k)
            for k in range(5)]
    cl = lb.cluster_minima(fits, index=0, tol=1e-3)
    assert len(cl) == 1 and cl[0]["n"] == 5


# -- curvature report --------------------------------------------------------
def test_curvature_report_finds_no_flat_direction_on_a_clean_problem(
        problem, noiseless):
    theta, x = noiseless
    _, _, hess, _ = lb.curvature_errors(problem, x, theta)
    rep = lb.curvature_report(hess, problem.param_names, problem.prior_span())
    assert not rep["singular"]
    assert rep["flat"] == []


def test_curvature_report_scales_by_the_prior_not_by_the_largest_eigenvalue():
    """Raw eigenvalues carry their parameters' units, so they cannot be compared.

    A diagonal Hessian in which every parameter is pinned to 1% of its prior is
    well determined in all four directions, even though the raw eigenvalues
    span twelve orders of magnitude.
    """
    span = np.array([6.0, 2.0, 90.0, 37.0])
    hess = np.diag((100.0 / span) ** 2)          # sigma = span/100 everywhere
    rep = lb.curvature_report(hess, list("abcd"), span)
    assert not rep["singular"]
    assert np.allclose(rep["eigenvalues"], 1e4)


def test_curvature_report_names_the_flat_parameter():
    span = np.array([6.0, 30.0, 2.0])
    hess = np.diag([1e6, 1e-14, 1e4])            # the middle one is flat
    rep = lb.curvature_report(hess, ["J_CH", "J_HH", "B_nT"], span)
    assert rep["singular"]
    assert len(rep["flat"]) == 1
    assert rep["flat"][0]["dominant"] == "J_HH"
    assert rep["flat"][0]["loading"] == pytest.approx(1.0)


def test_flat_direction_gives_no_usable_error_bar_and_does_not_say_so(problem):
    """The failure this whole case exists to show.

    J_HH inside the methyl group moves no line, so the Hessian is singular in
    that direction and inverting it produces either an error bar orders of
    magnitude wider than the prior or a silent NaN -- which one depends on
    whether roundoff pushed the near-zero eigenvalue below zero. Neither is an
    error, and neither tells the user the data never constrained J_HH at all.
    Only ``curvature_report`` does.
    """
    prob = zi.InferenceProblem(system="methanol", seed=0)
    theta = np.array([141.0, -12.4, 1.0, 55.0, 12.0])
    span = prob.prior_span()
    j_hh = prob.param_names.index("J_HH")
    j_ch = prob.param_names.index("J_CH")

    for noisy in (False, True):
        prob.rng = np.random.default_rng(0)
        x = prob.simulate_one(theta, noisy=noisy)
        sigma, _, hess, _ = lb.curvature_errors(prob, x, theta)
        s = sigma[j_hh]
        assert np.isnan(s) or s > 1e3 * span[j_hh]
        assert 0 < sigma[j_ch] < 1e-3 * span[j_ch]     # the same fit pins J_CH
        rep = lb.curvature_report(hess, prob.param_names, span)
        assert rep["singular"]
        assert [f["dominant"] for f in rep["flat"]] == ["J_HH"]
