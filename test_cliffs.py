"""Tests for the resolution-cliff diagnostic, and a regression test for the bug.

The T2 cliff is the important one here. The merge threshold used to be the
model's own linewidth, which made the observation model discontinuous in T2:
at the reference parameters a 0.1 s change merged three multiplet lines into
one and moved the log-likelihood by 5e5. ``test_no_cliff_in_T2`` is the guard
against that coming back.
"""

import numpy as np
import pytest

import zulf_infer as zi
import resolution_cliffs as rc


@pytest.fixture(scope="module")
def problem():
    return zi.InferenceProblem(seed=0)


@pytest.fixture(scope="module")
def theta(problem):
    return np.array([problem.J_center + 0.7, 1.0, 55.0, 12.0])


# -- the fix -----------------------------------------------------------------
def test_merge_width_defaults_to_the_linewidth_at_the_T2_prior_centre(problem):
    expected = 1.0 / (np.pi * np.mean(problem.T2_range))
    assert problem.merge_hz == pytest.approx(expected)


def test_merge_width_is_independent_of_the_parameters(problem, theta):
    """The whole point: it is an instrument constant, not a function of theta."""
    before = problem.merge_hz
    for T2 in (3.0, 12.0, 40.0):
        t = theta.copy()
        t[3] = T2
        problem.simulate_one(t, noisy=False)
        assert problem.merge_hz == before


def test_no_cliff_in_T2(problem, theta):
    """Regression: T2 used to carry a 5e5 discontinuity in the log-likelihood."""
    loc, _, _ = rc.cliff_scan(problem, theta, 3)
    assert len(loc) == 0


def test_log_likelihood_is_smooth_in_T2(problem, theta):
    """A second, sharper form of the same guard: no jump anywhere in the prior."""
    x = problem.simulate_one(theta, noisy=False)
    grid = np.linspace(problem.low[3], problem.high[3], 400)
    vals = []
    for v in grid:
        t = theta.copy()
        t[3] = v
        vals.append(-problem.log_likelihood(x, t)[0])
    jumps = np.abs(np.diff(vals))
    assert jumps.max() < 50 * np.median(jumps)


def test_J_direction_is_clean(problem, theta):
    """The parameter being measured must not sit on a discontinuous axis."""
    loc, _, _ = rc.cliff_scan(problem, theta, 0)
    assert len(loc) == 0


# -- the diagnostic ----------------------------------------------------------
def test_n_resolved_drops_when_the_resolution_is_coarsened(problem, theta):
    fine = rc.n_resolved(problem, theta)
    coarse_prob = zi.InferenceProblem(seed=0, merge_hz=10.0)
    assert rc.n_resolved(coarse_prob, theta) < fine


def test_n_resolved_never_exceeds_the_raw_line_count(problem, theta):
    import zulf_forward as zf
    f, _ = zf.line_list(problem.sys,
                        B=zf.field_vector(theta[1] * 1e-3, theta[2]),
                        J=problem.J_matrix(theta))
    assert 1 <= rc.n_resolved(problem, theta) <= len(f)


def test_cliff_scan_finds_the_field_cliff(problem, theta):
    """The multiplet becomes resolvable once the field splits it past merge_hz."""
    loc, _, _ = rc.cliff_scan(problem, theta, 1)
    assert len(loc) >= 1
    assert np.any(np.abs(loc - 0.556) < 0.05)


def test_adjacent_cells_are_reported_as_one_band(problem, theta):
    """Near theta = 0 the summary is unstable across a stretch, not at a point."""
    loc, grid, step = rc.cliff_scan(problem, theta, 2)
    dx = grid[1] - grid[0]
    # no two reported locations may be adjacent grid cells
    if len(loc) > 1:
        assert np.min(np.diff(np.sort(loc))) > 1.5 * dx


def test_cliff_scan_returns_a_step_profile_of_the_right_length(problem, theta):
    loc, grid, step = rc.cliff_scan(problem, theta, 1, n=120)
    assert len(grid) == 120 and len(step) == 119
    assert np.all(step >= 0)


def test_distance_to_cliff_is_a_fraction_of_the_prior_box(problem, theta):
    d = rc.distance_to_cliff(problem, theta, n=200)
    assert 0.0 <= d <= 1.0


def test_distance_to_cliff_is_small_next_to_a_known_cliff(problem, theta):
    near = theta.copy()
    near[1] = 0.556                      # sitting on the field cliff
    far = theta.copy()
    far[1] = 1.5
    assert rc.distance_to_cliff(problem, near, n=200) < \
        rc.distance_to_cliff(problem, far, n=200)


def test_distance_is_infinite_when_nothing_jumps(problem, theta, monkeypatch):
    monkeypatch.setattr(rc, "cliff_scan",
                        lambda *a, **k: (np.array([]), np.zeros(2), np.zeros(1)))
    assert not np.isfinite(rc.distance_to_cliff(problem, theta))


# -- the cache key -----------------------------------------------------------
def test_signature_tracks_the_merge_width():
    a = zi.InferenceProblem(seed=0)
    b = zi.InferenceProblem(seed=0, merge_hz=0.05)
    assert a.summary_signature() != b.summary_signature()


def test_signature_ignores_the_random_seed():
    assert zi.InferenceProblem(seed=0).summary_signature() == \
        zi.InferenceProblem(seed=7).summary_signature()


def test_signature_tracks_the_prior_and_the_system():
    base = zi.InferenceProblem(seed=0)
    assert zi.InferenceProblem(seed=0, J_half=1.0).summary_signature() != \
        base.summary_signature()
    assert zi.InferenceProblem(seed=0, system="glycine").summary_signature() != \
        base.summary_signature()
