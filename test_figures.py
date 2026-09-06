"""Tests for the figure helpers.

The plotting itself is not tested -- a figure is judged by eye. What is tested
is the arithmetic underneath it, because a weighted density or a weighted
covariance that is quietly wrong produces a figure that looks fine and says
something false.
"""

import numpy as np
import pytest

import make_figure2 as f2
import make_figure4 as f4


# -- weighted density --------------------------------------------------------
def test_density_integrates_to_one():
    rng = np.random.default_rng(0)
    v = rng.normal(3.0, 0.5, 4000)
    w = np.ones_like(v)
    grid = np.linspace(0.0, 6.0, 2000)
    d = f4._density(v, w, grid)
    assert np.trapezoid(d, grid) == pytest.approx(1.0, rel=0.02)


def test_density_recovers_a_known_gaussian():
    rng = np.random.default_rng(1)
    v = rng.normal(0.0, 1.0, 20000)
    grid = np.linspace(-4.0, 4.0, 800)
    d = f4._density(v, np.ones_like(v), grid)
    truth = np.exp(-0.5 * grid ** 2) / np.sqrt(2 * np.pi)
    assert np.max(np.abs(d - truth)) < 0.02


def test_density_respects_the_weights():
    """Two clusters; the weights should move the density onto one of them."""
    v = np.concatenate([np.full(500, -2.0), np.full(500, 2.0)])
    grid = np.linspace(-5.0, 5.0, 400)
    w = np.concatenate([np.full(500, 1.0), np.full(500, 0.01)])
    d = f4._density(v, w, grid)
    assert grid[np.argmax(d)] == pytest.approx(-2.0, abs=0.2)
    flipped = f4._density(v, w[::-1], grid)
    assert grid[np.argmax(flipped)] == pytest.approx(2.0, abs=0.2)


def test_density_is_bimodal_for_two_separated_clusters():
    rng = np.random.default_rng(2)
    v = np.concatenate([rng.normal(-3, 0.3, 3000), rng.normal(3, 0.3, 3000)])
    grid = np.linspace(-6, 6, 600)
    d = f4._density(v, np.ones_like(v), grid)
    centre = d[np.argmin(np.abs(grid))]
    assert centre < 0.05 * d.max()          # a real dip between the modes


def test_density_handles_a_single_dominant_weight():
    v = np.linspace(0.0, 1.0, 100)
    w = np.zeros(100)
    w[42] = 1.0
    d = f4._density(v, w, np.linspace(0.0, 1.0, 200))
    assert np.all(np.isfinite(d)) and d.max() > 0


# -- weighted covariance -----------------------------------------------------
def test_wcov_matches_numpy_for_equal_weights():
    rng = np.random.default_rng(3)
    s = rng.multivariate_normal([0, 1], [[2.0, 0.7], [0.7, 1.0]], 5000)
    got = f2._wcov(s, np.ones(len(s)))
    want = np.cov(s.T, bias=True)
    assert np.allclose(got, want, rtol=1e-8, atol=1e-8)


def test_wcov_is_symmetric_and_positive_semidefinite():
    rng = np.random.default_rng(4)
    s = rng.normal(size=(400, 3))
    w = rng.exponential(size=400)
    c = f2._wcov(s, w)
    assert np.allclose(c, c.T)
    assert np.all(np.linalg.eigvalsh(c) > -1e-12)


def test_wcov_is_invariant_to_weight_normalization():
    rng = np.random.default_rng(5)
    s = rng.normal(size=(300, 2))
    w = rng.exponential(size=300)
    assert np.allclose(f2._wcov(s, w), f2._wcov(s, 1000.0 * w))


def test_wcov_recovers_a_known_correlation():
    rng = np.random.default_rng(6)
    s = rng.multivariate_normal([0, 0], [[1.0, 0.8], [0.8, 1.0]], 20000)
    c = f2._wcov(s, np.ones(len(s)))
    rho = c[0, 1] / np.sqrt(c[0, 0] * c[1, 1])
    assert rho == pytest.approx(0.8, abs=0.02)


def test_density_bandwidth_floor_keeps_a_spike_visible():
    """A collapsed weighted spread must not render as an empty panel."""
    v = np.linspace(0.0, 1.0, 100)
    w = np.zeros(100)
    w[42] = 1.0
    grid = np.linspace(0.0, 1.0, 200)
    d = f4._density(v, w, grid)
    assert np.trapezoid(d, grid) == pytest.approx(1.0, rel=0.1)
    assert grid[np.argmax(d)] == pytest.approx(v[42], abs=0.02)
