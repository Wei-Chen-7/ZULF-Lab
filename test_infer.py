"""Tests for the inference layer: priors, line merging, peak summaries.

These cover everything that does not require a trained network, so they run in
about a second and can gate every commit.
"""

import numpy as np
import pytest

import zulf_forward as zf
import zulf_infer as zi


GH, GC = zf.NUCLEI["1H"], zf.NUCLEI["13C"]


# ---------------------------------------------------------------------------
# merge_lines
# ---------------------------------------------------------------------------
def test_merge_combines_unresolvable_components():
    """Two lines closer than a linewidth become one, with summed amplitude."""
    f = np.array([100.0, 100.0000012, 200.0])
    a = np.array([0.3 + 0j, 0.4 + 0j, 0.5 + 0j])
    mf, ma = zi.merge_lines(f, a, width_hz=0.01)
    assert len(mf) == 2
    assert np.isclose(mf[0], 100.0, atol=1e-5)
    assert np.isclose(np.abs(ma[0]), 0.7)
    assert np.isclose(mf[1], 200.0)


def test_merge_keeps_resolvable_components():
    f = np.array([100.0, 100.5])
    a = np.array([1.0 + 0j, 1.0 + 0j])
    mf, _ = zi.merge_lines(f, a, width_hz=0.01)
    assert len(mf) == 2


def test_merge_uses_amplitude_weighted_centre():
    f = np.array([100.0, 100.002])
    a = np.array([3.0 + 0j, 1.0 + 0j])
    mf, _ = zi.merge_lines(f, a, width_hz=0.01)
    assert np.isclose(mf[0], (3 * 100.0 + 1 * 100.002) / 4)


def test_merge_handles_empty_input():
    f, a = zi.merge_lines(np.array([]), np.array([]), 0.01)
    assert len(f) == 0 and len(a) == 0


# ---------------------------------------------------------------------------
# peak_summary
# ---------------------------------------------------------------------------
def test_summary_has_the_documented_layout():
    p = zi.InferenceProblem()
    x = p.simulate_one(np.array([222.2, 1.0, 55.0, 12.0]), noisy=False)
    assert len(x) == 9 == p.x_dim()
    assert np.isclose(x[-1], np.log10(1 / (np.pi * 12.0)))   # log10 width


def test_high_frequency_offsets_track_J_exactly():
    """Shifting J shifts every high-frequency offset by the same amount.

    This is what makes J readable from the summary: the multiplet is centred on
    J regardless of the residual field.
    """
    p = zi.InferenceProblem()
    base = p.simulate_one(np.array([222.2, 1.0, 55.0, 12.0]), noisy=False)
    moved = p.simulate_one(np.array([225.0, 1.0, 55.0, 12.0]), noisy=False)
    hi = slice(2, 5)
    assert np.allclose(moved[hi] - base[hi], 2.8, atol=1e-6)
    # the low-frequency (Larmor) line must NOT move with J
    assert np.isclose(moved[0], base[0], atol=1e-9)


def test_zero_field_gives_a_single_line_at_J():
    p = zi.InferenceProblem()
    x = p.simulate_one(np.array([222.2, 0.0, 0.0, 12.0]), noisy=False)
    assert np.isclose(x[2], 0.0, atol=1e-9)          # first high line at J
    assert x[5] > zi._LOGAMP_FLOOR + 1                # and it is present
    assert np.isclose(x[6], zi._LOGAMP_FLOOR)         # others absent
    assert np.isclose(x[7], zi._LOGAMP_FLOOR)
    assert np.isclose(x[1], zi._LOGAMP_FLOOR)         # no Larmor line either


def test_transverse_field_floors_the_centre_line():
    """At theta = 90 the centre component is forbidden, so a slot is empty."""
    p = zi.InferenceProblem()
    x = p.simulate_one(np.array([222.2, 1.0, 90.0, 12.0]), noisy=False)
    present = np.array([x[5], x[6], x[7]]) > zi._LOGAMP_FLOOR + 1
    assert present.sum() == 2, "expected a doublet at perpendicular field"


def test_low_frequency_line_sits_at_the_mean_larmor_frequency():
    p = zi.InferenceProblem()
    B_nT = 1.5
    x = p.simulate_one(np.array([222.2, B_nT, 55.0, 12.0]), noisy=False)
    mean_larmor = 0.5 * (GH + GC) * B_nT * 1e-3      # nT -> uT, in Hz
    assert np.isclose(x[0], mean_larmor, rtol=1e-6)


def test_absent_lines_are_padded_at_the_amplitude_floor():
    p = zi.InferenceProblem()
    x = p.simulate_one(np.array([222.2, 0.0, 0.0, 12.0]), noisy=False)
    assert np.isclose(x[3], 0.0) and np.isclose(x[6], zi._LOGAMP_FLOOR)


# ---------------------------------------------------------------------------
# priors and the simulator
# ---------------------------------------------------------------------------
def test_prior_samples_land_inside_the_box():
    p = zi.InferenceProblem(J_center=222.2, J_half=3.0)
    th = p.sample_prior(500)
    assert th.shape == (500, 4)
    assert np.all(th >= p.low) and np.all(th <= p.high)
    assert np.isclose(p.low[0], 219.2) and np.isclose(p.high[0], 225.2)


def test_prior_is_narrow_because_predictors_are_good():
    """Ref [18]: 1J_CH predicted to ~1 Hz, which narrows the search a lot."""
    p = zi.InferenceProblem()
    assert (p.high[0] - p.low[0]) <= 10.0             # not the 0-300 Hz band


def test_noise_scale_matches_sigma_f():
    p = zi.InferenceProblem(sigma_f=2e-3, seed=1)
    th = np.array([222.9, 1.0, 55.0, 12.0])
    clean = p.simulate_one(th, noisy=False)
    draws = np.stack([p.simulate_one(th) for _ in range(400)])
    scatter = draws[:, 2].std()                       # a high-frequency slot
    assert np.isclose(scatter, 2e-3, rtol=0.25)
    assert np.isclose(draws[:, 2].mean(), clean[2], atol=5e-4)


def test_simulate_is_batched_and_finite():
    p = zi.InferenceProblem()
    X = p.simulate(p.sample_prior(64))
    assert X.shape == (64, 9)
    assert np.all(np.isfinite(X))


def test_larger_systems_are_refused_for_now():
    with pytest.raises(NotImplementedError):
        zi.InferenceProblem(system="methanol")
