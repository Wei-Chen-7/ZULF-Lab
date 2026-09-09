"""Tests for the misspecified-data generators and the two detectors.

None of these load a trained network, so they stay fast. The one that matters
most is ``test_axis_scale_is_absorbed_by_rescaling``: it pins the claim that a
frequency-axis calibration error is *exactly* degenerate with rescaling
(J, |B|, 1/T2), which is why no residual-based check can ever see one.
"""

import numpy as np
import pytest

import zulf_infer as zi
import misspecification as ms


@pytest.fixture(scope="module")
def formic():
    p = zi.InferenceProblem(seed=0)
    th = 0.5 * (p.low + p.high)
    th[0] = p.J_center + 0.7
    return p, th


@pytest.fixture(scope="module")
def formaldehyde():
    p = zi.InferenceProblem(system="formaldehyde", seed=0)
    th = 0.5 * (p.low + p.high)
    th[0] = p.J_center + 0.7
    th[1] = 0.0
    return p, th


def _clean(prob, theta):
    f, a, w = ms.clean_lines(prob, theta)
    return ms._summary(prob, f, a, w)


# -- generators are no-ops at zero -------------------------------------------
def test_axis_scale_at_zero_is_the_clean_summary(formic):
    p, th = formic
    assert ms.gen_axis_scale(p, th, 0.0) == pytest.approx(_clean(p, th))


def test_drift_at_zero_is_the_clean_summary(formic):
    p, th = formic
    assert ms.gen_drift(p, th, 0.0) == pytest.approx(_clean(p, th))


def test_equivalence_at_zero_is_the_clean_summary(formaldehyde):
    p, th = formaldehyde
    assert ms.gen_equivalence(p, th, 0.0) == pytest.approx(_clean(p, th))


# -- the degeneracy that makes one systematic undetectable -------------------
def test_axis_scale_is_absorbed_by_rescaling(formic):
    """A mis-scaled frequency axis leaves no residual for any check to find.

    Multiplet lines sit at fixed multiples of J and the low-frequency line
    scales with the field, so scaling every frequency by 1+eps is reproduced
    by J -> J(1+eps), |B| -> |B|(1+eps), T2 -> T2/(1+eps). The consequence is
    the important part: J comes back biased by eps with an unchanged error bar
    and an unchanged goodness of fit.
    """
    p, th = formic
    for eps in (1e-6, 1e-5, 1e-4):
        resid, _ = ms.degeneracy_check(p, th, eps=eps)
        assert resid < 1e-3, f"eps={eps} left a {resid} sigma residual"


def test_axis_scale_biases_J_by_epsilon(formic):
    p, th = formic
    eps = 1e-4
    _, theta2 = ms.degeneracy_check(p, th, eps=eps)
    assert theta2[0] == pytest.approx(th[0] * (1 + eps), rel=1e-12)


# -- what the other two do to the data ---------------------------------------
def test_equivalence_response_is_quadratic(formaldehyde):
    """Breaking equivalence antisymmetrically is second order, not first.

    This is why a small inequivalence hides: the XA2 line sits at 3/2 of the
    *mean* coupling, so the antisymmetric part cannot enter linearly. Ten times
    the imbalance moves the summary by about a hundred times as much.
    """
    p, th = formaldehyde
    x0 = ms.gen_equivalence(p, th, 0.0)
    sig = p.slot_sigmas()

    def dev(d):
        return np.max(np.abs((ms.gen_equivalence(p, th, d) - x0) / sig))

    small, big = dev(0.05), dev(0.5)
    assert big / small == pytest.approx(100.0, rel=0.35)


def test_equivalence_needs_two_pairs_in_the_class(formic):
    p, th = formic                      # one C-H pair only
    with pytest.raises(ValueError):
        ms.gen_equivalence(p, th, 0.1)


def test_drift_broadens_the_reported_width(formic):
    p, th = formic
    w_clean = _clean(p, th)[-1]
    w_drift = ms.gen_drift(p, th, 1.0)[-1]
    assert w_drift > w_clean            # both are log10(width)


def test_drift_grows_with_the_excursion(formic):
    p, th = formic
    x0 = _clean(p, th)
    sig = p.slot_sigmas()
    devs = [np.max(np.abs((ms.gen_drift(p, th, d) - x0) / sig))
            for d in (0.1, 0.5, 1.0)]
    assert devs[0] < devs[1] < devs[2]


# -- the classical detector --------------------------------------------------
def test_chi2_matches_its_degrees_of_freedom_when_the_model_is_right(formic):
    """At the truth the summed squared residual should average x_dim."""
    p, th = formic
    rng = np.random.default_rng(0)
    sig = p.slot_sigmas()
    mu = p.simulate_one(th, noisy=False)
    stats_ = [ms.goodness_of_fit(p, mu + rng.normal(0, sig), th)[0]
              for _ in range(400)]
    assert np.mean(stats_) == pytest.approx(len(sig), rel=0.15)


def test_chi2_fires_on_a_spectrum_the_model_cannot_fit(formic):
    p, th = formic
    x = ms.gen_drift(p, th, 1.0)
    chi2, dof, pval = ms.goodness_of_fit(p, x, th)
    assert chi2 / dof > 5.0 and pval < 1e-3


def test_generate_adds_noise_only_when_given_a_generator(formic):
    p, th = formic
    quiet = ms.generate(p, th, "axis_scale", 0.0, rng=None)
    noisy = ms.generate(p, th, "axis_scale", 0.0,
                        rng=np.random.default_rng(0))
    assert quiet == pytest.approx(_clean(p, th))
    assert not np.allclose(quiet, noisy)
