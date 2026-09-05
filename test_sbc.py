"""Tests for the SBC machinery -- verifying the direction of the test itself.

An SBC implementation that cannot detect an overconfident posterior is worse
than useless, because it returns a reassuring verdict. So before relying on it,
these check that it flags each failure mode the right way round.
"""

import numpy as np
import pytest

import zulf_infer as zi
import sbc_check as sc


L = 99


def _diagnose_ranks(ranks):
    return sc.diagnose(ranks.reshape(-1, 1), L)[0]


def test_uniform_ranks_read_as_calibrated():
    rng = np.random.default_rng(0)
    ranks = rng.integers(0, L + 1, size=600)
    assert "calibrated" in _diagnose_ranks(ranks)["verdict"]


def test_edge_loaded_ranks_read_as_overconfident():
    """Too-narrow posteriors push the truth into the tails -> ranks at 0 and L."""
    rng = np.random.default_rng(1)
    edges = rng.choice([0, L], size=400)
    middle = rng.integers(0, L + 1, size=200)
    ranks = np.concatenate([edges, middle])
    d = _diagnose_ranks(ranks)
    assert "OVERCONFIDENT" in d["verdict"], d
    assert d["outer"] > 0.28


def test_centre_loaded_ranks_read_as_too_wide():
    rng = np.random.default_rng(2)
    centre = rng.integers(int(0.45 * L), int(0.55 * L) + 1, size=400)
    rest = rng.integers(0, L + 1, size=200)
    ranks = np.concatenate([centre, rest])
    d = _diagnose_ranks(ranks)
    assert "conservative" in d["verdict"], d
    assert d["centre"] > 0.28


def test_calibrated_case_is_not_flagged_by_chance():
    """Several independent uniform draws should mostly pass, not systematically fail."""
    passes = 0
    for s in range(8):
        rng = np.random.default_rng(100 + s)
        ranks = rng.integers(0, L + 1, size=400)
        if "calibrated" in _diagnose_ranks(ranks)["verdict"]:
            passes += 1
    assert passes >= 6, f"only {passes}/8 uniform samples read as calibrated"


# ---------------------------------------------------------------------------
# rank computation, using mock posteriors
# ---------------------------------------------------------------------------
class _MockPosterior:
    """Stands in for a trained posterior; ignores x by construction."""

    def __init__(self, prob, mode, rng):
        self.prob, self.mode, self.rng = prob, mode, rng

    def sample(self, shape, x=None, show_progress_bars=False):
        import torch
        n = shape[0]
        if self.mode == "prior":
            draws = self.rng.uniform(self.prob.low, self.prob.high,
                                     size=(n, len(self.prob.low)))
        else:                                   # tight blob, unrelated to theta
            mid = 0.5 * (self.prob.low + self.prob.high)
            span = (self.prob.high - self.prob.low) * 1e-4
            draws = mid + self.rng.normal(size=(n, len(mid))) * span
        return torch.as_tensor(draws, dtype=torch.float32)


def test_a_posterior_that_returns_the_prior_is_calibrated():
    """SBC cannot fault an uninformative posterior -- and must not pretend to."""
    prob = zi.InferenceProblem(seed=0)
    post = _MockPosterior(prob, "prior", np.random.default_rng(3))
    ranks = sc.run_sbc(prob, post, n_trials=200, n_post=L, seed=1, verbose=False)
    for row in sc.diagnose(ranks, L):
        assert "calibrated" in row["verdict"] or row["p_ks"] > 0.005, row


def test_run_sbc_flags_a_grossly_overconfident_posterior():
    prob = zi.InferenceProblem(seed=0)
    post = _MockPosterior(prob, "tight", np.random.default_rng(4))
    ranks = sc.run_sbc(prob, post, n_trials=150, n_post=L, seed=2, verbose=False)
    rows = sc.diagnose(ranks, L)
    assert any("OVERCONFIDENT" in r["verdict"] for r in rows), rows


def test_ranks_are_in_range():
    prob = zi.InferenceProblem(seed=0)
    post = _MockPosterior(prob, "prior", np.random.default_rng(5))
    ranks = sc.run_sbc(prob, post, n_trials=40, n_post=L, seed=3, verbose=False)
    assert ranks.shape == (40, 4)
    assert ranks.min() >= 0 and ranks.max() <= L
