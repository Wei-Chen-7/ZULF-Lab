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


# ---------------------------------------------------------------------------
# multi-spin problems
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,n_par,n_spin", [
    ("formic_acid", 4, 2), ("glycine", 5, 3),
    ("formaldehyde", 5, 3), ("methanol", 5, 4),
])
def test_multi_spin_problems_build(name, n_par, n_spin):
    p = zi.InferenceProblem(system=name)
    assert p.sys.n == n_spin
    assert len(p.param_names) == n_par == len(p.low) == len(p.high)
    assert p.param_names[-3:] == ["B_nT", "B_theta_deg", "T2_s"]
    x = p.simulate_one(p.sample_prior(1)[0], noisy=False)
    assert len(x) == p.x_dim() and np.all(np.isfinite(x))


def test_J_matrix_assigns_every_pair_in_a_class():
    p = zi.InferenceProblem(system="methanol")
    J = p.J_matrix(np.array([141.0, -12.4, 1.0, 55.0, 12.0]))
    assert np.allclose(J, J.T)
    for j in (1, 2, 3):
        assert np.isclose(J[0, j], 141.0)          # every C-H pair
    for i, j in ((1, 2), (1, 3), (2, 3)):
        assert np.isclose(J[i, j], -12.4)          # every H-H pair
    assert np.allclose(np.diag(J), 0.0)


@pytest.mark.parametrize("J_HH", [5.0, -12.4, 15.0])
def test_J_HH_inside_an_equivalent_group_is_a_flat_direction(J_HH):
    """Delta I_A = 0 means J_HH moves no line, so it cannot be measured.

    This is the methanol identifiability case: five of the six pairwise
    couplings are unmeasurable, and the summary must be literally unchanged.
    """
    p = zi.InferenceProblem(system="methanol")
    base = p.simulate_one(np.array([141.0, 0.0, 1.0, 55.0, 12.0]), noisy=False)
    alt = p.simulate_one(np.array([141.0, J_HH, 1.0, 55.0, 12.0]), noisy=False)
    assert np.allclose(base, alt, atol=1e-9)


def test_likelihood_is_flat_in_an_unmeasurable_coupling():
    p = zi.InferenceProblem(system="methanol", seed=11)
    x = p.simulate_one(np.array([141.0, 3.0, 1.0, 55.0, 12.0]))
    lls = p.log_likelihood(x, np.array([[141.0, j, 1.0, 55.0, 12.0]
                                        for j in (-10.0, 0.0, 7.0, 14.0)]))
    assert np.allclose(lls, lls[0], atol=1e-6), lls


def test_shrinkage_separates_measured_from_flat_directions():
    """A flat direction must report ~0 shrinkage, not a confident number."""
    p = zi.InferenceProblem(system="methanol", seed=3)
    rng = np.random.default_rng(0)
    s = rng.uniform(p.low, p.high, size=(4000, len(p.low)))   # = the prior
    s[:, 0] = 141.0 + rng.normal(0, 1e-3, 4000)               # J_CH pinned
    rows = {r["param"]: r for r in zi.shrinkage(p, s)}
    assert rows["J_CH"]["shrinkage"] > 0.99
    assert rows["J_CH"]["constrained"]
    assert abs(rows["J_HH"]["shrinkage"]) < 0.1
    assert not rows["J_HH"]["constrained"]


# ---------------------------------------------------------------------------
# The field-angle degeneracy
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("th", [10.0, 30.0, 55.0, 75.0])
def test_theta_and_180_minus_theta_are_indistinguishable(th):
    """Only |cos(theta)| is identifiable.

    R_x(pi) maps B=(Bx,0,Bz) to (Bx,0,-Bz) and sends both rho(0)=M and M to
    -M, so S(t) = Tr[rho(t) M] is unchanged. The line list is bit-identical;
    the summary agrees to floating-point roundoff on ~222 Hz values.
    """
    s = zf.build_system("formic_acid")
    Jm = np.array([[0.0, 222.9], [222.9, 0.0]])
    f1, a1 = zf.line_list(s, B=zf.field_vector(1e-3, th), J=Jm)
    f2, a2 = zf.line_list(s, B=zf.field_vector(1e-3, 180.0 - th), J=Jm)
    assert np.allclose(np.sort(f1), np.sort(f2), rtol=0, atol=1e-9)
    assert np.allclose(np.sort(np.abs(a1)), np.sort(np.abs(a2)), atol=1e-9)


def test_prior_breaks_the_angle_degeneracy_by_convention():
    """The prior stops at 90 degrees, as a canonical ordering would."""
    p = zi.InferenceProblem()
    assert np.isclose(p.high[2], 90.0)
    assert np.all(p.sample_prior(200)[:, 2] <= 90.0)


# ---------------------------------------------------------------------------
# Exact likelihood and reweighting helpers
# ---------------------------------------------------------------------------
def test_noise_is_applied_to_every_slot():
    """A fully Gaussian noise model is what makes the likelihood analytic."""
    p = zi.InferenceProblem(sigma_f=1e-3, sigma_logamp=0.02, seed=3)
    th = np.array([222.9, 0.0, 0.0, 12.0])          # zero field: padded slots
    draws = np.stack([p.simulate_one(th) for _ in range(600)])
    assert np.all(draws.std(axis=0) > 0), "every slot must be perturbed"
    assert np.allclose(draws.std(axis=0), p.slot_sigmas(), rtol=0.2)


def test_log_likelihood_peaks_at_the_truth():
    p = zi.InferenceProblem(seed=5)
    th = np.array([222.9, 1.0, 55.0, 12.0])
    x = p.simulate_one(th, noisy=False)
    grid = np.array([[J, 1.0, 55.0, 12.0]
                     for J in np.linspace(222.85, 222.95, 41)])
    ll = p.log_likelihood(x, grid)
    assert np.isclose(grid[np.argmax(ll), 0], 222.9, atol=3e-3)


def test_log_likelihood_matches_a_hand_computed_gaussian():
    p = zi.InferenceProblem(seed=7)
    th = np.array([222.9, 1.0, 55.0, 12.0])
    x_clean = p.simulate_one(th, noisy=False)
    x_obs = x_clean.copy()
    x_obs[2] += 0.002
    sig = p.slot_sigmas()
    want = (-0.5 * ((x_obs - x_clean) / sig) ** 2).sum() \
        - np.sum(np.log(sig)) - 0.5 * len(sig) * np.log(2 * np.pi)
    assert np.isclose(p.log_likelihood(x_obs, th)[0], want)


def test_weighted_quantile_reduces_to_the_plain_one():
    v = np.random.default_rng(0).normal(size=5000)
    w = np.ones_like(v)
    got = zi.weighted_quantile(v, [0.025, 0.5, 0.975], w)
    want = np.percentile(v, [2.5, 50, 97.5])
    assert np.allclose(got, want, atol=0.05)


def test_weighted_quantile_respects_weights():
    v = np.array([0.0, 1.0])
    assert np.isclose(zi.weighted_quantile(v, [0.5], np.array([1.0, 0.0]))[0], 0.0)
    assert np.isclose(zi.weighted_quantile(v, [0.5], np.array([0.0, 1.0]))[0], 1.0)


def test_in_prior_flags_out_of_box_samples():
    p = zi.InferenceProblem()
    good = p.sample_prior(20)
    bad = good.copy()
    bad[:, 2] = 150.0                                # beyond the 90 deg cut
    assert p.in_prior(good).all()
    assert not p.in_prior(bad).any()


# -- model persistence -------------------------------------------------------
def _tiny(prob, tag, tmpdir, **kw):
    return zi.train_or_load(prob, tag=tag, n_sims=400, seed=0,
                            model_dir=str(tmpdir), max_num_epochs=2,
                            training_batch_size=100, **kw)


def test_train_or_load_writes_then_reuses(tmp_path):
    prob = zi.InferenceProblem(seed=0)
    _, meta = _tiny(prob, "t", tmp_path)
    assert (tmp_path / "t.pt").exists()
    assert meta["train_seconds"] > 0
    _, meta2 = _tiny(prob, "t", tmp_path)
    assert "train_seconds" not in meta2 or meta2["train_seconds"] == \
        meta["train_seconds"]                      # loaded, not retrained


def test_loaded_posterior_samples_the_same_shape(tmp_path):
    import torch
    prob = zi.InferenceProblem(seed=0)
    post, _ = _tiny(prob, "s", tmp_path)
    x = prob.simulate_one(prob.sample_prior(1)[0])
    draws = post.sample((16,), x=torch.as_tensor(x, dtype=torch.float32),
                        show_progress_bars=False).numpy()
    assert draws.shape == (16, len(prob.param_names))


def test_a_stale_signature_forces_a_retrain(tmp_path):
    """A network trained on a different observation model must not be reused."""
    a = zi.InferenceProblem(seed=0)
    _tiny(a, "sig", tmp_path)
    b = zi.InferenceProblem(seed=0, merge_hz=0.05)   # different summary
    _, meta = _tiny(b, "sig", tmp_path)
    assert "train_seconds" in meta                  # retrained, not reused
    assert meta["signature"] == repr(sorted(b.summary_signature().items()))


def test_force_retrains_even_on_a_hit(tmp_path):
    prob = zi.InferenceProblem(seed=0)
    _, m1 = _tiny(prob, "f", tmp_path)
    _, m2 = _tiny(prob, "f", tmp_path, force=True)
    assert m2["train_seconds"] > 0 and m2 is not m1


def test_save_and_load_round_trip_carries_metadata(tmp_path):
    prob = zi.InferenceProblem(seed=0)
    post, _ = _tiny(prob, "rt", tmp_path)
    path = zi.save_posterior(post, "rt2", meta=dict(note="hello"),
                             model_dir=str(tmp_path))
    assert path.endswith("rt2.pt")
    _, meta = zi.load_posterior("rt2", model_dir=str(tmp_path))
    assert meta["note"] == "hello"
    assert "sbi_version" in meta and "torch_version" in meta
