"""Validation suite for the ZULF forward model.

Every test here encodes a fact taken from the literature, with the source named
in the test. These are the "check against [1,3,6,7]" tests the project needs:
if a refactor breaks the physics, one of these fails.

Run with:  pytest -q test_forward.py
"""

import time

import numpy as np
import pytest

import zulf_forward as zf


TWO_PI = 2 * np.pi
GH, GC = zf.NUCLEI["1H"], zf.NUCLEI["13C"]


def xan(n_h, J=140.0, J_HH=0.0, hetero="13C"):
    nuclei, Jm = zf._xan(hetero, n_h, J, J_HH)
    return zf.SpinSystem(nuclei, Jm)


def peaks(sys, rho0=None, B=(0, 0, 0), rel=1e-6):
    """Line frequencies whose amplitude exceeds ``rel`` times the largest."""
    f, a = zf.line_list(sys, rho0=rho0, B=B)
    if len(f) == 0:
        return np.array([])
    m = np.abs(a)
    return f[m > rel * m.max()]


# ---------------------------------------------------------------------------
# Conventions and the 2*pi test
# ---------------------------------------------------------------------------
def test_hamiltonian_is_hermitian():
    s = xan(3)
    H = zf.hamiltonian(s, zf.field_vector(1.5, 37.0))
    assert np.allclose(H, H.conj().T, atol=1e-12)


def test_two_spin_energies_butler():
    """Ref [7] Sec. III A: F_A=0 lies at -3J/4, F_A=1 at +J/4."""
    J = 140.0
    E, _ = np.linalg.eigh(zf.hamiltonian(xan(1, J)))
    E_hz = np.sort(E / TWO_PI)
    assert np.isclose(E_hz[0], -0.75 * J, atol=1e-9)
    assert np.allclose(E_hz[1:], 0.25 * J, atol=1e-9)
    assert np.isclose(E_hz[-1] - E_hz[0], J, atol=1e-9)   # gap is exactly J


def test_peak_sits_at_exactly_J_not_J_over_2pi():
    """The 2*pi test: a dropped factor would put the line at J/(2*pi)."""
    J = 140.0
    f = peaks(xan(1, J))
    assert len(f) == 1
    assert np.isclose(f[0], J, atol=1e-9)
    assert not np.isclose(f[0], J / TWO_PI, atol=1.0)


# ---------------------------------------------------------------------------
# XA_n line positions: nu = J (I_A + 1/2)
# ---------------------------------------------------------------------------
def _predicted_xan(n_h, J):
    """Refs [6] Eq. (1)-(2), [7] Eq. (40): nu = J (I_A + 1/2), I_A >= 1/2."""
    kmax = n_h / 2.0
    ks, k = [], kmax
    while k >= 0:
        ks.append(k)
        k -= 1
    return sorted({J * (K + 0.5) for K in ks if K >= 0.5})


@pytest.mark.parametrize("n_h", [1, 2, 3, 4, 5])
def test_xan_line_positions(n_h):
    J = 140.0
    got = np.unique(np.round(peaks(xan(n_h, J)), 6))
    want = np.array(_predicted_xan(n_h, J))
    assert np.allclose(got, want, atol=1e-6), f"XA{n_h}: {got} != {want}"


def test_theis_stated_cases():
    """Ref [6]: 'XA2 produces one line at 3/2 J_XA, XA3 lines at J and 2 J'."""
    J = 140.0
    assert np.allclose(np.unique(np.round(peaks(xan(2, J)), 6)), [1.5 * J])
    assert np.allclose(np.unique(np.round(peaks(xan(3, J)), 6)), [J, 2 * J])


def test_glycine_is_a2x_single_line_at_three_halves_J():
    """Ref [13]: '[1-13C]-glycine ... gives a single peak at (3/2) J'."""
    s = zf.build_system("glycine")
    f = np.unique(np.round(peaks(s), 6))
    assert len(f) == 1 and np.isclose(f[0], 1.5 * 140.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Selection rules and exact degeneracies
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("J_HH", [0.0, 5.0, 10.0, -12.4])
def test_equivalent_proton_coupling_moves_no_line(J_HH):
    """Refs [6] Eq. (2), [7] Eq. (46c): Delta I_A = 0, so J_HH is invisible.

    This is the methanol identifiability result: five of the six pairwise
    couplings in the methyl group are unmeasurable.
    """
    ref = np.unique(np.round(peaks(xan(3, 140.0, J_HH=0.0)), 6))
    got = np.unique(np.round(peaks(xan(3, 140.0, J_HH=J_HH)), 6))
    assert np.allclose(ref, got, atol=1e-7)


def test_global_sign_flip_is_a_degeneracy():
    """Flipping every coupling leaves the spectrum unchanged."""
    nuclei, J = zf._xan("13C", 2, 140.0, J_HH=7.0)
    a = np.unique(np.round(peaks(zf.SpinSystem(nuclei, J)), 6))
    b = np.unique(np.round(peaks(zf.SpinSystem(nuclei, -J)), 6))
    assert np.allclose(a, b, atol=1e-7)


def test_permuting_equal_gamma_nuclei_is_a_degeneracy():
    """Swapping two nuclei with the same gamma leaves the spectrum unchanged."""
    nuclei = ["13C", "1H", "1H"]
    J = np.array([[0.0, 140.0, 90.0],
                  [140.0, 0.0, 7.0],
                  [90.0, 7.0, 0.0]])
    P = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])       # swap the protons
    a = np.unique(np.round(peaks(zf.SpinSystem(nuclei, J)), 6))
    b = np.unique(np.round(peaks(zf.SpinSystem(nuclei, P @ J @ P.T)), 6))
    assert np.allclose(a, b, atol=1e-7)


def test_homonuclear_system_has_no_spectrum():
    """Ref [14]: equal gammas make M commute with H_J -- no J-spectrum."""
    s = xan(3, 140.0, hetero="1H")            # all protons
    f, a = zf.line_list(s)
    assert len(f) == 0 or np.abs(a).max() < 1e-12


# ---------------------------------------------------------------------------
# The field is a vector
# ---------------------------------------------------------------------------
def test_longitudinal_field_does_essentially_nothing():
    """A 1 nT field along the sensor axis shifts the line by ~2.3e-6 Hz."""
    J, B = 221.0, 1e-3                                    # 1 nT in uT
    s = xan(1, J)
    f0 = peaks(s)[0]
    f1 = peaks(s, B=zf.field_vector(B, 0.0))[0]
    shift = f1 - f0
    predicted = np.sqrt(J ** 2 + ((GH - GC) * B) ** 2) - J
    assert np.isclose(shift, predicted, rtol=1e-3)
    assert 1e-6 < shift < 1e-5


def test_transverse_field_gives_doublet_split_by_the_sum():
    """Ref [13]: perpendicular field -> two peaks about J separated by the SUM
    of the two Larmor frequencies, plus one line at their MEAN."""
    J, B = 221.0, 1e-3
    s = xan(1, J)
    nu_sum = (GH + GC) * B
    f = np.sort(peaks(s, B=zf.field_vector(B, 90.0)))
    hi = f[f > J / 2]
    lo = f[f <= J / 2]
    assert len(hi) == 2, f"expected a doublet about J, got {hi}"
    assert np.isclose(hi[1] - hi[0], nu_sum, rtol=1e-6)
    assert np.isclose(0.5 * (hi[0] + hi[1]), J, atol=1e-5)
    assert len(lo) >= 1
    assert np.allclose(lo, nu_sum / 2, atol=1e-5)         # the MEAN


def test_centre_component_scales_as_cos_squared_theta():
    """The centre line fades as cos^2(theta) and is forbidden at 90 degrees."""
    J, B = 221.0, 1e-3
    s = xan(1, J)
    ref = None
    for th in (0.0, 30.0, 45.0, 60.0, 75.0):
        f, a = zf.line_list(s, B=zf.field_vector(B, th))
        centre = np.abs(a[np.abs(f - J) < 1e-3])
        assert len(centre) == 1
        scaled = centre[0] / np.cos(np.radians(th)) ** 2
        if ref is None:
            ref = scaled
        assert np.isclose(scaled, ref, rtol=1e-6)
    f, a = zf.line_list(s, B=zf.field_vector(B, 90.0))
    assert not np.any(np.abs(f - J) < 1e-3), "centre line must vanish at 90 deg"


def test_centre_line_forbidden_for_every_preparation():
    """<T0|M|S> = 0 at 90 deg, so no rho(0) can produce the centre line."""
    J, B = 221.0, 1e-3
    s = xan(1, J)
    rng = np.random.default_rng(0)
    R = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    for rho0 in (s.M, s.Iz[0], s.Iz[1], R + R.conj().T):
        f, a = zf.line_list(s, rho0=rho0, B=zf.field_vector(B, 90.0),
                            amp_tol=1e-10)
        assert not np.any(np.abs(f - J) < 1e-3)


# ---------------------------------------------------------------------------
# Preparations
# ---------------------------------------------------------------------------
def test_thermal_state_is_gamma_weighted():
    """Ref [14] Eqs. (21)-(24): rho ~ sum_l P_l I_lz with P_l ~ gamma_l."""
    s = xan(1)
    rho = zf.rho_thermal(s)
    assert np.allclose(rho, s.M, atol=1e-12)
    exact = zf.rho_thermal(s, B_pol=2.0, temperature=298.0)
    assert np.allclose(exact, s.M, rtol=1e-4, atol=1e-6)   # tanh ~ linear


def test_adiabatic_state_is_stationary_and_silent():
    """Ref [14]: an adiabatic drop with no pulse gives no oscillating signal."""
    s = xan(1, 140.0)
    rho = zf.rho_adiabatic(s)
    H = zf.hamiltonian(s)
    assert np.allclose(H @ rho - rho @ H, 0, atol=1e-9)    # stationary
    f, a = zf.line_list(s, rho0=rho)
    assert len(f) == 0 or np.abs(a).max() < 1e-9           # silent


def test_adiabatic_shortcut_matches_explicit_ramp():
    """Level following reproduces ref [14] Eq. (42) propagation."""
    s = xan(1, 140.0)
    fast = zf.rho_adiabatic(s)
    slow = zf.rho_adiabatic_ramp(s, B_start=200.0, t_decay=0.5, tau=0.05,
                                 n_steps=4000)
    # compare populations in the zero-field eigenbasis (phases are irrelevant)
    _, V = np.linalg.eigh(zf.hamiltonian(s))
    pf = np.real(np.diag(V.conj().T @ fast @ V))
    ps = np.real(np.diag(V.conj().T @ slow @ V))
    assert np.allclose(np.sort(pf), np.sort(ps), atol=2e-3)


def test_pulse_acquire_weights_go_negative():
    """Pulse-acquire breaks the non-negativity of the sudden-drop weights.

    A sudden drop gives rho(0) = M and hence weights |<n|M|m>|^2 >= 0. After an
    adiabatic drop plus a DC pulse the weight is signed and depends on the pulse
    angle: it is negative at both pi/2 and pi *proton* flip angles for a
    13C-1H pair, which is exactly why rho(0) cannot be fixed to M.

    At zero field H, rho(0) and M are all real symmetric, so the eigenvectors
    are real and the weights come out real -- signed, not complex. Genuinely
    complex weights require a residual field or a complex rho(0).
    """
    s = xan(1, 140.0)
    _, a_sudden = zf.line_list(s, rho0=zf.rho_thermal(s))
    assert np.all(a_sudden.real > 0) and np.allclose(a_sudden.imag, 0, atol=1e-9)

    base = zf.rho_adiabatic(s)
    for angle in (np.pi / 2, np.pi):
        rho = zf.apply_hard_pulse(s, base, angle, axis="x", ref="1H")
        f, a = zf.line_list(s, rho0=rho, amp_tol=1e-14)
        assert len(f) == 1
        assert a[0].real < -1e-3, f"proton angle {angle}: {a[0]}"
        assert abs(a[0].imag) < 1e-12          # real at zero field


def test_field_pulse_differs_from_hard_pulse():
    """Ref [14] keeps H_J on during the pulse; ref [1] does not."""
    s = xan(1, 140.0)
    base = zf.rho_adiabatic(s)
    r1 = zf.apply_hard_pulse(s, base, np.pi, axis="x")
    r2 = zf.apply_field_pulse(s, base, B_pulse=50.0, duration=910e-6, axis="x")
    assert not np.allclose(r1, r2, atol=1e-6)


# ---------------------------------------------------------------------------
# Detector response
# ---------------------------------------------------------------------------
def test_single_pole_reproduces_the_proposal_numbers():
    """5/4 -> 0.81 and about -62 deg at 280 Hz, for a lone 150 Hz pole."""
    f = np.array([140.0, 280.0])
    H = zf.detector_response(f, f_3db=150.0, lowpass_order=0)
    ratio = 1.25 * np.abs(H[1]) / np.abs(H[0])
    assert np.isclose(ratio, 0.807, atol=0.005)
    assert np.isclose(np.degrees(np.angle(H[1])), -61.8, atol=0.5)


def test_hardware_lowpass_changes_phase_not_amplitude():
    """Ref [16]: the 6th-order 500 Hz filter is ~flat below 300 Hz in
    amplitude but contributes about -67 deg of differential phase."""
    f = np.array([140.0, 280.0])
    H1 = zf.detector_response(f, lowpass_order=0)
    H2 = zf.detector_response(f)                    # with the 6th-order filter
    r1 = np.abs(H1[1]) / np.abs(H1[0])
    r2 = np.abs(H2[1]) / np.abs(H2[0])
    assert np.isclose(r1, r2, rtol=1e-3)            # amplitude ratio unchanged
    d1 = np.degrees(np.angle(H1[1]) - np.angle(H1[0]))
    d2 = np.degrees(np.angle(H2[1]) - np.angle(H2[0]))
    extra = (d2 - d1 + 180) % 360 - 180
    assert np.isclose(extra, -67.0, atol=1.0)


# ---------------------------------------------------------------------------
# Signal synthesis
# ---------------------------------------------------------------------------
def test_synth_spectrum_has_the_right_position_and_width():
    """The absorption line (real part) has FWHM = 1/(pi T2).

    The magnitude spectrum of the same complex Lorentzian is broader by exactly
    sqrt(3), which is why the width must be quoted on the absorption part.
    """
    T2 = 0.5
    f0 = np.array([140.0])
    a0 = np.array([1.0 + 0j])
    grid = np.linspace(138, 142, 40001)
    spec = zf.synth_spectrum(f0, a0, grid, T2=T2)

    absorption = spec.real
    assert np.isclose(grid[np.argmax(absorption)], 140.0, atol=2e-3)
    above = grid[absorption >= absorption.max() / 2]
    assert np.isclose(above.max() - above.min(), 1 / (np.pi * T2), rtol=2e-3)

    mag = np.abs(spec)
    above_m = grid[mag >= mag.max() / 2]
    assert np.isclose(above_m.max() - above_m.min(),
                      np.sqrt(3) / (np.pi * T2), rtol=2e-3)


def test_synth_fid_matches_the_line_list_by_fft():
    """A time grid built from the line list peaks where the lines are."""
    s = zf.build_system("methanol")
    f, a = zf.line_list(s)
    dt, N = 0.5e-3, 6000
    t = np.arange(N) * dt
    S = zf.synth_fid(f, a, t, T2=1.0)
    mag = np.abs(np.fft.rfft(S))
    grid = np.fft.rfftfreq(N, dt)
    found = grid[np.argsort(-mag)[:2]]
    assert np.isclose(min(found), 140.0, atol=0.5)
    assert np.isclose(max(found), 280.0, atol=0.5)


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------
def test_four_spin_line_list_is_fast_enough_for_inference():
    """The proposal budgets 0.22 ms for a four-spin spectrum."""
    s = zf.build_system("methanol")               # 4 spins, dim 16
    zf.line_list(s)                               # warm up
    t0 = time.perf_counter()
    n = 200
    for _ in range(n):
        zf.line_list(s)
    per_ms = (time.perf_counter() - t0) / n * 1e3
    assert per_ms < 2.0, f"{per_ms:.3f} ms per four-spin spectrum"


def test_benzene_reference_system_builds_and_runs():
    """Ref [1] Table I: 13 distinct couplings on a 7-spin system."""
    s = zf.build_system("benzene_13c1")
    assert s.n == 7 and s.dim == 128
    assert np.isclose(s.J[0, 1], 158.363)
    off = s.J[np.triu_indices(7, 1)]
    # Table I lists 13 rows, but 3J_HH(H2,H3) and 4J_HH(H3,H4) share the same
    # fitted value 7.543, so only 12 distinct magnitudes appear in the matrix.
    assert len(np.unique(np.round(np.abs(off), 6))) == 12
    assert len(off) == 21                                  # all C(7,2) pairs set
    f, a = zf.line_list(s)
    assert len(f) > 50
    assert np.all(np.isfinite(f)) and np.all(np.isfinite(a))
