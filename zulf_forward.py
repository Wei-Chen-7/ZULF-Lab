#!/usr/bin/env python3
"""ZULF forward model: spectra from spin systems, fast enough for inference.

This is the simulation engine behind the SBI project: given a spin system and a
parameter vector (couplings, residual field, preparation, relaxation, detector
response) it returns the observable ZULF NMR spectrum. It is deliberately
separate from ``zulf_nmr.py``, which is the interactive teaching demo.

Design targets, taken from the proposal:

* **Line lists, not time grids.** The network is fed (frequency, amplitude,
  width) triples, and the likelihood is evaluated from them. Everything here is
  built around an exact line list obtained from one diagonalization; a time
  grid is synthesized only when explicitly asked for. This is ~240x faster than
  building a 6000-point FID and FFT-ing it, and it is exact.
* **The field is a vector.** A field along the sensor axis does essentially
  nothing (a 2.3e-6 Hz second-order shift at 1 nT); a field with a transverse
  component restructures the spectrum at the 10-mHz scale. Only |B| and the
  angle to the sensor axis matter, so the field is two parameters.
* **rho(0) is general.** The thermal/sudden-drop case rho(0) ~ M_z is only one
  of the protocols in use, and the others give complex (out-of-phase) weights.

Conventions (all three cross-checked against the literature):

    H = 2*pi * sum_{i<j} J_ij I_i . I_j  -  sum_i gamma_i B . I_i          (1)
    S(t) = Tr[rho(t) M],  M = sum_i gamma_i I_iz                           (2)

with J in Hz, B in microtesla, gamma_i quoted as gamma/2pi in MHz/T, so that
nu_i [Hz] = gamma_i [MHz/T] * B [uT]. Every term carries its 2*pi, so U(t) =
exp(-i H t) oscillates at J rather than J/(2*pi).

References
----------
[1]  Wilzewski, Afach, Blanchard, Budker, J. Magn. Reson. 284, 66 (2017).
     Baseline fitting method; hard-pulse preparation; curvature error bars.
[3]  Blanchard et al., JACS 135, 3607 (2013).
[6]  Theis et al., Chem. Phys. Lett. 580, 160 (2013).
     nu = J(I_A + 1/2); XA2 -> 3/2 J; XA3 -> J, 2J.
[7]  Butler et al., J. Chem. Phys. 138, 184202 (2013).
     E = (J/2)[F_A(F_A+1) - I_A(I_A+1) - S(S+1)]; selection rules.
[13] Put, Pustelny, Budker, Druga, Sjolander, Pines, Barskiy,
     Anal. Chem. 93, 3226 (2021).  Transverse field -> doublet about J split by
     the SUM of the Larmor frequencies, plus one line at their MEAN.
[14] Stern & Sheberstov, Magn. Reson. 4, 87 (2023).
     rho_eq = sum_l P_l I_lz with P_l ~ gamma_l; adiabatic ramp; finite pulses.
[16] Omar et al., Commun. Chem. 9, 123 (2026).
     OPM: 150 Hz bandwidth (3 dB) AND a 6th-order hardware low-pass at 500 Hz.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "NUCLEI", "SpinSystem", "field_vector", "hamiltonian",
    "rho_thermal", "rho_adiabatic", "rho_adiabatic_ramp",
    "apply_hard_pulse", "apply_field_pulse",
    "line_list", "synth_fid", "synth_spectrum",
    "detector_response", "SYSTEMS", "build_system",
]

# Gyromagnetic ratios gamma/2*pi in MHz/T (so nu[Hz] = gamma[MHz/T] * B[uT]).
NUCLEI = {
    "1H": 42.577478,
    "2H": 6.536,
    "13C": 10.708397,
    "15N": -4.3172,
    "19F": 40.077,
    "31P": 17.235,
}

_TWO_PI = 2.0 * np.pi

# --- Pauli / spin-1/2 operators (S = sigma/2) ------------------------------
_SX = np.array([[0, 1], [1, 0]], dtype=complex) / 2
_SY = np.array([[0, -1j], [1j, 0]], dtype=complex) / 2
_SZ = np.array([[1, 0], [0, -1]], dtype=complex) / 2
_ID = np.eye(2, dtype=complex)
_SPIN = {"x": _SX, "y": _SY, "z": _SZ}


def _embed(op: np.ndarray, i: int, n: int) -> np.ndarray:
    """Embed a single-spin operator at site i of an n-spin system."""
    out = np.array([[1.0 + 0j]])
    for k in range(n):
        out = np.kron(out, op if k == i else _ID)
    return out


class SpinSystem:
    """A spin-1/2 system: nuclei, a full J matrix, and cached operators.

    Parameters
    ----------
    nuclei : sequence of str
        Nucleus labels, e.g. ``["13C", "1H", "1H"]``. Site order defines the
        index order of the J matrix.
    J : (n, n) array, optional
        Symmetric matrix of scalar couplings in Hz. Diagonal is ignored. The
        full matrix is stored so that *any* topology is representable, not just
        one heteronucleus coupled to equivalent protons.
    """

    def __init__(self, nuclei, J=None):
        self.nuclei = list(nuclei)
        self.n = len(self.nuclei)
        self.dim = 2 ** self.n
        self.gammas = np.array([NUCLEI[x] for x in self.nuclei], float)

        if J is None:
            J = np.zeros((self.n, self.n))
        J = np.asarray(J, float)
        if J.shape != (self.n, self.n):
            raise ValueError(f"J must be {(self.n, self.n)}, got {J.shape}")
        self.J = 0.5 * (J + J.T)          # symmetrize; diagonal never used

        self.Ix = [_embed(_SX, i, self.n) for i in range(self.n)]
        self.Iy = [_embed(_SY, i, self.n) for i in range(self.n)]
        self.Iz = [_embed(_SZ, i, self.n) for i in range(self.n)]

        # Detected observable: total magnetic moment along the sensor axis,
        # M = sum_i gamma_i I_iz  (ref [14] Table 2). Scaled by 1/gamma_H so
        # numbers stay O(1); an overall scale is a fitted nuisance anyway.
        w = self.gammas / NUCLEI["1H"]
        self.M = sum(wi * Iz for wi, Iz in zip(w, self.Iz))
        # Total F_z, conserved whenever the field lies along the sensor axis.
        self.Fz = sum(self.Iz)
        # Dot-product operators I_i . I_j, cached: the J-coupling basis.
        self._dot = {}
        for i in range(self.n):
            for j in range(i + 1, self.n):
                self._dot[(i, j)] = (self.Ix[i] @ self.Ix[j]
                                     + self.Iy[i] @ self.Iy[j]
                                     + self.Iz[i] @ self.Iz[j])

    def __repr__(self):
        return f"SpinSystem({'-'.join(self.nuclei)}, dim={self.dim})"


def field_vector(B_magnitude: float, theta_deg: float) -> np.ndarray:
    """Field vector (uT) of given magnitude at angle theta to the sensor axis.

    By symmetry about the sensor axis (z) only |B| and theta matter, so the
    residual field is two parameters, not three. theta = 0 is longitudinal
    (does essentially nothing); theta = 90 is transverse.
    """
    th = np.radians(theta_deg)
    return np.array([B_magnitude * np.sin(th), 0.0, B_magnitude * np.cos(th)])


def hamiltonian(sys: SpinSystem, B=(0.0, 0.0, 0.0), J=None) -> np.ndarray:
    """H = 2*pi sum_{i<j} J_ij I_i.I_j  -  2*pi sum_i nu_i (Bhat . I_i).

    ``B`` is a 3-vector in microtesla. Matches ref [14] Eqs. (4) and (12).
    ``J`` optionally overrides ``sys.J`` with another coupling matrix, so a
    parameter sweep can reuse the cached operators instead of rebuilding the
    system for every sample.
    """
    Jm = sys.J if J is None else np.asarray(J, float)
    H = np.zeros((sys.dim, sys.dim), dtype=complex)
    for (i, j), dot in sys._dot.items():
        Jij = Jm[i, j]
        if Jij:
            H += _TWO_PI * Jij * dot
    B = np.asarray(B, float)
    if np.any(B):
        for i in range(sys.n):
            nu = sys.gammas[i]              # Hz per uT
            H -= _TWO_PI * nu * (B[0] * sys.Ix[i]
                                 + B[1] * sys.Iy[i]
                                 + B[2] * sys.Iz[i])
    return H


# ===========================================================================
# Initial states -- the three preparation protocols in use
# ===========================================================================
def rho_thermal(sys: SpinSystem, axis="z", B_pol=None, temperature=298.0):
    """Prepolarized thermal state (the *sudden field drop* preparation).

    High-field thermal equilibrium is a product state whose deviation part is
    ``sum_l P_l I_l,axis`` with polarization ``P_l = tanh(hbar gamma_l B / 2kT)``
    (ref [14] Eqs. 21-24). Since ``P_l ~ gamma_l`` to better than 1 part in 1e5
    at any realistic prepolarizing field, the default is the linear form, i.e.
    exactly the detected observable M. Ref [14] warns explicitly against
    dropping the gamma weighting (their Eq. 25), so we never do.

    Pass ``B_pol`` (tesla) to use exact tanh polarizations instead.
    """
    ops = {"x": sys.Ix, "y": sys.Iy, "z": sys.Iz}[axis]
    if B_pol is None:
        w = sys.gammas / NUCLEI["1H"]
    else:
        hbar, kB = 1.054571817e-34, 1.380649e-23
        gamma_rad = sys.gammas * 1e6 * _TWO_PI          # MHz/T -> rad/s/T
        w = np.tanh(hbar * gamma_rad * B_pol / (2 * kB * temperature))
        w = w / np.abs(w).max()
    return sum(wi * op for wi, op in zip(w, ops))


def rho_adiabatic(sys: SpinSystem, rho_high=None):
    """Adiabatic field drop, by level following (fast, exact in the limit).

    Ref [14] simulates this by propagating through a monoexponential ramp in
    5000 steps. That is unnecessary: an adiabatic ramp along the sensor axis
    conserves total F_z, so within each F_z block the k-th eigenstate of the
    high-field Hamiltonian maps onto the k-th eigenstate of the zero-field
    Hamiltonian, carrying its population. The result is diagonal in the
    zero-field eigenbasis and therefore stationary -- an adiabatic drop with no
    pulse produces no oscillating signal at all, which is the published result.

    ``rho_adiabatic_ramp`` does it the slow way and is used to test this.
    """
    if rho_high is None:
        rho_high = rho_thermal(sys)
    H_zf = hamiltonian(sys)
    # High-field limit: Zeeman along z dominates. Its eigenbasis is the
    # computational basis, in which rho_high is already diagonal.
    H_hi = -sum(g * Iz for g, Iz in zip(sys.gammas, sys.Iz))

    m = np.round(np.real(np.diag(sys.Fz)), 9)
    rho = np.zeros((sys.dim, sys.dim), dtype=complex)
    for mv in np.unique(m):
        idx = np.flatnonzero(m == mv)
        sub = np.ix_(idx, idx)
        _, V_hi = np.linalg.eigh(H_hi[sub])
        _, V_zf = np.linalg.eigh(H_zf[sub])
        # population of each high-field eigenstate
        pops = np.einsum("ki,kl,li->i", V_hi.conj(), rho_high[sub], V_hi).real
        # adiabatic connection: k-th (ascending energy) -> k-th
        block = (V_zf * pops) @ V_zf.conj().T
        rho[sub] += block
    return rho


def rho_adiabatic_ramp(sys: SpinSystem, rho_high=None, B_start=200.0,
                       t_decay=0.5, tau=0.05, n_steps=5000):
    """Adiabatic drop by explicit propagation through ref [14] Eq. (42).

    Monoexponential ramp B(t) from ``B_start`` (uT) to 0 over ``t_decay`` s.
    Slow; kept as the reference implementation that validates
    ``rho_adiabatic``.
    """
    from scipy.linalg import expm
    if rho_high is None:
        rho_high = rho_thermal(sys)
    dt = t_decay / n_steps
    e_td = np.exp(-t_decay / tau)
    rho = rho_high.astype(complex)
    for k in range(n_steps):
        t = k * dt
        B = B_start * (np.exp(-t / tau) - e_td) / (1.0 - e_td)
        U = expm(-1j * hamiltonian(sys, (0.0, 0.0, B)) * dt)
        rho = U @ rho @ U.conj().T
    return rho


def apply_hard_pulse(sys: SpinSystem, rho, angle, axis="x", ref="13C"):
    """Instantaneous (hard) pulse, the ref [1] convention.

    Wilzewski et al. apply ``U = exp(-i sum_j gamma_j I_{x,j} * angle/gamma_ref)``
    and explicitly neglect J-coupling during the pulse. The rotation angle for
    spin j is therefore ``angle * gamma_j / gamma_ref``: a pi pulse on 13C is a
    ~3.98 pi pulse on 1H. This is what makes the resulting line weights complex
    (out of phase) rather than the non-negative |<n|M|m>|^2 of a sudden drop.
    """
    from scipy.linalg import expm
    ops = {"x": sys.Ix, "y": sys.Iy, "z": sys.Iz}[axis]
    scale = angle / NUCLEI[ref]
    gen = sum(g * op for g, op in zip(sys.gammas, ops))
    U = expm(-1j * scale * gen)
    return U @ rho @ U.conj().T


def apply_field_pulse(sys: SpinSystem, rho, B_pulse, duration, axis="x"):
    """Finite DC field pulse, the ref [14] convention (J-coupling included).

    ``U = exp(-i (H_J + H_Z(B_pulse)) tau)``. Ref [14] uses 50 uT for 150 us
    (z axis) or 910 us (x axis) on an XA system, chosen from Rabi curves.
    Unlike ``apply_hard_pulse`` this keeps H_J on during the pulse, so the two
    conventions genuinely differ.
    """
    from scipy.linalg import expm
    B = {"x": (B_pulse, 0, 0), "y": (0, B_pulse, 0), "z": (0, 0, B_pulse)}[axis]
    U = expm(-1j * hamiltonian(sys, B) * duration)
    return U @ rho @ U.conj().T


# ===========================================================================
# The line list -- the core primitive
# ===========================================================================
def line_list(sys: SpinSystem, rho0=None, B=(0.0, 0.0, 0.0), H=None,
              amp_tol=1e-12, collapse=True, decimals=9, J=None):
    """Exact spectral line list from one diagonalization.

    Diagonalizing H = V diag(E) V^dagger and writing rho~ = V^H rho0 V,
    M~ = V^H M V, the signal is exactly

        S(t) = Re sum_{n,m} rho~[n,m] M~[m,n] exp(-i (E_n - E_m) t),

    a sum of complex exponentials at the Bohr frequencies. We return the
    positive-frequency half; amplitudes are kept **complex** because pulse
    preparations produce out-of-phase (and effectively negative) weights, which
    a magnitude would destroy.

    Returns
    -------
    freqs : (k,) float array, Hz, ascending
    amps  : (k,) complex array, such that
            S(t) = Re sum_k amps[k] * exp(-2j*pi*freqs[k]*t) + DC
    """
    if H is None:
        H = hamiltonian(sys, B, J=J)
    if rho0 is None:
        rho0 = sys.M
    E, V = np.linalg.eigh(H)
    Vd = V.conj().T
    A = (Vd @ rho0 @ V) * (Vd @ sys.M @ V).T      # A[n,m] = rho~[n,m] M~[m,n]
    F = (E[:, None] - E[None, :]) / _TWO_PI       # Hz

    iu = np.triu_indices(sys.dim, k=1)
    # F[n,m] = E_n - E_m; take whichever ordering gives a positive frequency
    f_hi, f_lo = F[iu], -F[iu]
    a_hi, a_lo = A[iu], A.T[iu]
    pos = f_hi > 0
    freqs = np.where(pos, f_hi, f_lo)
    # a coherence and its conjugate partner both contribute -> factor 2
    amps = 2.0 * np.where(pos, a_hi, a_lo)

    keep = (freqs > 1e-12) & (np.abs(amps) > amp_tol)
    freqs, amps = freqs[keep], amps[keep]
    if not collapse:
        order = np.argsort(freqs)
        return freqs[order], amps[order]

    key = np.round(freqs, decimals)
    uniq, inv = np.unique(key, return_inverse=True)
    summed = np.zeros(uniq.shape, dtype=complex)
    np.add.at(summed, inv, amps)
    keep = np.abs(summed) > amp_tol
    return uniq[keep], summed[keep]


def synth_fid(freqs, amps, t, T2=1.0, dead_time=0.0):
    """FID from a line list. ``T2`` is a scalar or a per-line array."""
    t = np.asarray(t, float) + dead_time
    T2 = np.broadcast_to(np.asarray(T2, float), freqs.shape)
    phase = np.exp(-_TWO_PI * 1j * np.outer(t, freqs) - np.outer(t, 1.0 / T2))
    return (phase @ amps).real


def synth_spectrum(freqs, amps, grid, T2=1.0, dead_time=0.0):
    """Complex Lorentzian spectrum on ``grid`` (Hz), analytic -- no FFT.

    ``.real`` is the absorption lineshape, with FWHM 1/(pi*T2) -- the usual
    line-broadening convention, and the same 0.3183 Hz at T2 = 1 s that ref [14]
    Table 1 uses. ``.imag`` is dispersion. Note the *magnitude* spectrum of the
    same line is broader by exactly sqrt(3), so widths must be quoted on the
    absorption part.
    """
    grid = np.asarray(grid, float)
    T2 = np.broadcast_to(np.asarray(T2, float), freqs.shape)
    gam = 1.0 / (_TWO_PI * T2)
    a = amps * np.exp(-_TWO_PI * 1j * freqs * dead_time)
    denom = (grid[:, None] - freqs[None, :]) + 1j * gam[None, :]
    return (a[None, :] / denom).sum(axis=1) * (1j / _TWO_PI)


# ===========================================================================
# Detector response
# ===========================================================================
def detector_response(freqs, f_3db=150.0, gain=1.0, dead_time=0.0,
                      lowpass_hz=500.0, lowpass_order=6):
    """Complex OPM transfer function at the given frequencies.

    Ref [16] specifies the QuSpin OPM as 150 Hz bandwidth (the -3 dB point)
    *and* a sharp 6th-order hardware digital low-pass at 500 Hz. The second
    filter matters: between 140 and 300 Hz it changes amplitudes by less than
    0.05%, but it contributes about -67 degrees of *differential* phase across
    that band, which is comparable to the -19 degrees from the 150 Hz pole
    alone. Because its corner and order are fixed hardware it costs no free
    parameters -- only ``gain``, ``f_3db`` and ``dead_time`` are fitted.

    Set ``lowpass_order=0`` to disable the hardware filter.
    """
    f = np.asarray(freqs, float)
    H = gain / (1.0 + 1j * f / f_3db)                  # sensor response
    if lowpass_order:
        from scipy import signal
        b, a = signal.butter(lowpass_order, _TWO_PI * lowpass_hz,
                             btype="low", analog=True)
        _, Hlp = signal.freqs(b, a, worN=_TWO_PI * f)
        H = H * Hlp
    if dead_time:
        H = H * np.exp(-_TWO_PI * 1j * f * dead_time)
    return H


# ===========================================================================
# Molecule presets
# ===========================================================================
def _xan(hetero, n_h, J_XA, J_HH=0.0):
    """X coupled to n equivalent protons: the canonical XA_n topology."""
    nuclei = [hetero] + ["1H"] * n_h
    J = np.zeros((n_h + 1, n_h + 1))
    J[0, 1:] = J[1:, 0] = J_XA
    for a in range(1, n_h + 1):
        for b in range(a + 1, n_h + 1):
            J[a, b] = J[b, a] = J_HH
    return nuclei, J


def _benzene_13c1():
    """Benzene-13C1, the 7-spin benchmark of ref [1], with its fitted values.

    Sites: 0 = 13C (bonded to H1), then H1..H6 around the ring. The 13C breaks
    the ring symmetry, leaving a mirror plane through H1 and H4; couplings are
    grouped by that symmetry, which reproduces exactly the 13 distinct values
    of ref [1] Table I.
    """
    nuclei = ["13C"] + ["1H"] * 6
    J = np.zeros((7, 7))
    C, H = 0, {k: k for k in range(1, 7)}          # H[k] -> site index k
    # C-H couplings by ring distance from the substituted carbon
    J[C, H[1]] = 158.363                            # 1J_CH
    J[C, H[2]] = J[C, H[6]] = 1.136                 # 2J_CH
    J[C, H[3]] = J[C, H[5]] = 7.609                 # 3J_CH
    J[C, H[4]] = -1.285                             # 4J_CH
    # H-H: ortho (3J), meta (4J), para (5J), split by the mirror symmetry
    J[H[1], H[2]] = J[H[1], H[6]] = 7.534
    J[H[2], H[3]] = J[H[6], H[5]] = 7.543
    J[H[3], H[4]] = J[H[5], H[4]] = 7.543
    J[H[1], H[3]] = J[H[1], H[5]] = 1.381
    J[H[2], H[4]] = J[H[6], H[4]] = 1.382
    J[H[2], H[6]] = 1.384
    J[H[3], H[5]] = 1.387
    J[H[1], H[4]] = 0.658
    J[H[2], H[5]] = J[H[3], H[6]] = 0.660
    return nuclei, J + J.T - np.diag(np.diag(J))


SYSTEMS = {
    # name: (builder, note)
    "formic_acid": (lambda: _xan("13C", 1, 222.2),
                    "[13C]-formic acid, XA; J = 222.2 Hz (ref [13])"),
    "glycine": (lambda: _xan("13C", 2, 140.0),
                "[1-13C]-glycine, effective A2X -> one line at 3/2 J (ref [13])"),
    "methanol": (lambda: _xan("13C", 3, 140.0),
                 "[13C]-methanol methyl, XA3 -> lines at J and 2J (ref [6])"),
    "benzene_13c1": (_benzene_13c1,
                     "benzene-13C1, 7 spins, 13 couplings (ref [1] Table I)"),
}


def build_system(name: str) -> SpinSystem:
    """Instantiate one of the presets in :data:`SYSTEMS`."""
    builder, _ = SYSTEMS[name]
    nuclei, J = builder()
    return SpinSystem(nuclei, J)


if __name__ == "__main__":  # pragma: no cover - smoke demo
    import time
    for nm in SYSTEMS:
        s = build_system(nm)
        t0 = time.perf_counter()
        f, a = line_list(s)
        dt = (time.perf_counter() - t0) * 1e3
        top = np.argsort(-np.abs(a))[:4]
        peaks = ", ".join(f"{f[i]:.3f} Hz" for i in sorted(top, key=lambda k: f[k]))
        print(f"{nm:14s} {str(s):34s} {len(f):5d} lines  {dt:7.3f} ms   {peaks}")
