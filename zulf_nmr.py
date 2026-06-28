#!/usr/bin/env python3
"""ZULF Lab -- an interactive zero- to ultralow-field NMR spin simulator.

A scoped, from-scratch model of zero-field NMR (the kind detected with an
atomic magnetometer, as pioneered by the Budker / Pines / Blanchard groups).
At *zero* magnetic field there is no Zeeman term: a set of coupled nuclear
spins evolves purely under the scalar J-coupling, and the detected
magnetization oscillates at the J-coupling frequency. Turn on a small leading
field B_z and the Zeeman term starts to compete with J -- the ZULF -> ULF
crossover, where the single J line shifts and splits.

Everything is built by hand with NumPy/SciPy so the physics stays explicit
(this doubles as teaching code): Pauli matrices -> spin operators -> Kronecker
products -> Hamiltonian -> propagation by diagonalization -> FFT.

Run it:
    python zulf_nmr.py
An interactive Matplotlib window opens with live sliders for J, B_z and T2,
plus a spin-system selector and a "Play" button that animates the FID. A
static snapshot of the default state is always written to ``zulf_demo.png`` so
the result is verifiable even in a headless run.

----------------------------------------------------------------------------
UNITS & THE ALL-IMPORTANT 2*pi
----------------------------------------------------------------------------
Frequencies are in Hz, time in seconds, hbar = 1. Because J is given in Hz and
t in seconds, every Hamiltonian term carries a factor of 2*pi so that the
propagator U(t) = exp(-i H t) produces oscillations at the right rate. Drop the
2*pi and the spectral peak lands at J/(2*pi) -- off by ~6.28x. The correctness
test is therefore simple: at B_z = 0, the spectral peak must sit at exactly
f = J (within the frequency resolution).
"""

import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Backend selection: use a non-interactive backend when there is no display
# (e.g. a headless CI run) so that we can still build the figure and save the
# PNG. With a display we keep the default interactive backend for the sliders.
# ---------------------------------------------------------------------------
import matplotlib


def _has_display():
    if sys.platform.startswith(("darwin", "win")):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


if not _has_display():
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.animation as manim  # noqa: E402
from matplotlib.widgets import Slider, Button, RadioButtons  # noqa: E402


# ===========================================================================
# Physical constants
# ===========================================================================
# Gyromagnetic ratios gamma / 2*pi, in MHz/T. With B_z expressed in microtesla
# this gives the Larmor frequency conveniently as nu[Hz] = gamma[MHz/T] * B[uT]
# (e.g. 1H at 1 uT -> 42.577 Hz, 13C at 1 uT -> 10.7084 Hz).
GAMMA = {"1H": 42.577, "13C": 10.7084}  # MHz/T
GAMMA_H = GAMMA["1H"]

# Time grid: dt = 0.5 ms, T = 3 s  ->  Nyquist = 1000 Hz, df ~ 0.33 Hz.
DT = 0.5e-3
T_TOTAL = 3.0
N_SAMPLES = int(round(T_TOTAL / DT))
TIME = np.arange(N_SAMPLES) * DT

# Look & feel: one restrained accent for the data, one marker colour for J.
ACCENT = "#2a9d8f"   # teal -- time signal and spectrum
MARKER = "#e76f51"   # muted coral -- the J reference line(s)
INK = "#22333b"      # near-black text


# ===========================================================================
# 1. SPIN OPERATORS
# ===========================================================================
# Pauli matrices. The spin-1/2 operators are S = sigma / 2 -- do NOT forget the
# factor of 1/2, it sets the eigenvalues of S_z to +/- 1/2.
_PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
_PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_SPIN = {"x": _PAULI_X / 2, "y": _PAULI_Y / 2, "z": _PAULI_Z / 2}
_ID2 = np.eye(2, dtype=complex)


def spin_op(i, k, n):
    """Single-spin operator I_i^k embedded in an n-spin Hilbert space (dim 2^n).

    Built as a Kronecker product  Id (x) ... (x) S^k (at slot i) (x) ... (x) Id,
    so that I_i^k acts as the spin operator S^k on spin i and as the identity
    on every other spin. ``k`` is one of 'x', 'y', 'z'.
    """
    ops = [_ID2] * n
    ops[i] = _SPIN[k]
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


# ===========================================================================
# 2. SPIN SYSTEM + HAMILTONIAN
# ===========================================================================
# Each preset is one 13C ("X") coupled to n equivalent 1H, the canonical ZULF
# building blocks. The H-H couplings are left at zero: equivalent protons are
# magnetically equivalent, so a mutual J only shifts whole spin manifolds and
# does not move the observed lines -- omitting it keeps the model clean.
SYSTEMS = {
    "XH":  [GAMMA["13C"], GAMMA["1H"]],
    "XH2": [GAMMA["13C"], GAMMA["1H"], GAMMA["1H"]],
    "XH3": [GAMMA["13C"], GAMMA["1H"], GAMMA["1H"], GAMMA["1H"]],
}


class SpinSystem:
    """A heteronuclear spin system with precomputed operators.

    Holds the per-site spin operators, the initial deviation density matrix
    rho0 and the detected observable M, then assembles the Hamiltonian for any
    (J, B_z) on demand. Building H this way -- explicit operators summed with
    their 2*pi factors -- keeps every term of the physics visible.
    """

    def __init__(self, name):
        self.name = name
        self.gammas = np.array(SYSTEMS[name], dtype=float)  # MHz/T, site 0 = 13C
        self.n = len(self.gammas)
        self.dim = 2 ** self.n
        # Couple site 0 (the X = 13C) to every proton with the same J.
        self.pairs = [(0, j) for j in range(1, self.n)]

        # Precompute the Cartesian spin operators for each site once.
        self.Ix = [spin_op(i, "x", self.n) for i in range(self.n)]
        self.Iy = [spin_op(i, "y", self.n) for i in range(self.n)]
        self.Iz = [spin_op(i, "z", self.n) for i in range(self.n)]

        # High-field prepolarization in the high-temperature limit gives a
        # deviation density matrix proportional to the gamma-weighted z spins:
        #   rho0 = sum_i (gamma_i / gamma_H) I_i^z .
        # The magnetometer detects the same gamma-weighted total z magnetization
        # (dominated by 1H), so M = rho0.
        #
        # PHYSICS CHECK: the signal is nonzero ONLY because the system is
        # heteronuclear (gamma_H != gamma_C). If all gammas were equal, rho0
        # would be proportional to the total F_z = sum_i I_i^z, which commutes
        # with H_J, so it would never evolve and the signal would be exactly
        # zero. The gamma weighting is what makes ZULF NMR observable.
        weights = self.gammas / GAMMA_H
        self.M = sum(w * Iz for w, Iz in zip(weights, self.Iz))
        self.rho0 = self.M.copy()

    def hamiltonian(self, J, Bz):
        """Hermitian Hamiltonian H = H_J + H_Z for coupling J (Hz), field Bz (uT).

        H_J = 2*pi * sum_{i<j} J_ij (Ix_i Ix_j + Iy_i Iy_j + Iz_i Iz_j)   (scalar J-coupling)
        H_Z = -2*pi * sum_i nu_i I_i^z,   nu_i = gamma_i * Bz  (Larmor freq, Hz)

        The 2*pi factors convert the Hz-valued couplings/frequencies into the
        angular rates that exp(-i H t) needs.
        """
        two_pi = 2.0 * np.pi
        H = np.zeros((self.dim, self.dim), dtype=complex)

        # Isotropic J-coupling (the only term that survives at zero field).
        for i, j in self.pairs:
            H += two_pi * J * (
                self.Ix[i] @ self.Ix[j]
                + self.Iy[i] @ self.Iy[j]
                + self.Iz[i] @ self.Iz[j]
            )

        # Leading Zeeman field along z (zero at B_z = 0).
        if Bz != 0.0:
            for i in range(self.n):
                nu_i = self.gammas[i] * Bz  # Hz
                H += -two_pi * nu_i * self.Iz[i]

        return H


# ===========================================================================
# 3. DYNAMICS & SIGNAL
# ===========================================================================
def evolve_signal(system, J, Bz, T2, t):
    """Time-domain signal S(t) = Re Tr[rho(t) M], with relaxation.

    The Hamiltonian is diagonalized ONCE, H = V diag(E) V^dagger, and the
    dynamics are evaluated in the eigenbasis. With rho0~ = V^dag rho0 V and
    M~ = V^dag M V, the trace becomes an exact sum of complex exponentials at
    the transition (Bohr) frequencies (E_n - E_m):

        S(t) = Re sum_{n,m} rho0~[n,m] M~[m,n] exp(-i (E_n - E_m) t).

    Each oscillation frequency is (E_n - E_m) / (2*pi) in Hz. This closed form
    is both fast (matrices are tiny) and exact (no time-stepping error).

    Two post-processing steps:
      * Remove the DC component (the part of the magnetization that commutes
        with H and so never evolves). A real magnetometer FID is AC-coupled;
        dropping DC leaves the clean oscillating signal whose peaks we want.
      * Apply a phenomenological relaxation envelope exp(-t / T2), which gives
        Lorentzian lines of width ~ 1/(pi*T2).
    """
    H = system.hamiltonian(J, Bz)
    E, V = np.linalg.eigh(H)          # H is Hermitian -> real E, unitary V
    Vd = V.conj().T

    rho0_e = Vd @ system.rho0 @ V
    M_e = Vd @ system.M @ V

    # amp[n,m] = rho0_e[n,m] * M_e[m,n];  omega[n,m] = E_n - E_m  (rad/s)
    amp = rho0_e * M_e.T
    omega = E[:, None] - E[None, :]

    phases = np.exp(-1j * np.outer(t, omega.ravel()))   # (Nt, dim^2)
    S = (phases @ amp.ravel()).real

    S = S - S.mean()                  # AC-couple: drop the static (DC) part
    S = S * np.exp(-t / T2)           # phenomenological T2 relaxation
    return S, E


def spectrum(S, dt):
    """Magnitude spectrum of S(t) via a real FFT. Returns (freqs_Hz, magnitude)."""
    mag = np.abs(np.fft.rfft(S))
    freqs = np.fft.rfftfreq(len(S), dt)
    return freqs, mag


def proton_total_spins(n_protons):
    """Distinct total-spin quantum numbers K for n equivalent spin-1/2 protons."""
    kmax = n_protons / 2.0
    ks, k = [], kmax
    while k >= 0:
        ks.append(k)
        k -= 1
    return ks


def zero_field_lines(n_protons, J):
    """Predicted zero-field line positions for an X-Hn system, in Hz.

    At B_z = 0 the Hamiltonian is 2*pi*J * S . K where S is the 13C spin and
    K = sum of proton spins. Within each proton manifold of total spin K, the
    observable connects the states F = K + 1/2 and F = K - 1/2, giving a line
    at f = J * (K + 1/2). Hence:
        XH  -> {J},   XH2 -> {1.5 J},   XH3 -> {J, 2J}  (the classic methyl pair).
    """
    return sorted({J * (K + 0.5) for K in proton_total_spins(n_protons) if K >= 0.5})


# ===========================================================================
# 4. (STRETCH) PROPER LINDBLAD RELAXATION SUPEROPERATOR
# ===========================================================================
# Scaffold for replacing the phenomenological exp(-t/T2) envelope with a real
# dissipative model. The Liouvillian L acts on a vectorized density matrix
# (column-stacking convention), rho(t) = expm(L t) @ vec(rho0). Independent
# transverse dephasing on each spin with rate 1/T2 reproduces Lorentzian lines.
# Kept here for completeness/teaching; the live UI uses the fast closed form
# above so dragging stays instant.
def lindblad_liouvillian(system, J, Bz, T2):
    """Vectorized Liouvillian L (dim^2 x dim^2): drho/dt = L rho, column-stacked."""
    from scipy.linalg import kron as spkron  # local import keeps SciPy optional

    H = system.hamiltonian(J, Bz)
    d = system.dim
    Id = np.eye(d, dtype=complex)
    # Coherent part: -i (I (x) H - H^T (x) I)   (column-stacking convention)
    L = -1j * (spkron(Id, H) - spkron(H.T, Id))
    # Dephasing collapse operators sqrt(rate) * sqrt(2) I_i^z give 1/T2 decay.
    rate = 1.0 / T2
    for Iz in system.Iz:
        C = np.sqrt(2.0 * rate) * Iz
        Cd = C.conj().T
        CdC = Cd @ C
        L += (spkron(C.conj(), C)
              - 0.5 * spkron(Id, CdC)
              - 0.5 * spkron(CdC.T, Id))
    return L


# ===========================================================================
# 5. INTERACTIVE UI
# ===========================================================================
DEFAULTS = dict(system="XH", J=140.0, Bz=0.0, T2=1.0)
TIME_WINDOW_S = 0.050   # show the first 50 ms of the FID
FREQ_MAX_HZ = 300.0     # spectrum view 0..300 Hz


def build_app(defaults=DEFAULTS):
    """Build the figure, axes, widgets and callbacks. Returns the Figure."""
    state = {
        "system": SpinSystem(defaults["system"]),
        "S": None,          # last computed (windowed) time signal, for Play
        "anim": None,       # keep a reference so the animation isn't GC'd
    }
    tmask = TIME <= TIME_WINDOW_S
    t_ms = TIME[tmask] * 1e3

    fig = plt.figure(figsize=(8.2, 7.6))
    fig.patch.set_facecolor("white")
    fig.suptitle("ZULF Lab — zero/ultralow-field NMR spin simulator",
                 fontsize=13, fontweight="bold", color=INK)

    ax_time = fig.add_axes([0.12, 0.66, 0.84, 0.24])
    ax_freq = fig.add_axes([0.12, 0.36, 0.84, 0.24])
    for ax in (ax_time, ax_freq):
        ax.set_facecolor("white")
        ax.grid(True, alpha=0.25)
        for s in ax.spines.values():
            s.set_color("#cccccc")
        ax.tick_params(colors=INK, labelsize=9)

    (line_time,) = ax_time.plot([], [], color=ACCENT, lw=1.6)
    ax_time.set_xlim(0, TIME_WINDOW_S * 1e3)
    ax_time.set_xlabel("time (ms)", color=INK)
    ax_time.set_ylabel("S(t)  —  z-magnetization (a.u.)", color=INK)
    ax_time.axhline(0, color="#bbbbbb", lw=0.8)

    (line_freq,) = ax_freq.plot([], [], color=ACCENT, lw=1.6)
    ax_freq.set_xlim(0, FREQ_MAX_HZ)
    ax_freq.set_xlabel("frequency (Hz)", color=INK)
    ax_freq.set_ylabel("spectrum |FFT| (a.u.)", color=INK)
    jline = ax_freq.axvline(defaults["J"], color=MARKER, lw=1.4, ls="--")
    jlabel = ax_freq.annotate("", xy=(0, 0), color=MARKER, fontsize=9,
                              fontweight="bold", ha="left", va="top")
    # faint reference lines for the other predicted zero-field positions
    aux_lines = [ax_freq.axvline(0, color=MARKER, lw=0.9, ls=":", alpha=0.0)
                 for _ in range(3)]

    # ---- sliders ----
    ax_J = fig.add_axes([0.14, 0.235, 0.58, 0.03])
    ax_B = fig.add_axes([0.14, 0.185, 0.58, 0.03])
    ax_T2 = fig.add_axes([0.14, 0.135, 0.58, 0.03])
    s_J = Slider(ax_J, "J (Hz)", 0.0, 300.0, valinit=defaults["J"],
                 color=ACCENT, valfmt="%.0f")
    s_B = Slider(ax_B, "B_z (µT)", 0.0, 5.0, valinit=defaults["Bz"],
                 color=ACCENT, valfmt="%.2f")
    s_T2 = Slider(ax_T2, "T2 (s)", 0.1, 5.0, valinit=defaults["T2"],
                  color=ACCENT, valfmt="%.2f")
    for s in (s_J, s_B, s_T2):
        s.label.set_color(INK)
        s.valtext.set_color(INK)

    # ---- system selector (radio) ----
    ax_radio = fig.add_axes([0.78, 0.135, 0.18, 0.13])
    ax_radio.set_title("spin system", fontsize=9, color=INK)
    radio = RadioButtons(ax_radio, ("XH", "XH2", "XH3"),
                         active=("XH", "XH2", "XH3").index(defaults["system"]))
    for lbl in radio.labels:
        lbl.set_color(INK)
        lbl.set_fontsize(9)

    # ---- buttons ----
    ax_play = fig.add_axes([0.14, 0.045, 0.16, 0.05])
    ax_reset = fig.add_axes([0.34, 0.045, 0.16, 0.05])
    b_play = Button(ax_play, "▶ Play FID", color="#e9f5f3", hovercolor="#cdeae6")
    b_reset = Button(ax_reset, "Reset", color="#f1f1f1", hovercolor="#dddddd")
    b_play.label.set_color(INK)
    b_reset.label.set_color(INK)

    # ---- the core update: recompute everything and redraw ----
    def recompute(_=None):
        system = state["system"]
        J, Bz, T2 = s_J.val, s_B.val, s_T2.val

        S, E = evolve_signal(system, J, Bz, T2, TIME)
        freqs, mag = spectrum(S, DT)

        Sw = S[tmask]
        state["S"] = Sw
        line_time.set_data(t_ms, Sw)
        ax_time.relim()
        ax_time.autoscale_view(scalex=False)

        line_freq.set_data(freqs, mag)
        ax_freq.relim()
        ax_freq.autoscale_view(scalex=False)

        # J marker + predicted zero-field lines (references; valid at B_z = 0)
        jline.set_xdata([J, J])
        ymax = mag.max() if mag.size else 1.0
        jlabel.set_text(f"J = {J:.0f} Hz")
        jlabel.set_position((J + 4, ymax * 0.97))
        zf = zero_field_lines(system.n - 1, J)
        others = [f for f in zf if abs(f - J) > 1e-6][:3]
        for k, aux in enumerate(aux_lines):
            if k < len(others) and others[k] <= FREQ_MAX_HZ:
                aux.set_xdata([others[k], others[k]])
                aux.set_alpha(0.55)
            else:
                aux.set_alpha(0.0)

        ax_time.set_title(
            f"{system.name}   |   J = {J:.0f} Hz,  B_z = {Bz:.2f} µT,  "
            f"T2 = {T2:.2f} s   |   linewidth ≈ {1/(np.pi*T2):.2f} Hz",
            fontsize=10, color=INK)
        fig.canvas.draw_idle()

    def on_system(label):
        state["system"] = SpinSystem(label)
        recompute()

    def on_reset(_):
        s_J.reset()
        s_B.reset()
        s_T2.reset()
        # radio has no clean programmatic reset across versions; leave as-is.
        recompute()

    def on_play(_):
        """Animate the FID building up point-by-point over the 50 ms window."""
        Sw = state["S"]
        if Sw is None:
            return
        nframes = 90
        idx = np.linspace(2, len(Sw), nframes).astype(int)

        def frame(k):
            j = idx[k]
            line_time.set_data(t_ms[:j], Sw[:j])
            return (line_time,)

        state["anim"] = manim.FuncAnimation(
            fig, frame, frames=nframes, interval=25, blit=False, repeat=False)
        fig.canvas.draw_idle()

    s_J.on_changed(recompute)
    s_B.on_changed(recompute)
    s_T2.on_changed(recompute)
    radio.on_clicked(on_system)
    b_reset.on_clicked(on_reset)
    b_play.on_clicked(on_play)

    recompute()
    # keep widget references alive on the figure
    fig._zulf_widgets = (s_J, s_B, s_T2, radio, b_play, b_reset)
    return fig


# ===========================================================================
# 6. SANITY CHECKS  (printed on startup -- the physics, made verifiable)
# ===========================================================================
def sanity_check():
    print("=" * 64)
    print("ZULF Lab sanity checks")
    print("=" * 64)

    sysXH = SpinSystem("XH")
    H = sysXH.hamiltonian(140.0, 0.0)
    herm_err = np.max(np.abs(H - H.conj().T))
    E, _ = np.linalg.eigh(H)
    print(f"[XH, J=140, B=0] Hermiticity error : {herm_err:.2e}")
    print(f"[XH, J=140, B=0] eigenvalues (Hz)  : "
          f"{np.round(E/(2*np.pi), 3)}")
    print("   -> expect singlet -105 Hz and triplet +35 Hz (gap = J = 140 Hz)")

    S, _ = evolve_signal(sysXH, 140.0, 0.0, 1.0, TIME)
    f, mag = spectrum(S, DT)
    peak = f[np.argmax(mag)]
    df = f[1] - f[0]
    ok = abs(peak - 140.0) <= df
    print(f"[XH, J=140, B=0] spectral peak     : {peak:.3f} Hz  "
          f"(expected 140, df={df:.3f})  -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        if abs(peak - 140.0 / (2 * np.pi)) <= df:
            print("   !! peak ~ J/(2*pi): a factor of 2*pi is missing.")

    # Heteronuclear requirement: force gamma_C = gamma_H -> signal must vanish.
    homo = SpinSystem("XH")
    homo.gammas = np.array([GAMMA_H, GAMMA_H])
    homo.M = sum((g / GAMMA_H) * Iz for g, Iz in zip(homo.gammas, homo.Iz))
    homo.rho0 = homo.M.copy()
    Sh, _ = evolve_signal(homo, 140.0, 0.0, 1.0, TIME)
    rms = np.sqrt(np.mean(Sh ** 2))
    print(f"[homonuclear gamma_C=gamma_H] signal RMS : {rms:.2e}  "
          f"-> {'PASS (zero)' if rms < 1e-9 else 'FAIL'}")

    for name in ("XH", "XH2", "XH3"):
        s = SpinSystem(name)
        print(f"[{name}] predicted zero-field lines (J=140): "
              f"{[round(x, 1) for x in zero_field_lines(s.n - 1, 140.0)]} Hz")
    print("=" * 64)


# ===========================================================================
# 7. MAIN
# ===========================================================================
def main():
    sanity_check()
    fig = build_app()

    # Always save a static snapshot of the default state (works headless too).
    out = "zulf_demo.png"
    fig.savefig(out, dpi=130, facecolor="white")
    print(f"Saved static snapshot -> {out}")

    if matplotlib.get_backend().lower() == "agg":
        print("No display detected: ran headless, skipping interactive window.")
    else:
        plt.show()


if __name__ == "__main__":
    main()
