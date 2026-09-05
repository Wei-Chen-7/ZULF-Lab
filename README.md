# ZULF Lab

An interactive **zero- to ultralow-field NMR spin simulator**, built from scratch
with NumPy/SciPy/Matplotlib (no qutip, no spin-dynamics libraries — the operators
and propagation are written by hand so the physics is explicit).

This is a scoped model of zero-field NMR (the kind detected with an atomic
magnetometer, as in the Budker/Pines/Blanchard work): at zero field there is no
Zeeman term, the coupled spins evolve under the scalar **J-coupling alone**, and the
detected magnetization oscillates at the J-coupling frequency. Drag a slider and
watch the spectrum reshape.

## Run

```bash
pip install numpy scipy matplotlib
python zulf_nmr.py
```

An interactive window opens with live sliders for **J**, **leading field B_z**, and
**T2**, a **spin-system selector**, a **relaxation-model toggle**, and a **Play**
button that animates the FID building up. A static snapshot of the default state is
always written to `zulf_demo.png` (so it also works headless).

![demo](zulf_demo.png)

### Spin systems

Two heteronuclear families, all ≤ 4 spins (Hilbert dimension ≤ 16): a **¹³C**
series (`XH`, `XH2`, `XH3`) and a **¹⁵N** series (`15NH`, `15NH2`, `15NH3`). They
share the same zero-field line pattern but behave differently in the ULF crossover
— ¹⁵N has a much smaller, *negative* gyromagnetic ratio, so its lines shift the
other way as B_z grows.

### Relaxation model (live toggle)

- **`exp(-t/T2)`** — fast phenomenological envelope; every line gets width ≈ 1/(πT2).
- **`Lindblad`** — a proper dissipative superoperator with independent transverse
  dephasing on each spin (collapse operators `√(2/T2)·Iᵢᶻ`). The Liouvillian is
  diagonalized once and the signal is summed over its damped eigenmodes, so
  linewidths *emerge from the model* and differ line-to-line (more dephasing spins
  ⇒ broader line). Instant for the 2–3 spin systems; ~75 ms/update for the 4-spin
  (dim-16) ones, since it diagonalizes a 256×256 matrix.

## Physics, in brief

- Units: frequencies in Hz, time in s, ħ = 1. Because J is in Hz and t in seconds,
  every Hamiltonian term carries a factor of **2π**, and `U(t) = exp(-i H t)`.
- `H = H_J + H_Z` with
  `H_J = 2π Σ_{i<j} J_ij (Ix_i Ix_j + Iy_i Iy_j + Iz_i Iz_j)` and
  `H_Z = -2π Σ_i ν_i I_i^z`, `ν_i = γ_i B_z`.
- Prepolarized initial state and detected observable are the γ-weighted z-spins:
  `ρ0 = M = Σ_i (γ_i/γ_H) I_i^z`.
- The signal is nonzero **only** because the system is heteronuclear (γ_H ≠ γ_C):
  if the γ's were equal, ρ0 ∝ total F_z, which commutes with H_J, and nothing would
  evolve.
- H is diagonalized once; `S(t) = Re Tr[ρ(t) M]` is then an exact sum of
  exponentials at the transition frequencies `(E_n − E_m)/2π`. A phenomenological
  `exp(-t/T2)` envelope gives Lorentzian lines of width ≈ `1/(πT2)`.

## `zulf_forward.py` — the inference forward model

`zulf_nmr.py` is the interactive teaching demo. `zulf_forward.py` is the
simulation engine for the simulation-based-inference project: no UI, built for
speed and generality.

```bash
python zulf_forward.py          # smoke demo over the molecule presets
pytest -q test_forward.py       # 32 validation tests
```

- **Line lists, not time grids.** One diagonalization gives exact (frequency,
  complex amplitude) pairs — the summary statistic the network needs, and ~240×
  faster than building an FID and FFT-ing it. **0.204 ms** for a four-spin
  spectrum; 10⁵ spectra in **21 s**.
- **Vector field.** `field_vector(|B|, θ)` — two parameters, since only the
  magnitude and the angle to the sensor axis matter. A longitudinal field shifts
  the line by 2.3×10⁻⁶ Hz at 1 nT; a transverse one restructures the spectrum at
  the 10-mHz scale.
- **General J matrix.** Any topology, not just X coupled to equivalent protons.
  Includes benzene-¹³C₁ (7 spins, ref [1]'s benchmark with its fitted couplings).
- **General ρ(0).** All three protocols: `rho_thermal` (sudden drop),
  `rho_adiabatic` (level-following, exact in the adiabatic limit and validated
  against an explicit ramp), and pulses in both the ref [1] hard-pulse and
  ref [14] finite-field conventions.
- **Detector response.** The 150 Hz sensor pole *and* the documented 6th-order
  500 Hz hardware low-pass, which is nearly flat in amplitude below 300 Hz but
  contributes ≈ −67° of differential phase across the J/2J band.

### Validated against the literature

Every test in `test_forward.py` encodes a published fact. Line positions
ν = J(I_A + ½) are confirmed three ways — Butler Eq. (40), Stern & Sheberstov's
odd/even rule, and Theis Eq. (1)–(2) — giving XA → J, XA₂ → 3/2 J, XA₃ → J & 2J
(the last two measured in refs [6] and [13]). Also encoded: the ΔI_A = 0
selection rule that makes J_HH invisible inside an equivalent group, the global
sign-flip and equal-γ permutation degeneracies, the transverse-field doublet
split by the *sum* of the Larmor frequencies with a line at their *mean*
(ref [13]), and the negative pulse-acquire weights at π/2 and π proton angles.

## `zulf_infer.py` — priors, peak-list summaries, NPE

```bash
python zulf_infer.py            # trains a single-round NPE on [13C]-formic acid
pytest -q test_infer.py         # 26 tests
```

Parameters are `(J, |B|, θ_B, T2)`. The coupling prior is a few Hz wide rather
than the full band, since DFT/ML predictors already fix ¹J_CH to ~1 Hz; the
nuisances get deliberately wide priors. The observation is a fixed-length
(frequency, amplitude, width) summary — a low-frequency Larmor group and a
high-frequency J multiplet, with multiplet positions as offsets from the prior
centre. Lines closer than a linewidth are merged, as a real spectrometer would.

Because the forward model is deterministic with additive Gaussian noise, the
likelihood is available in closed form, so NPE samples can be reweighted into an
exact posterior — the case for SBI here is global search and multimodality, not
intractability.

### First result (simulated data, [¹³C]-formic acid, 50k simulations)

| | J, 95% width | shrinkage |
|---|---|---|
| prior | 6000 mHz | — |
| raw NPE | 11.4 mHz | 528× |
| **after importance reweighting** | **2.18 mHz** | **2750×** |
| information floor (3 lines at σ_f = 1 mHz) | 2.26 mHz | — |

Reweighting reaches the information floor, and the nuisances are recovered
(B_θ = 54.97° against a true 55°). **Sample efficiency is 1.7%**, above the 1%
floor but not by much — the raw NPE proposal is ~5× wider than the true
posterior, so most samples land in the wings. This is the perfectly-specified
simulated case; real data will only lower it, so a tighter proposal is worth
having before then.

> These numbers are conditional on σ_f, the peak-position measurement error,
> which here stands in for the whole acquisition and SNR chain rather than being
> derived from a simulated FID.

### A third exact degeneracy

θ_B and 180° − θ_B give bit-identical spectra: R_x(π) maps **B** = (Bx, 0, Bz) to
(Bx, 0, −Bz) and sends both ρ(0) = M and M to −M, leaving S(t) = Tr[ρ(t)M]
unchanged, so only |cos θ_B| is identifiable. This sits alongside the global sign
flip and the equal-γ permutations, and is broken here by capping the prior at
90°. Before that cap the angle posterior was bimodal at 55° and 125° with a mean
of 90° — a posterior mean over a symmetric two-peaked distribution, which is a
number with no meaning.

## Correctness test

At **B_z = 0 the spectral peak sits at exactly f = J** (within the ~0.33 Hz
resolution). If it lands at J/2π (off by ~6.28×), a factor of 2π is missing; if
there is no peak at all, the system isn't heteronuclear. The predicted zero-field
lines are XH → {J}, XH2 → {1.5 J}, XH3 → {J, 2J} (the classic methyl pair). These
checks print on every run.
