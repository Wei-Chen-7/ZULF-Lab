#!/usr/bin/env python3
"""How well can a line position actually be measured? Closing the sigma_f gap.

Every inference result so far is conditional on sigma_f, the peak-position
measurement error, which was *assumed* to be 1 mHz. That assumption stands in
for the whole acquisition and SNR chain. This script measures it instead:
synthesize an FID at a given SNR, acquisition time and T2, Fourier transform it,
fit the line, and look at the scatter of the fitted centre over many noise
realizations.

The output is a curve rather than a single number, which is the more useful
object: when real acquisition parameters arrive, you read off the point rather
than re-running the study.

Definitions
-----------
SNR is the usual NMR one: peak height in the magnitude spectrum divided by the
RMS of the magnitude-spectrum baseline away from any line. That is deliberately
the quantity you can read straight off a measured spectrum, which is what makes
the resulting table usable against real data.

Note the parameterization: because SNR is fixed *in the spectrum*, lengthening
the acquisition does not simply add noise -- it also narrows the line, so
sigma_f keeps improving roughly as 1/sqrt(T_acq) with no plateau at T_acq ~ T2.
(An earlier version of this docstring predicted such a plateau; the measurement
below shows there is none under this parameterization. Holding the *time-domain*
noise density fixed instead would be the way to ask "how long is it worth
acquiring for", which is a different question and a separate study.)

Empirical result
----------------
Across T2 = 3-30 s and SNR = 30-300 the measurements collapse onto

    sigma_f  ~  FWHM / (1.3 x SNR),        FWHM = 1/(pi T2)

with the coefficient drifting from about 1.5 to 1.1 over that range. So the
1 mHz that the inference work assumed corresponds to only SNR ~ 24 at T2 = 10 s
-- a modest spectrum. The assumption was conservative, not optimistic.

Caveats: this is a single isolated line, fitted with the correct lineshape, on a
flat baseline. Real spectra have overlapping multiplets, baseline drift and mains
harmonics (ref [1] excised 2 Hz around every multiple of 60 Hz), all of which
make the achievable sigma_f worse than this idealized figure.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

import zulf_forward as zf

SAMPLE_RATE = 1000.0          # Hz; Nyquist 500 Hz covers the J lines of interest


def synth_noisy_spectrum(freqs, amps, T2, t_acq, snr, rng,
                         sample_rate=SAMPLE_RATE, zero_fill=8):
    """FID -> noise -> FFT. Returns (grid_Hz, magnitude spectrum)."""
    dt = 1.0 / sample_rate
    n = int(round(t_acq / dt))
    t = np.arange(n) * dt
    fid = zf.synth_fid(freqs, amps, t, T2=T2)

    # Noise is added in the time domain, then scaled so the *spectrum* has the
    # requested peak-height-to-baseline-RMS ratio.
    noise = rng.normal(size=n)
    clean = np.abs(np.fft.rfft(fid, n=zero_fill * n))
    nspec = np.abs(np.fft.rfft(noise, n=zero_fill * n))
    scale = (clean.max() / snr) / nspec.mean()
    spec = np.abs(np.fft.rfft(fid + scale * noise, n=zero_fill * n))
    grid = np.fft.rfftfreq(zero_fill * n, dt)
    return grid, spec


def _lorentz_mag(f, f0, gamma, a, base):
    return a / np.sqrt((f - f0) ** 2 + gamma ** 2) + base


def fit_line_centre(grid, spec, f_guess, half_window):
    """Fit a magnitude Lorentzian near ``f_guess``; return the centre in Hz."""
    m = np.abs(grid - f_guess) < half_window
    x, y = grid[m], spec[m]
    if len(x) < 8:
        return np.nan
    gamma0 = max(half_window / 10, 1e-3)
    p0 = [x[np.argmax(y)], gamma0, y.max() * gamma0, np.median(y)]
    try:
        p, _ = curve_fit(_lorentz_mag, x, y, p0=p0, maxfev=20000)
    except Exception:
        return np.nan
    return p[0]


def measure_sigma_f(J=222.2, T2=10.0, t_acq=20.0, snr=100.0, n_trials=60,
                    seed=0):
    """Empirical std of the fitted line centre, in Hz."""
    sysx = zf.SpinSystem(["13C", "1H"], np.array([[0.0, J], [J, 0.0]]))
    f, a = zf.line_list(sysx)
    rng = np.random.default_rng(seed)
    fwhm = 1.0 / (np.pi * T2)
    half_window = max(8 * fwhm, 20.0 / t_acq)      # cover the line and a few bins
    got = []
    for _ in range(n_trials):
        grid, spec = synth_noisy_spectrum(f, a, T2, t_acq, snr, rng)
        got.append(fit_line_centre(grid, spec, J, half_window))
    got = np.array(got)
    got = got[np.isfinite(got)]
    if len(got) < 5:
        return np.nan, np.nan, len(got)
    return float(np.std(got)), float(np.mean(got) - J), len(got)


def main():  # pragma: no cover - study
    J = 222.2
    print("=" * 78)
    print("sigma_f: measured scatter of a fitted line centre, [13C]-formic acid")
    print(f"J = {J} Hz, sample rate {SAMPLE_RATE:.0f} Hz")
    print("=" * 78)

    print("\n(a) sigma_f vs acquisition time, at T2 = 10 s (FWHM 31.8 mHz), SNR = 100")
    print(f"{'T_acq (s)':>10} {'T_acq/T2':>9} {'sigma_f (mHz)':>14} {'bias (mHz)':>12}")
    for t_acq in (1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0):
        s, b, n = measure_sigma_f(J=J, T2=10.0, t_acq=t_acq, snr=100.0)
        print(f"{t_acq:>10.1f} {t_acq/10.0:>9.1f} {s*1e3:>14.2f} {b*1e3:>12.2f}")

    print("\n(b) sigma_f vs SNR, at T2 = 10 s, T_acq = 20 s")
    print(f"{'SNR':>10} {'sigma_f (mHz)':>14} {'x SNR':>10}")
    for snr in (10.0, 30.0, 100.0, 300.0, 1000.0):
        s, b, n = measure_sigma_f(J=J, T2=10.0, t_acq=20.0, snr=snr)
        print(f"{snr:>10.0f} {s*1e3:>14.2f} {s*1e3*snr:>10.1f}")
    print("   (constant last column => sigma_f scales as 1/SNR)")

    print("\n(c) sigma_f vs T2 (linewidth), at T_acq = 2 T2, SNR = 100")
    print(f"{'T2 (s)':>8} {'FWHM (mHz)':>12} {'T_acq (s)':>10} {'sigma_f (mHz)':>14}")
    for T2 in (1.0, 3.0, 10.0, 30.0):
        s, b, n = measure_sigma_f(J=J, T2=T2, t_acq=2 * T2, snr=100.0)
        print(f"{T2:>8.1f} {1e3/(np.pi*T2):>12.1f} {2*T2:>10.1f} {s*1e3:>14.2f}")

    print("\n(d) the scaling law: sigma_f ~ FWHM / (k x SNR)")
    ks = []
    print(f"{'T2 (s)':>8} {'SNR':>7} {'k':>8}")
    for T2 in (3.0, 10.0, 30.0):
        for snr in (30.0, 100.0, 300.0):
            s, _, _ = measure_sigma_f(J=J, T2=T2, t_acq=2 * T2, snr=snr,
                                      n_trials=40)
            k = (1e3 / (np.pi * T2)) / (s * 1e3 * snr)
            ks.append(k)
            print(f"{T2:>8.0f} {snr:>7.0f} {k:>8.2f}")
    kbar = float(np.mean(ks))
    print(f"   k = {kbar:.2f} +/- {np.std(ks):.2f}")

    print("\nWhat this means for the failure criterion")
    print("-" * 78)
    print(f"  The inference work ASSUMED sigma_f = 1 mHz. By the law above that")
    print(f"  corresponds to only SNR ~ {(1e3/(np.pi*10.0))/(kbar*1.0):.0f} at T2 = 10 s, so the")
    print(f"  assumption was conservative rather than optimistic.\n")
    for T2, t_acq, snr in ((10.0, 20.0, 100.0), (30.0, 60.0, 300.0)):
        s, _, _ = measure_sigma_f(J=J, T2=T2, t_acq=t_acq, snr=snr)
        floor = 2 * 1.96 * s / np.sqrt(3) * 1e3     # 3 multiplet lines
        print(f"  T2={T2:.0f}s, T_acq={t_acq:.0f}s, SNR={snr:.0f}: "
              f"sigma_f={s*1e3:.2f} mHz -> J floor {floor:.2f} mHz "
              f"({'PASS' if floor < 10 else 'FAIL'} vs 10 mHz)")
    print("\n  Idealized: one isolated line, correct lineshape, flat baseline.")
    print("  Overlapping multiplets and baseline drift make real sigma_f worse.")


if __name__ == "__main__":
    main()
