"""Tests for the sigma_f study.

Deliberately small trial counts and loose tolerances: these check that the
measurement machinery behaves the way frequency estimation must, not that it
reproduces a particular number to three digits.
"""

import numpy as np

import sigma_f_study as sf


def test_synth_spectrum_puts_the_line_where_it_belongs():
    import zulf_forward as zf
    J = 222.2
    s = zf.SpinSystem(["13C", "1H"], np.array([[0.0, J], [J, 0.0]]))
    f, a = zf.line_list(s)
    rng = np.random.default_rng(0)
    grid, spec = sf.synth_noisy_spectrum(f, a, T2=10.0, t_acq=10.0, snr=200.0,
                                         rng=rng)
    assert abs(grid[np.argmax(spec)] - J) < 0.05


def test_fit_recovers_the_centre_of_a_clean_line():
    import zulf_forward as zf
    J = 222.2
    s = zf.SpinSystem(["13C", "1H"], np.array([[0.0, J], [J, 0.0]]))
    f, a = zf.line_list(s)
    rng = np.random.default_rng(1)
    grid, spec = sf.synth_noisy_spectrum(f, a, T2=10.0, t_acq=10.0, snr=1000.0,
                                         rng=rng)
    got = sf.fit_line_centre(grid, spec, J, half_window=0.3)
    assert abs(got - J) < 5e-3


def test_sigma_f_is_unbiased():
    """The fitted centre must scatter about the truth, not away from it."""
    s, bias, n = sf.measure_sigma_f(T2=10.0, t_acq=10.0, snr=100.0,
                                    n_trials=40, seed=2)
    assert n >= 35
    assert abs(bias) < 0.5 * s, f"bias {bias*1e3:.2f} mHz vs sigma {s*1e3:.2f}"


def test_sigma_f_scales_inversely_with_snr():
    a, _, _ = sf.measure_sigma_f(T2=10.0, t_acq=10.0, snr=50.0,
                                 n_trials=40, seed=3)
    b, _, _ = sf.measure_sigma_f(T2=10.0, t_acq=10.0, snr=200.0,
                                 n_trials=40, seed=3)
    assert np.isclose(a / b, 4.0, rtol=0.35), f"ratio {a/b:.2f}, expected ~4"


def test_sigma_f_improves_with_acquisition_time():
    short, _, _ = sf.measure_sigma_f(T2=10.0, t_acq=2.0, snr=100.0,
                                     n_trials=30, seed=4)
    long, _, _ = sf.measure_sigma_f(T2=10.0, t_acq=20.0, snr=100.0,
                                    n_trials=30, seed=4)
    assert long < short


def test_sigma_f_tracks_the_linewidth():
    """Halving the linewidth (doubling T2) should roughly halve sigma_f."""
    broad, _, _ = sf.measure_sigma_f(T2=5.0, t_acq=20.0, snr=100.0,
                                     n_trials=30, seed=5)
    narrow, _, _ = sf.measure_sigma_f(T2=10.0, t_acq=20.0, snr=100.0,
                                      n_trials=30, seed=5)
    assert narrow < broad
    assert np.isclose(broad / narrow, 2.0, rtol=0.6)


def test_the_assumed_one_millihertz_is_reachable_at_modest_snr():
    """The inference work assumed sigma_f = 1 mHz; check that is not optimistic."""
    s, _, _ = sf.measure_sigma_f(T2=10.0, t_acq=20.0, snr=30.0,
                                 n_trials=40, seed=6)
    assert s * 1e3 < 1.5, f"sigma_f {s*1e3:.2f} mHz at SNR 30"
