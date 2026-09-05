#!/usr/bin/env python3
"""Simulation-based inference for ZULF J-couplings: priors, summaries, NPE.

This is the inference layer on top of :mod:`zulf_forward`. It implements the
three pieces the project needs before any network can be trained:

1. **Priors.** Not flat. DFT/ML predictors give 1J_CH to about 1 Hz, which is
   useless as a measurement but excellent as a prior, so the coupling prior is
   a few Hz wide rather than the full 0-300 Hz band. Nuisance parameters
   (residual field magnitude and angle, relaxation) get deliberately wide
   priors, since a network trained on a world messier than the real one
   transfers and one trained on a tidy world does not.

2. **Peak-list summaries, not raw spectra.** A 10 mHz linewidth over a 500 Hz
   band needs ~200,000 bins, and asking a flow to resolve one part in 1e6 of
   that vector is the known failure mode. Instead the observation is a short
   vector of (frequency, amplitude, width) built from the exact line list.
   Lines closer together than a linewidth are merged first, because a real
   spectrometer cannot separate them either.

3. **Amortized NPE, single round.** Sequential methods tune the proposal to one
   observation and lose amortization; a single round keeps it, at the cost of
   more simulations. Simulations are cheap here (0.12 ms for a two-spin
   spectrum), so that trade is easy.

The summary is deliberately structured rather than a flat list of peaks: a
low-frequency group (Larmor precession, near 0 Hz) and a high-frequency group
(the J multiplet), with high-frequency positions expressed as offsets from the
prior centre so the network sees O(0.1 Hz) numbers instead of O(200 Hz) ones.

Run ``python zulf_infer.py`` for a self-contained demo on [13C]-formic acid.
"""

from __future__ import annotations

import numpy as np

import zulf_forward as zf

__all__ = [
    "InferenceProblem", "PARAM_NAMES", "merge_lines", "peak_summary",
]

PARAM_NAMES = ("J_Hz", "B_nT", "B_theta_deg", "T2_s")

# log10 amplitude assigned to an absent line. A missing line genuinely means
# zero amplitude, so flooring the amplitude (rather than inventing a frequency)
# is the physically meaningful padding: the network learns "floor => absent".
_LOGAMP_FLOOR = -6.0


def merge_lines(freqs, amps, width_hz):
    """Merge lines that a spectrometer of the given linewidth cannot separate.

    Components closer than ``width_hz`` are replaced by a single line at their
    amplitude-weighted mean frequency with the summed amplitude. Without this
    the summary would contain pairs of formally distinct but experimentally
    indistinguishable lines (for a 13C-1H pair in a transverse field, the two
    low-frequency components differ by ~1e-6 Hz).
    """
    if len(freqs) == 0:
        return freqs, amps
    order = np.argsort(freqs)
    f, a = freqs[order], amps[order]
    out_f, out_a = [], []
    cf, ca = [f[0]], [a[0]]
    for k in range(1, len(f)):
        if f[k] - cf[-1] <= width_hz:
            cf.append(f[k])
            ca.append(a[k])
        else:
            w = np.abs(ca)
            out_f.append(np.average(cf, weights=w) if w.sum() else np.mean(cf))
            out_a.append(np.sum(ca))
            cf, ca = [f[k]], [a[k]]
    w = np.abs(ca)
    out_f.append(np.average(cf, weights=w) if w.sum() else np.mean(cf))
    out_a.append(np.sum(ca))
    return np.array(out_f), np.array(out_a)


def peak_summary(freqs, amps, width_hz, J_center, n_low=1, n_high=3,
                 split_hz=None):
    """Fixed-length summary vector from a line list.

    Layout (``n_low=1``, ``n_high=3`` gives length 9)::

        [f_low ..., log10|a_low| ...,
         (f_high - J_center) ..., log10|a_high| ...,
         log10(width)]

    The high-frequency positions are offsets from the prior centre, so the
    network sees numbers of order 0.1 Hz rather than 200 Hz. Groups are sorted
    by frequency and padded with an amplitude floor.
    """
    if split_hz is None:
        split_hz = 0.5 * J_center
    f, a = merge_lines(np.asarray(freqs), np.asarray(amps), width_hz)
    mag = np.abs(a)

    def _group(mask, n, offset):
        fs, ms = f[mask], mag[mask]
        if len(fs) > n:                        # keep the strongest n
            idx = np.argsort(-ms)[:n]
            fs, ms = fs[idx], ms[idx]
        idx = np.argsort(fs)
        fs, ms = fs[idx], ms[idx]
        pf = np.zeros(n)
        pa = np.full(n, _LOGAMP_FLOOR)
        if len(fs):
            pf[:len(fs)] = fs - offset
            pa[:len(fs)] = np.log10(np.maximum(ms, 10.0 ** _LOGAMP_FLOOR))
        return pf, pa

    lf, la = _group(f < split_hz, n_low, 0.0)
    hf, ha = _group(f >= split_hz, n_high, J_center)
    return np.concatenate([lf, la, hf, ha, [np.log10(width_hz)]])


class InferenceProblem:
    """A ZULF inference problem: prior, simulator and summary in one object.

    Parameters
    ----------
    system : str
        Name of a preset in :data:`zulf_forward.SYSTEMS`.
    J_center : float
        Centre of the coupling prior, i.e. the DFT/ML predicted value.
    J_half : float
        Half-width of the coupling prior in Hz. Ref [18] quotes ~1 Hz accuracy
        for 1J_CH, so a few Hz is a faithful and still conservative choice.
    B_max_nT, T2_range :
        Wide nuisance priors on the residual field and relaxation.
    sigma_f : float
        Standard deviation of the peak-position measurement error, in Hz. This
        stands in for the whole acquisition and SNR chain: published ZULF work
        reaches ~1 mHz on a fitted line, so 1e-3 is the realistic default.
    sigma_logamp : float
        Standard deviation of the log10 amplitude error.
    """

    def __init__(self, system="formic_acid", J_center=222.2, J_half=3.0,
                 B_max_nT=2.0, T2_range=(3.0, 40.0),
                 sigma_f=1e-3, sigma_logamp=0.03, seed=0):
        self.sys = zf.build_system(system)
        if self.sys.n != 2:
            raise NotImplementedError(
                "the two-spin problem is the one the failure criterion names; "
                "larger systems need a richer parameterization of J")
        self.J_center = float(J_center)
        self.low = np.array([J_center - J_half, 0.0, 0.0, T2_range[0]])
        self.high = np.array([J_center + J_half, B_max_nT, 180.0, T2_range[1]])
        self.sigma_f = float(sigma_f)
        self.sigma_logamp = float(sigma_logamp)
        self.rng = np.random.default_rng(seed)

    # -- prior -------------------------------------------------------------
    def prior(self):
        """A box-uniform prior over ``PARAM_NAMES`` as a torch distribution."""
        import torch
        from sbi.utils import BoxUniform
        return BoxUniform(low=torch.as_tensor(self.low, dtype=torch.float32),
                          high=torch.as_tensor(self.high, dtype=torch.float32))

    def sample_prior(self, n):
        return self.rng.uniform(self.low, self.high, size=(n, len(self.low)))

    # -- simulator ---------------------------------------------------------
    def simulate_one(self, theta, noisy=True):
        """One parameter vector -> one summary vector."""
        J, B_nT, ang, T2 = (float(v) for v in theta)
        Jm = np.array([[0.0, J], [J, 0.0]])
        B = zf.field_vector(B_nT * 1e-3, ang)          # nT -> uT
        f, a = zf.line_list(self.sys, B=B, J=Jm)
        width = 1.0 / (np.pi * T2)                     # absorption FWHM
        x = peak_summary(f, a, width, self.J_center)
        if noisy:
            n_low, n_high = 1, 3
            x = x.copy()
            # frequency slots: positions 0..n_low-1 and 2*n_low..2*n_low+n_high-1
            fslots = list(range(n_low)) + \
                list(range(2 * n_low, 2 * n_low + n_high))
            aslots = list(range(n_low, 2 * n_low)) + \
                list(range(2 * n_low + n_high, 2 * n_low + 2 * n_high))
            x[fslots] += self.rng.normal(0.0, self.sigma_f, len(fslots))
            present = x[aslots] > _LOGAMP_FLOOR + 1e-9
            noise = self.rng.normal(0.0, self.sigma_logamp, len(aslots))
            x[aslots] = np.where(present, x[aslots] + noise, x[aslots])
            x[-1] += self.rng.normal(0.0, self.sigma_logamp)
        return x

    def simulate(self, thetas, noisy=True):
        return np.stack([self.simulate_one(t, noisy) for t in np.asarray(thetas)])

    # -- convenience -------------------------------------------------------
    def x_dim(self):
        return len(self.simulate_one(self.sample_prior(1)[0], noisy=False))


# ===========================================================================
# Demo / first-network experiment
# ===========================================================================
def _train_and_report(n_sims=20000, sigma_f=1e-3, seed=0, epochs=None,
                      verbose=True):
    """Train a single-round NPE and report the posterior width on J."""
    import torch
    from sbi.inference import NPE

    torch.manual_seed(seed)
    prob = InferenceProblem(sigma_f=sigma_f, seed=seed)

    theta = prob.sample_prior(n_sims)
    x = prob.simulate(theta)

    inference = NPE(prior=prob.prior(), density_estimator="nsf")
    inference.append_simulations(torch.as_tensor(theta, dtype=torch.float32),
                                 torch.as_tensor(x, dtype=torch.float32))
    kw = {"show_train_summary": verbose}
    if epochs:
        kw["max_num_epochs"] = epochs
    inference.train(**kw)
    posterior = inference.build_posterior()

    # A representative observation: mid-prior J, a real residual field.
    theta_true = np.array([prob.J_center + 0.7, 1.0, 55.0, 12.0])
    x_obs = prob.simulate_one(theta_true)
    samples = posterior.sample((4000,),
                               x=torch.as_tensor(x_obs, dtype=torch.float32),
                               show_progress_bars=False).numpy()
    return prob, theta_true, samples


def main():  # pragma: no cover - demo
    import torch
    np.set_printoptions(suppress=True)
    prob, theta_true, s = _train_and_report()

    print("\n" + "=" * 70)
    print("First NPE posterior, [13C]-formic acid (2 spins)")
    print("=" * 70)
    for i, name in enumerate(PARAM_NAMES):
        lo, hi = np.percentile(s[:, i], [2.5, 97.5])
        print(f"  {name:12s} true {theta_true[i]:9.4f} | "
              f"post {s[:, i].mean():9.4f} +/- {s[:, i].std():8.4f} | "
              f"95% [{lo:.4f}, {hi:.4f}]")
    width_mHz = (np.percentile(s[:, 0], 97.5) - np.percentile(s[:, 0], 2.5)) * 1e3
    prior_mHz = (prob.high[0] - prob.low[0]) * 1e3
    print(f"\n  prior width on J : {prior_mHz:9.1f} mHz")
    print(f"  95% posterior    : {width_mHz:9.1f} mHz")
    print(f"  shrinkage factor : {prior_mHz / width_mHz:9.1f}x")
    print(f"  failure criterion: posterior on J wider than 10 mHz -> "
          f"{'FAIL' if width_mHz > 10 else 'PASS'}")


if __name__ == "__main__":
    main()
