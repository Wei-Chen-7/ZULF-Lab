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

#: Default parameter names (the two-spin case). A problem's own ordering is
#: always ``prob.param_names``, which grows with the coupling classes.
PARAM_NAMES = ("J_CH", "B_nT", "B_theta_deg", "T2_s")

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


#: Symmetry-distinct coupling classes per system, as
#: ``(name, [(i, j), ...], prior centre in Hz, prior half-width in Hz)``.
#:
#: Parameterizing by symmetry-distinct couplings rather than by all N(N-1)/2
#: pairs is the point: a posterior over every pair would report the prior back
#: in the flat directions while looking like a result. The J_HH class inside an
#: equivalent proton group is kept deliberately, with a wide prior, because it
#: is the textbook flat direction -- the Delta I_A = 0 selection rule makes it
#: move no line at all -- and the posterior should say so out loud.
COUPLING_CLASSES = {
    "formic_acid": [("J_CH", [(0, 1)], 222.2, 3.0)],
    "glycine": [("J_CH", [(0, 1), (0, 2)], 140.0, 3.0),
                ("J_HH", [(1, 2)], 0.0, 15.0)],
    "methanol": [("J_CH", [(0, 1), (0, 2), (0, 3)], 141.0, 3.0),
                 ("J_HH", [(1, 2), (1, 3), (2, 3)], 0.0, 15.0)],
    "formaldehyde": [("J_CH", [(0, 1), (0, 2)], 163.9, 3.0),
                     ("J_HH", [(1, 2)], 0.0, 15.0)],
}

#: How many lines to keep in each summary group, per system.
SUMMARY_SHAPE = {
    "formic_acid": (1, 3),
    "glycine": (1, 4),
    "formaldehyde": (1, 4),
    "methanol": (2, 6),
}


class InferenceProblem:
    """A ZULF inference problem: prior, simulator and summary in one object.

    Parameters
    ----------
    system : str
        Name of a preset in :data:`COUPLING_CLASSES`. The free couplings are
        that system's symmetry-distinct classes, followed by the three nuisance
        parameters (residual field magnitude and angle, relaxation).
    classes : list, optional
        Override the coupling classes, in the format of
        :data:`COUPLING_CLASSES`. Use this for a molecule with no preset.
    B_max_nT, T2_range :
        Wide nuisance priors on the residual field and relaxation.
    sigma_f : float
        Standard deviation of the peak-position measurement error, in Hz. This
        stands in for the whole acquisition and SNR chain. Measured in
        ``sigma_f_study.py`` to be about FWHM/(1.3 x SNR), so the 1 mHz default
        corresponds to roughly SNR 24 at T2 = 10 s -- a conservative choice.
    sigma_logamp : float
        Standard deviation of the log10 amplitude error.
    """

    def __init__(self, system="formic_acid", classes=None, J_center=None,
                 J_half=None, B_max_nT=2.0, T2_range=(3.0, 40.0),
                 sigma_f=1e-3, sigma_logamp=0.03, seed=0, theta_max_deg=90.0,
                 summary_shape=None):
        self.system = system
        self.sys = zf.build_system(system)
        self.classes = list(classes if classes is not None
                            else COUPLING_CLASSES[system])
        if J_center is not None or J_half is not None:      # tweak the first class
            name, pairs, c, h = self.classes[0]
            self.classes[0] = (name, pairs,
                               c if J_center is None else float(J_center),
                               h if J_half is None else float(J_half))
        self.n_low, self.n_high = (summary_shape or
                                   SUMMARY_SHAPE.get(system, (1, 4)))
        # Reference for the high-frequency offsets: the centre of the dominant
        # coupling prior, so the network sees numbers of order 0.1 Hz.
        self.J_center = float(self.classes[0][2])

        # The field angle prior stops at 90 degrees on purpose. theta and
        # 180 - theta give bit-identical spectra: R_x(pi) maps B=(Bx,0,Bz) to
        # (Bx,0,-Bz) and sends both rho(0)=M and M to -M, leaving
        # S(t) = Tr[rho(t) M] unchanged, so only |cos theta| is identifiable.
        # Restricting the prior removes the degeneracy by convention, exactly
        # as a canonical ordering removes the equal-gamma permutations. Without
        # it the posterior is bimodal and its mean is a meaningless number.
        jlo = [c - h for _, _, c, h in self.classes]
        jhi = [c + h for _, _, c, h in self.classes]
        self.low = np.array(jlo + [0.0, 0.0, T2_range[0]])
        self.high = np.array(jhi + [B_max_nT, theta_max_deg, T2_range[1]])
        self.param_names = [n for n, _, _, _ in self.classes] + \
            ["B_nT", "B_theta_deg", "T2_s"]
        self.n_couplings = len(self.classes)
        self.sigma_f = float(sigma_f)
        self.sigma_logamp = float(sigma_logamp)
        self.rng = np.random.default_rng(seed)

    def J_matrix(self, theta):
        """Build the full J matrix from the coupling-class values in theta."""
        n = self.sys.n
        J = np.zeros((n, n))
        for k, (_, pairs, _, _) in enumerate(self.classes):
            for i, j in pairs:
                J[i, j] = J[j, i] = float(theta[k])
        return J

    def prior_span(self):
        return self.high - self.low

    # -- prior -------------------------------------------------------------
    def prior(self):
        """A box-uniform prior over ``self.param_names`` as a torch distribution."""
        import torch
        from sbi.utils import BoxUniform
        return BoxUniform(low=torch.as_tensor(self.low, dtype=torch.float32),
                          high=torch.as_tensor(self.high, dtype=torch.float32))

    def sample_prior(self, n):
        return self.rng.uniform(self.low, self.high, size=(n, len(self.low)))

    # -- simulator ---------------------------------------------------------
    def slot_sigmas(self):
        """Per-slot noise scale, matching :meth:`simulate_one` exactly.

        Every slot is perturbed, including padded ones, so the likelihood is a
        plain Gaussian with no delta functions. That is what makes the exact
        log-likelihood below available, and hence importance reweighting.
        """
        nl, nh = self.n_low, self.n_high
        sig = np.empty(2 * nl + 2 * nh + 1)
        sig[:nl] = self.sigma_f                       # low-frequency positions
        sig[nl:2 * nl] = self.sigma_logamp            # low-frequency amplitudes
        sig[2 * nl:2 * nl + nh] = self.sigma_f        # multiplet positions
        sig[2 * nl + nh:2 * nl + 2 * nh] = self.sigma_logamp
        sig[-1] = self.sigma_logamp                   # log width
        return sig

    def simulate_one(self, theta, noisy=True):
        """One parameter vector -> one summary vector."""
        theta = np.asarray(theta, float)
        Jm = self.J_matrix(theta)
        B_nT, ang, T2 = (float(v) for v in theta[self.n_couplings:])
        B = zf.field_vector(B_nT * 1e-3, ang)          # nT -> uT
        f, a = zf.line_list(self.sys, B=B, J=Jm)
        width = 1.0 / (np.pi * T2)                     # absorption FWHM
        x = peak_summary(f, a, width, self.J_center,
                         n_low=self.n_low, n_high=self.n_high)
        if noisy:
            x = x + self.rng.normal(0.0, self.slot_sigmas())
        return x

    def simulate(self, thetas, noisy=True):
        return np.stack([self.simulate_one(t, noisy) for t in np.asarray(thetas)])

    # -- exact likelihood --------------------------------------------------
    def log_likelihood(self, x_obs, thetas):
        """log p(x_obs | theta), exactly, for one or many theta.

        The forward model is deterministic and the noise is additive Gaussian,
        so the likelihood is available in closed form at the cost of one
        simulation. This is worth stating plainly: the case for SBI here is not
        intractability, it is that the network searches globally and can hold
        several separated solutions at once. Having the likelihood is what lets
        the network samples be reweighted into an exact posterior.
        """
        thetas = np.atleast_2d(np.asarray(thetas, float))
        sig = self.slot_sigmas()
        mu = np.stack([self.simulate_one(t, noisy=False) for t in thetas])
        r = (np.asarray(x_obs, float)[None, :] - mu) / sig
        return -0.5 * np.sum(r * r, axis=1) - np.sum(np.log(sig)) \
            - 0.5 * len(sig) * np.log(2 * np.pi)

    def in_prior(self, thetas):
        thetas = np.atleast_2d(np.asarray(thetas, float))
        return np.all((thetas >= self.low) & (thetas <= self.high), axis=1)

    # -- convenience -------------------------------------------------------
    def x_dim(self):
        return len(self.simulate_one(self.sample_prior(1)[0], noisy=False))


# ===========================================================================
# Demo / first-network experiment
# ===========================================================================
def weighted_quantile(values, quantiles, weights):
    """Weighted quantiles of a 1-D sample.

    Uses the midpoint convention, ``c = cumsum(w) - w/2`` normalized, which is
    what keeps the result correct when the weights are very uneven -- exactly
    the importance-sampling case, where naive interpolation on the raw
    cumulative sum drags every quantile toward the middle of the support.
    """
    order = np.argsort(values)
    v, w = np.asarray(values, float)[order], np.asarray(weights, float)[order]
    total = w.sum()
    if not np.isfinite(total) or total <= 0:
        return np.full(np.shape(np.atleast_1d(quantiles)), np.nan)
    c = (np.cumsum(w) - 0.5 * w) / total
    return np.interp(np.asarray(quantiles, float), c, v)


def importance_reweight(prob, posterior, x_obs, samples):
    """Reweight NPE samples by likelihood x prior / network density.

    Returns ``(weights, efficiency)``. The efficiency is ESS/N: it is the
    diagnostic that flags a wrong model, and the project's own failure
    criterion puts a floor of 1% on it for real data.
    """
    import torch
    log_q = posterior.log_prob(torch.as_tensor(samples, dtype=torch.float32),
                              x=torch.as_tensor(x_obs, dtype=torch.float32),
                              norm_posterior=False).detach().numpy()
    log_l = prob.log_likelihood(x_obs, samples)
    log_w = log_l - log_q                      # uniform prior: constant, drops
    log_w = np.where(prob.in_prior(samples), log_w, -np.inf)
    log_w -= np.nanmax(log_w[np.isfinite(log_w)])
    w = np.exp(log_w)
    w[~np.isfinite(w)] = 0.0
    eff = (w.sum() ** 2) / (len(w) * np.sum(w ** 2)) if np.any(w) else 0.0
    return w, eff


def _report(prob, theta_true, s, w=None, label=""):
    print(f"\n{label}")
    print("-" * len(label))
    for i, name in enumerate(prob.param_names):
        if w is None:
            lo, hi = np.percentile(s[:, i], [2.5, 97.5])
            mean = s[:, i].mean()
        else:
            lo, hi = weighted_quantile(s[:, i], [0.025, 0.975], w)
            mean = np.average(s[:, i], weights=w)
        print(f"  {name:12s} true {theta_true[i]:9.4f} | mean {mean:9.4f} | "
              f"95% [{lo:.4f}, {hi:.4f}]  width {(hi - lo):.4g}")
    if w is None:
        lo, hi = np.percentile(s[:, 0], [2.5, 97.5])
    else:
        lo, hi = weighted_quantile(s[:, 0], [0.025, 0.975], w)
    return (hi - lo) * 1e3


def train_npe(prob, n_sims=50000, seed=0, density_estimator="nsf",
              hidden_features=None, num_transforms=None, verbose=True,
              **train_kw):
    """Train a single-round amortized NPE and return (posterior, theta, x).

    Single round on purpose: sequential methods tune the proposal to one
    observation and lose amortization, which is the property that makes each
    later spectrum cost one forward pass.
    """
    import torch
    from sbi.inference import NPE
    from sbi.neural_nets import posterior_nn

    torch.manual_seed(seed)
    theta = prob.sample_prior(n_sims)
    x = prob.simulate(theta)

    if hidden_features or num_transforms:
        kw = {}
        if hidden_features:
            kw["hidden_features"] = hidden_features
        if num_transforms:
            kw["num_transforms"] = num_transforms
        estimator = posterior_nn(model=density_estimator, **kw)
    else:
        estimator = density_estimator

    inference = NPE(prior=prob.prior(), density_estimator=estimator)
    inference.append_simulations(torch.as_tensor(theta, dtype=torch.float32),
                                 torch.as_tensor(x, dtype=torch.float32))
    inference.train(show_train_summary=verbose, **train_kw)
    return inference.build_posterior(), theta, x


def evaluate(prob, posterior, theta_true, n_post=20000, seed=0, label=""):
    """Sample the posterior, reweight, and return a metrics dict."""
    import torch
    rng = np.random.default_rng(seed)
    saved, prob.rng = prob.rng, rng
    x_obs = prob.simulate_one(theta_true)
    prob.rng = saved

    s = posterior.sample((n_post,),
                         x=torch.as_tensor(x_obs, dtype=torch.float32),
                         show_progress_bars=False).numpy()
    w, eff = importance_reweight(prob, posterior, x_obs, s)

    lo, hi = np.percentile(s[:, 0], [2.5, 97.5])
    raw = (hi - lo) * 1e3
    rlo, rhi = weighted_quantile(s[:, 0], [0.025, 0.975], w)
    rew = (rhi - rlo) * 1e3
    prior_mHz = (prob.high[0] - prob.low[0]) * 1e3
    floor = 2 * 1.96 * prob.sigma_f / np.sqrt(3) * 1e3
    return dict(label=label, raw_mHz=raw, reweighted_mHz=rew,
                efficiency=eff, floor_mHz=floor, prior_mHz=prior_mHz,
                raw_over_floor=raw / floor, samples=s, weights=w, x_obs=x_obs)


def shrinkage(prob, samples, weights=None, q=(0.025, 0.975)):
    """How far each posterior moved from its prior, per parameter.

    Returns a list of dicts with the prior and posterior widths and

        shrinkage = 1 - posterior width / prior width

    which is ~0 for a direction the data does not constrain and ~1 for one it
    pins down. Reporting this is the point of parameterizing by
    symmetry-distinct couplings: a flat direction becomes visible instead of
    being quietly reported back as the prior while looking like a result.
    """
    span = prob.prior_span()
    out = []
    for i, name in enumerate(prob.param_names):
        if weights is None:
            lo, hi = np.percentile(samples[:, i], [100 * q[0], 100 * q[1]])
        else:
            lo, hi = weighted_quantile(samples[:, i], q, weights)
        width = hi - lo
        prior_width = span[i] * (q[1] - q[0])   # the prior is uniform on the box
        out.append(dict(param=name, prior_width=prior_width,
                        post_width=width,
                        shrinkage=1.0 - width / prior_width,
                        constrained=(1.0 - width / prior_width) > 0.5))
    return out


def efficiency_spread(prob, posterior, n_obs=40, n_post=4000, seed=0):
    """Efficiency and reweighted width over many observations drawn from the prior.

    The project's failure criterion is stated on a single number ("the sample
    efficiency on real data falls below 1%"), but efficiency is a property of
    the particular observation, not just of the network. This measures how much
    it actually moves, which is what says whether a single-observation threshold
    is meaningful.
    """
    import torch
    rng = np.random.default_rng(seed)
    saved, prob.rng = prob.rng, rng
    effs, widths, truths = [], [], []
    for _ in range(n_obs):
        theta = prob.sample_prior(1)[0]
        x_obs = prob.simulate_one(theta)
        s = posterior.sample((n_post,),
                             x=torch.as_tensor(x_obs, dtype=torch.float32),
                             show_progress_bars=False).numpy()
        w, eff = importance_reweight(prob, posterior, x_obs, s)
        lo, hi = weighted_quantile(s[:, 0], [0.025, 0.975], w)
        effs.append(eff)
        widths.append((hi - lo) * 1e3)
        truths.append(theta)
    prob.rng = saved
    return np.array(effs), np.array(widths), np.array(truths)


def main():  # pragma: no cover - demo
    np.set_printoptions(suppress=True)
    n_sims, seed = 50000, 0
    prob = InferenceProblem(seed=seed)
    posterior, _, _ = train_npe(prob, n_sims=n_sims, seed=seed)
    theta_true = np.array([prob.J_center + 0.7, 1.0, 55.0, 12.0])
    m = evaluate(prob, posterior, theta_true, seed=seed)

    print("\n" + "=" * 74)
    print(f"[13C]-formic acid, single-round NPE, {n_sims} simulations")
    print("=" * 74)
    _report(prob, theta_true, m["samples"], None, "Raw NPE posterior")
    _report(prob, theta_true, m["samples"], m["weights"],
            f"After importance reweighting (efficiency {m['efficiency']:.1%})")
    print(f"\n  prior width on J        : {m['prior_mHz']:9.1f} mHz")
    print(f"  raw NPE 95% width       : {m['raw_mHz']:9.2f} mHz")
    print(f"  reweighted 95% width    : {m['reweighted_mHz']:9.2f} mHz")
    print(f"  information floor       : {m['floor_mHz']:9.2f} mHz")
    print(f"  sample efficiency       : {m['efficiency']:9.1%}  (criterion: > 1%)")
    print(f"\n  criterion is on the REWEIGHTED posterior, < 10 mHz -> "
          f"{'PASS' if m['reweighted_mHz'] < 10 else 'FAIL'}")


if __name__ == "__main__":
    main()
