#!/usr/bin/env python3
"""Nested sampling on the exact likelihood: the reference the network is judged against.

Because the ZULF likelihood is available in closed form, the gold standard is
not the local fit -- it is nested sampling or MCMC on the exact likelihood. This
runs that reference on the two-spin case, where it is cheap, so the network's
reweighted posterior can be checked against something independent rather than
against itself.

Two things come out of it:

* a reference posterior on J, computed without any network, and
* the evidence log Z, which nested sampling gives for free.

The observation is constructed exactly as in ``zulf_infer.evaluate`` so the
comparison is like-for-like on the same data.
"""

from __future__ import annotations

import time

import numpy as np

import zulf_infer as zi


def make_observation(prob, theta_true, seed=0):
    """Reproduce the same x_obs that ``zulf_infer.evaluate`` builds."""
    saved, prob.rng = prob.rng, np.random.default_rng(seed)
    x_obs = prob.simulate_one(theta_true)
    prob.rng = saved
    return x_obs


def run_nested(prob, x_obs, nlive=600, seed=0, dlogz=0.05):
    """Nested sampling over the box prior with the exact Gaussian likelihood."""
    import dynesty

    lo, hi = prob.low, prob.high
    span = hi - lo
    ndim = len(lo)

    def prior_transform(u):
        return lo + u * span

    def loglike(theta):
        return float(prob.log_likelihood(x_obs, theta)[0])

    sampler = dynesty.NestedSampler(loglike, prior_transform, ndim,
                                    nlive=nlive, rstate=np.random.default_rng(seed))
    t0 = time.perf_counter()
    sampler.run_nested(dlogz=dlogz, print_progress=False)
    elapsed = time.perf_counter() - t0

    res = sampler.results
    logw = res.logwt - res.logz[-1]
    w = np.exp(logw - logw.max())
    w /= w.sum()
    return res.samples, w, float(res.logz[-1]), float(res.logzerr[-1]), elapsed


def main():  # pragma: no cover - study
    np.set_printoptions(suppress=True)
    seed = 0
    prob = zi.InferenceProblem(seed=seed)
    theta_true = np.array([prob.J_center + 0.7, 1.0, 55.0, 12.0])
    x_obs = make_observation(prob, theta_true, seed=seed)

    print("=" * 78)
    print("Nested sampling on the exact likelihood, [13C]-formic acid")
    print("=" * 78)
    samples, w, logz, logzerr, elapsed = run_nested(prob, x_obs, seed=seed)
    print(f"  {len(samples)} samples in {elapsed:.1f} s")
    print(f"  log Z = {logz:.3f} +/- {logzerr:.3f}\n")

    print(f"{'parameter':>12} {'true':>10} {'mean':>11} {'95% interval':>26} {'width':>10}")
    for i, name in enumerate(zi.PARAM_NAMES):
        lo, hi = zi.weighted_quantile(samples[:, i], [0.025, 0.975], w)
        mean = np.average(samples[:, i], weights=w)
        print(f"{name:>12} {theta_true[i]:>10.4f} {mean:>11.4f} "
              f"  [{lo:9.4f},{hi:9.4f}] {hi - lo:>10.4g}")

    jlo, jhi = zi.weighted_quantile(samples[:, 0], [0.025, 0.975], w)
    width = (jhi - jlo) * 1e3
    floor = 2 * 1.96 * prob.sigma_f / np.sqrt(3) * 1e3
    print(f"\n  nested sampling 95% width on J : {width:8.2f} mHz")
    print(f"  information floor              : {floor:8.2f} mHz")
    print(f"  ratio                          : {width / floor:8.2f}")
    print(f"\n  This is the reference. A reweighted NPE posterior that agrees")
    print(f"  with it is exact; one that does not is not, whatever its width.")

    np.savez("nested_reference.npz", samples=samples, weights=w,
             theta_true=theta_true, x_obs=x_obs, logz=logz, logzerr=logzerr)
    print("\n  wrote nested_reference.npz")


if __name__ == "__main__":
    main()
