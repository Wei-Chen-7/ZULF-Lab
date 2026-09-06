#!/usr/bin/env python3
"""Simulation-based calibration: is the network honest about its own width?

The worry with a trained network is not that it will be wrong. It is that it
will be wrong and sound sure. Overconfident posteriors are the known failure
mode of these methods, and a width alone cannot reveal one.

SBC is the standard test. Draw theta from the prior, simulate, ask the network,
and record where the true value sits among the posterior samples. Repeat a few
hundred times. If the network is honest the rank is uniform:

    ranks piling up in the MIDDLE  -> the posterior is too WIDE (conservative)
    ranks piling up at the EDGES   -> the posterior is too NARROW (overconfident)

The second is the failure that matters, because it is the one that produces
confident wrong answers.

Note what SBC can and cannot do: it tests the algorithm against its own
simulator, so it cannot catch a wrong lineshape or a missing instrument term.
Those need the nested-sampling reference and the importance-sampling efficiency
on real data.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

import zulf_infer as zi


def run_sbc(prob, posterior, n_trials=300, n_post=99, seed=0, verbose=True):
    """Return an (n_trials, n_params) array of ranks in {0, ..., n_post}."""
    import torch
    rng = np.random.default_rng(seed)
    saved, prob.rng = prob.rng, rng
    ranks = np.empty((n_trials, len(prob.low)), dtype=int)
    for i in range(n_trials):
        theta = prob.sample_prior(1)[0]
        x = prob.simulate_one(theta)
        draws = posterior.sample(
            (n_post,), x=torch.as_tensor(x, dtype=torch.float32),
            show_progress_bars=False).numpy()
        ranks[i] = (draws < theta[None, :]).sum(axis=0)
        if verbose and (i + 1) % 50 == 0:
            print(f"    {i + 1}/{n_trials}", flush=True)
    prob.rng = saved
    return ranks


def diagnose(ranks, n_post, n_bins=20, names=None):
    """Uniformity test plus a plain-language reading of the failure mode."""
    out = []
    n_trials = len(ranks)
    names = list(names or zi.PARAM_NAMES)
    for d in range(ranks.shape[1]):
        r = ranks[:, d] / n_post                      # to [0, 1]
        # chi-square against uniform
        counts, _ = np.histogram(r, bins=n_bins, range=(0, 1))
        expect = n_trials / n_bins
        chi2 = ((counts - expect) ** 2 / expect).sum()
        p_chi2 = 1 - stats.chi2.cdf(chi2, n_bins - 1)
        p_ks = stats.kstest(r, "uniform").pvalue

        # Where is the mass? Outer 20% and central 20% of the range, each
        # expected at 0.20. Compare in sigma, not against a fixed cut: a
        # *depletion* of the edges is as strong a signal of a too-wide
        # posterior as a pile-up in the centre, and is often the more
        # sensitive of the two, so testing only "centre > some cut" misses it.
        outer = float(np.mean((r < 0.1) | (r > 0.9)))
        centre = float(np.mean((r > 0.4) & (r < 0.6)))
        sd = np.sqrt(0.2 * 0.8 / n_trials)
        z_outer = (outer - 0.2) / sd
        z_centre = (centre - 0.2) / sd
        # a systematic shift shows up as the mean rank leaving 0.5
        z_shift = (r.mean() - 0.5) / (np.sqrt(1 / 12.0) / np.sqrt(n_trials))

        if p_chi2 > 0.01 and p_ks > 0.01:
            verdict = "uniform -> calibrated"
        elif z_outer > 2:
            verdict = "edges loaded -> OVERCONFIDENT (posterior too narrow)"
        elif z_outer < -2 or z_centre > 2:
            verdict = "edges depleted -> conservative (posterior too wide)"
        elif abs(z_shift) > 3:
            verdict = f"shifted -> biased (mean rank {r.mean():.3f}, not 0.5)"
        else:
            verdict = "non-uniform -> shape differs, no clear direction"
        out.append(dict(param=names[d], chi2=chi2, p_chi2=p_chi2,
                        p_ks=p_ks, outer=outer, centre=centre,
                        z_outer=z_outer, z_centre=z_centre, z_shift=z_shift,
                        verdict=verdict))
    return out


def plot_ranks(ranks, n_post, path="sbc_ranks.png", n_bins=20, title="",
               names=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n_trials, n_par = ranks.shape
    names = list(names or zi.PARAM_NAMES)
    fig, axes = plt.subplots(1, n_par, figsize=(3.1 * n_par, 3.0), sharey=True)
    expect = n_trials / n_bins
    band = 1.96 * np.sqrt(expect * (1 - 1 / n_bins))
    for d, ax in enumerate(np.atleast_1d(axes)):
        ax.hist(ranks[:, d] / n_post, bins=n_bins, range=(0, 1),
                color="#2a9d8f", edgecolor="white")
        ax.axhline(expect, color="#22333b", lw=1)
        ax.axhspan(expect - band, expect + band, color="#e76f51", alpha=0.15)
        ax.set_title(names[d], fontsize=10)
        ax.set_xlabel("rank / L")
    np.atleast_1d(axes)[0].set_ylabel("count")
    fig.suptitle(title or "Simulation-based calibration: ranks should be uniform",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor="white", bbox_inches="tight")
    return path


def main():  # pragma: no cover - study
    seed, n_sims = 0, 50_000
    prob = zi.InferenceProblem(seed=seed)
    print(f"training NPE on {n_sims} simulations ...", flush=True)
    posterior, _, _ = zi.train_npe(prob, n_sims=n_sims, seed=seed, verbose=False)

    n_trials, n_post = 300, 99
    print(f"running SBC: {n_trials} trials x {n_post} posterior draws", flush=True)
    ranks = run_sbc(prob, posterior, n_trials=n_trials, n_post=n_post, seed=seed)

    print("\n" + "=" * 86)
    print(f"SBC, {n_trials} trials, L = {n_post}")
    print("=" * 86)
    print(f"{'param':>12} {'chi2 p':>9} {'KS p':>9} {'outer20%':>10} "
          f"{'centre20%':>10}  verdict")
    for row in diagnose(ranks, n_post, names=prob.param_names):
        print(f"{row['param']:>12} {row['p_chi2']:>9.3f} {row['p_ks']:>9.3f} "
              f"{row['outer']:>10.3f} {row['centre']:>10.3f}  {row['verdict']}")
    print("\n  (both fractions are 0.20 when calibrated)")

    path = plot_ranks(ranks, n_post, names=prob.param_names)
    np.save("sbc_ranks.npy", ranks)
    print(f"\n  wrote {path} and sbc_ranks.npy")


if __name__ == "__main__":
    main()
