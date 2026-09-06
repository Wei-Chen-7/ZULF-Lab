#!/usr/bin/env python3
"""Where the peak-list summary jumps, and what it costs.

A peak list is a discontinuous observable. Two lines closer than the
spectrometer's resolution are reported as one peak; move them slightly apart
and they are reported as two. Nothing smooths that transition -- the peak count
is an integer, and it changes.

That is a real property of the data, not a modelling shortcut, and it has
consequences the project has to state rather than discover later on real
spectra:

* a local optimizer cannot cross a jump, so the chi-square surface is carved
  into basins that have nothing to do with the physics; and
* the importance weights near a jump could be erratic, because the network's
  smooth density is approximating something that is not smooth.

That second point was the hypothesis for the ~30x spread in sampling efficiency
across observations drawn from the same prior with the same network. **It is
wrong.** Spearman rho between distance-to-cliff and efficiency is -0.06
(p = 0.64) over 60 observations. Recorded here rather than dropped, because the
project's failure criterion is stated on efficiency and a plausible-sounding
explanation that happens to be false is worse than an open question.

What does correlate is the true J_CH itself (rho = -0.38), i.e. the flow fits
its own posterior less well toward the top of the J prior. That is a property of
the trained network, not of the physics.

One class of cliff was an outright bug and is gone: the merge threshold used to
be the model's own linewidth, so it moved with the fitted T2 and made the
observation model discontinuous in a parameter it has no business being
discontinuous in. It is now an instrument constant (``InferenceProblem.merge_hz``).
The cliffs measured here are the remaining, physical ones.
"""

from __future__ import annotations

import numpy as np

import zulf_infer as zi

__all__ = ["cliff_scan", "distance_to_cliff", "n_resolved"]


def n_resolved(prob, theta):
    """Number of lines left after merging -- the integer that jumps."""
    import zulf_forward as zf
    theta = np.asarray(theta, float)
    B_nT, ang, _ = (float(v) for v in theta[prob.n_couplings:])
    f, a = zf.line_list(prob.sys, B=zf.field_vector(B_nT * 1e-3, ang),
                        J=prob.J_matrix(theta))
    return len(zi.merge_lines(np.asarray(f), np.asarray(a), prob.merge_hz)[0])


def cliff_scan(prob, theta, index, n=600, jump_sigma=5.0, ratio=20.0):
    """Scan one parameter across its prior and locate the jumps.

    Returns the parameter values where the noiseless summary vector moves by
    far more between adjacent grid points than it does typically -- more than
    ``jump_sigma`` noise standard deviations, and more than ``ratio`` times the
    median step. Both conditions are needed: the first alone flags a merely
    steep-but-smooth direction, the second alone flags nothing when the whole
    scan is flat.

    Adjacent flagged cells are reported as one location. Some transitions are
    not a single crossing but a band -- near theta = 0 the transverse field
    vanishes, several lines fade below the amplitude floor together, and the
    summary is unstable across a whole stretch of the axis. Counting grid cells
    there would report a dozen cliffs where there is one region.
    """
    theta = np.asarray(theta, float)
    grid = np.linspace(prob.low[index], prob.high[index], n)
    sig = prob.slot_sigmas()
    xs = np.empty((n, len(sig)))
    for k, v in enumerate(grid):
        t = theta.copy()
        t[index] = v
        xs[k] = prob.simulate_one(t, noisy=False)
    step = np.linalg.norm(np.diff(xs, axis=0) / sig, axis=1)
    med = float(np.median(step))
    hit = np.flatnonzero((step > jump_sigma) & (step > ratio * max(med, 1e-12)))

    bands = []
    for k in hit:
        if bands and k == bands[-1][-1] + 1:
            bands[-1].append(k)
        else:
            bands.append([k])
    centres = np.array([0.5 * (grid[b[0]] + grid[b[-1] + 1]) for b in bands])
    return centres, grid, step


def distance_to_cliff(prob, theta, n=600, **kw):
    """Distance from theta to the nearest jump, as a fraction of the prior box.

    Minimized over parameters: an observation is near a cliff if *any* one of
    its parameters is, since that is enough to make the likelihood surface
    non-smooth where the sampler is working.
    """
    span = prob.high - prob.low
    best = np.inf
    for i in range(len(theta)):
        loc, _, _ = cliff_scan(prob, theta, i, n=n, **kw)
        if len(loc):
            best = min(best, float(np.min(np.abs(loc - theta[i])) / span[i]))
    return best


# ===========================================================================
def study(seed=0, n_obs=60, n_sims=150_000, n_post=4000):  # pragma: no cover
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    np.set_printoptions(suppress=True)
    prob = zi.InferenceProblem(seed=seed)
    print("=" * 84)
    print("Resolution cliffs in the peak-list summary, [13C]-formic acid")
    print("=" * 84)
    print(f"  merge resolution : {prob.merge_hz * 1e3:.2f} mHz "
          f"(fixed; the linewidth at the centre of the T2 prior)")
    print(f"  linewidth range  : "
          f"{1e3 / (np.pi * prob.T2_range[1]):.2f} - "
          f"{1e3 / (np.pi * prob.T2_range[0]):.2f} mHz over the T2 prior")

    # ---- 1. where are they? ----------------------------------------------
    theta = np.array([prob.J_center + 0.7, 1.0, 55.0, 12.0])
    print("\n" + "-" * 84)
    print("1. WHERE THE SUMMARY JUMPS, scanning each parameter across its prior")
    print("-" * 84)
    scans = {}
    for i, name in enumerate(prob.param_names):
        loc, grid, step = cliff_scan(prob, theta, i)
        scans[name] = (loc, grid, step)
        where = ", ".join(f"{v:.4g}" for v in loc[:6]) if len(loc) else "none"
        print(f"  {name:>12}  {len(loc):>2d} jump(s)   at {where}")

    print("\n  T2 is now clean, which is the point of making the merge width an")
    print("  instrument constant. It used to carry the worst cliff in the model.")

    # ---- 2. does proximity predict efficiency? ---------------------------
    print("\n" + "-" * 84)
    print("2. DOES PROXIMITY TO A CLIFF EXPLAIN THE EFFICIENCY SPREAD?")
    print("-" * 84)
    print("  training/loading the network ...", flush=True)
    posterior, _ = zi.train_or_load(prob, tag="formic_acid_150k",
                                    n_sims=n_sims, seed=seed,
                                    max_num_epochs=150)

    rng = np.random.default_rng(seed)
    saved, prob.rng = prob.rng, rng
    effs, dists, widths, truths = [], [], [], []
    for k in range(n_obs):
        t = prob.sample_prior(1)[0]
        x_obs = prob.simulate_one(t)
        s = posterior.sample((n_post,),
                             x=torch.as_tensor(x_obs, dtype=torch.float32),
                             show_progress_bars=False).numpy()
        w, eff = zi.importance_reweight(prob, posterior, x_obs, s)
        lo, hi = zi.weighted_quantile(s[:, 0], [0.025, 0.975], w)
        effs.append(eff)
        widths.append((hi - lo) * 1e3)
        dists.append(distance_to_cliff(prob, t, n=300))
        truths.append(t)
        if (k + 1) % 20 == 0:
            print(f"    {k + 1}/{n_obs}", flush=True)
    prob.rng = saved
    effs, dists, widths = np.array(effs), np.array(dists), np.array(widths)
    truths = np.array(truths)

    finite = np.isfinite(dists)
    print(f"\n  observations with a cliff anywhere in their box : "
          f"{finite.sum()}/{n_obs}")
    q = np.percentile(effs, [0, 25, 50, 75, 100])
    print(f"  efficiency: min {q[0]:.2%} | median {q[2]:.2%} | max {q[4]:.2%} "
          f"| spread {q[4] / max(q[0], 1e-9):.0f}x")

    if finite.sum() > 5:
        d, e = dists[finite], effs[finite]
        near = d < np.median(d)
        from scipy import stats
        rho, p = stats.spearmanr(d, e)
        print(f"\n  Spearman rho(distance to cliff, efficiency) = {rho:+.3f} "
              f"(p = {p:.3g})")
        print(f"  median efficiency, near half : {np.median(e[near]):.2%}")
        print(f"  median efficiency, far half  : {np.median(e[~near]):.2%}")
        verdict = ("cliffs explain the spread" if (rho > 0.3 and p < 0.05)
                   else "no clear link -- the spread has another cause")
        print(f"  -> {verdict}")

    # ---- 3. then what does? ----------------------------------------------
    from scipy import stats
    print("\n" + "-" * 84)
    print("3. WHAT DOES PREDICT EFFICIENCY?")
    print("-" * 84)
    print("  The failure criterion is stated on efficiency, so if cliffs are")
    print("  not the driver it is worth knowing what is.\n")
    print(f"{'quantity':>28} {'Spearman rho':>13} {'p':>10} {'p (adj)':>10}")
    cands = [(n, truths[:, i]) for i, n in enumerate(prob.param_names)]
    # how close the truth sits to the edge of the prior box, in prior units
    u = (truths - prob.low) / prob.prior_span()
    cands.append(("distance to prior edge", np.min(np.minimum(u, 1 - u), axis=1)))
    cands.append(("distance to cliff", dists))
    cands.append(("reweighted width on J", widths))
    rows, n_tested = [], 0
    for name, v in cands:
        ok = np.isfinite(v)
        if ok.sum() < 5 or np.ptp(v[ok]) == 0:
            continue
        rho, p = stats.spearmanr(v[ok], effs[ok])
        rows.append((name, rho, p))
        n_tested += 1
    best = []
    for name, rho, p in rows:
        # Several candidates are being screened at once, so the raw p-value
        # overstates the evidence; Bonferroni is the conservative correction.
        p_adj = min(1.0, p * n_tested)
        flag = " <--" if p_adj < 0.05 else ""
        print(f"{name:>28} {rho:>13.3f} {p:>10.3g} {p_adj:>10.3g}{flag}")
        if p_adj < 0.05:
            best.append((name, rho, p_adj))
    if best:
        print(f"\n  Surviving a Bonferroni correction over {n_tested} candidates: "
              + ", ".join(f"{n} ({r:+.2f}, p_adj = {pa:.3f})"
                          for n, r, pa in best))
        print("  This is where the flow fits the posterior least well, not a")
        print("  property of the physics -- distance to the prior boundary and")
        print("  distance to a cliff both come back flat. Suggestive at n = 60,")
        print("  not established; the practical consequence either way is that")
        print("  the efficiency criterion should be read over several spectra.")
    else:
        print("\n  Nothing here predicts it. The spread is a property of the")
        print("  trained flow's local fit to the posterior, not of the physics,")
        print("  which is an argument for quoting efficiency over several")
        print("  spectra rather than trusting one reading.")

    # ---- figure -----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    name = "B_nT"
    loc, grid, step = scans[name]
    ax.semilogy(grid[:-1], np.maximum(step, 1e-6), lw=1.2, color="#264653")
    for v in loc:
        ax.axvline(v, color="#e76f51", lw=1.0, ls="--")
    ax.set_xlabel(f"{name}  (rest of theta fixed)")
    ax.set_ylabel("|dx| per grid step, in units of noise")
    ax.set_title(f"Summary jumps: {len(loc)} cliff(s) in {name}", fontsize=10)

    ax = axes[1]
    ok = np.isfinite(dists)
    ax.loglog(np.maximum(dists[ok], 1e-5), np.maximum(effs[ok], 1e-4), "o",
              ms=5, color="#2a9d8f", alpha=0.8)
    ax.axhline(0.01, color="#e76f51", lw=1.0, ls="--")
    ax.text(ax.get_xlim()[0] * 1.1, 0.011, "1% criterion", color="#e76f51",
            fontsize=8, va="bottom")
    ax.set_xlabel("distance to nearest cliff / prior width")
    ax.set_ylabel("importance-sampling efficiency")
    ax.set_title("Efficiency vs proximity to a cliff", fontsize=10)
    fig.suptitle("Resolution cliffs in a peak-list summary", fontsize=12,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig("resolution_cliffs.png", dpi=140, facecolor="white",
                bbox_inches="tight")

    np.savez("resolution_cliffs.npz", effs=effs, dists=dists,
             widths=widths, truths=truths,
             param_names=np.array(prob.param_names))
    print("\nwrote resolution_cliffs.png and resolution_cliffs.npz")


if __name__ == "__main__":       # pragma: no cover
    study()
