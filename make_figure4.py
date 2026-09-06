#!/usr/bin/env python3
"""Figure 4: against exact-likelihood sampling, and against ref [1]'s method.

The fourth of the five figures. Three panels, because the comparison has three
different answers depending on the problem, and showing only the flattering one
would be dishonest:

(a) **They agree.** On [13C]-formic acid the local fit's curvature error bar,
    the reweighted network posterior and nested sampling on the exact
    likelihood all land on the same interval. This is the panel that validates
    all three methods at once, and it is also the panel that says the network
    wins nothing here on accuracy.

(b) **The local fit returns an arbitrary number.** On [13C]-methanol, J_HH
    inside the equivalent methyl group moves no line, so every value fits
    equally well and the simplex drifts along the flat direction until the
    prior box stops it. Plotting fitted against starting value gives a flat
    line for J_CH and, for J_HH, a cloud filling the prior -- spread wider than
    the prior itself, correlated with the start at only +0.20. It is not the
    guess handed back; it is noise, reported with an error bar.

(c) **The local fit cannot see the second mode.** theta_B and 180 - theta_B give
    bit-identical spectra. The network holds both; the local fit returns
    whichever it started nearest, with a tight interval that excludes the other.

    python make_figure4.py        # writes figure4_comparison.png
"""

from __future__ import annotations

import os

import numpy as np

import zulf_infer as zi
import local_baseline as lb

OUT = "figure4_comparison.png"
SEED = 0

C_NESTED, C_NPE, C_LOCAL = "#264653", "#2a9d8f", "#e76f51"


def _density(v, w, grid, bw=None):
    """Weighted Gaussian KDE, evaluated on a grid.

    Silverman's rule on the effective sample size, with the bandwidth floored
    at half a grid spacing. The floor matters: when importance sampling is
    inefficient a single weight can dominate, the weighted spread collapses,
    and an unfloored kernel underflows to zero between grid points -- rendering
    an empty panel rather than a spike. Nothing finer than the grid can be
    displayed anyway.
    """
    v, w = np.asarray(v, float), np.asarray(w, float)
    w = w / w.sum()
    mu = w @ v
    sd = np.sqrt(max(w @ (v - mu) ** 2, 0.0))
    if bw is None:
        ess = 1.0 / np.sum(w ** 2)
        bw = 1.06 * sd * ess ** -0.2
    bw = max(bw, 0.5 * float(np.min(np.diff(grid))))
    z = (grid[:, None] - v[None, :]) / bw
    return (np.exp(-0.5 * z ** 2) @ w) / (bw * np.sqrt(2 * np.pi))


def panel_a(ax, seed=SEED):
    """Three methods, one observation, on [13C]-formic acid."""
    prob = zi.InferenceProblem(seed=seed)
    ref = np.load("nested_reference.npz")
    theta_true, x_obs = ref["theta_true"], ref["x_obs"]

    posterior, _ = zi.train_or_load(prob, tag="formic_acid_150k",
                                    n_sims=150_000, seed=seed,
                                    max_num_epochs=150)
    m = zi.evaluate(prob, posterior, theta_true, seed=seed, n_post=20000)
    fit = lb.local_fit(prob, x_obs, 0.5 * (prob.low + prob.high))
    sigma, _, _, _ = lb.curvature_errors(prob, x_obs, fit["theta"])

    j0 = theta_true[0]
    grid = np.linspace(-4.0, 4.0, 800)                    # mHz from the truth
    hz = j0 + grid * 1e-3

    dn = _density(ref["samples"][:, 0], ref["weights"], hz)
    dp = _density(m["samples"][:, 0], m["weights"], hz)
    dl = np.exp(-0.5 * ((hz - fit["theta"][0]) / sigma[0]) ** 2) \
        / (sigma[0] * np.sqrt(2 * np.pi))

    ax.fill_between(grid, dn / dn.max(), color=C_NESTED, alpha=0.18)
    ax.plot(grid, dn / dn.max(), color=C_NESTED, lw=2.0,
            label="nested sampling (exact)")
    ax.plot(grid, dp / dp.max(), color=C_NPE, lw=2.0, ls="-",
            label="NPE, reweighted")
    ax.plot(grid, dl / dl.max(), color=C_LOCAL, lw=1.8, ls="--",
            label="local fit, curvature")
    ax.axvline(0.0, color="#22333b", lw=1.0, ls=":")

    def w95(v, w):
        lo, hi = zi.weighted_quantile(v, [0.025, 0.975], w)
        return (hi - lo) * 1e3
    wn = w95(ref["samples"][:, 0], ref["weights"])
    wp = w95(m["samples"][:, 0], m["weights"])
    wl = 2 * 1.96 * sigma[0] * 1e3
    floor = 2 * 1.96 * prob.sigma_f / np.sqrt(3) * 1e3
    ax.text(0.02, 0.97,
            f"95% width on $J_{{\\rm CH}}$\n"
            f"  nested      {wn:5.2f} mHz\n"
            f"  NPE         {wp:5.2f} mHz\n"
            f"  local       {wl:5.2f} mHz\n"
            f"  floor       {floor:5.2f} mHz",
            transform=ax.transAxes, va="top", ha="left", fontsize=7.6,
            family="monospace",
            bbox=dict(fc="white", ec="#adb5bd", alpha=0.9, pad=4))
    ax.set_xlabel(r"$J_{\rm CH}$ $-$ truth  [mHz]", fontsize=9)
    ax.set_ylabel("posterior density (scaled)", fontsize=9)
    ax.set_title("(a) On a clean problem all three agree", fontsize=9.5,
                 fontweight="bold")
    ax.legend(fontsize=7.5, loc="center right", framealpha=0.9)
    ax.tick_params(labelsize=8)
    return dict(nested=wn, npe=wp, local=wl, floor=floor)


def panel_b(ax, seed=SEED, n_starts=60):
    """Fitted vs starting value on methanol: the flat direction."""
    path = "local_baseline_methanol.npz"
    prob = zi.InferenceProblem(system="methanol", seed=seed)
    names = list(prob.param_names)
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        arr, starts = d["multistart_theta"], d["multistart_start"]
        theta_true = d["theta_true"]
    else:
        theta_true = np.array([141.7, -12.4, 1.0, 55.0, 12.0])
        saved, prob.rng = prob.rng, np.random.default_rng(seed)
        x_obs = prob.simulate_one(theta_true)
        prob.rng = saved
        fits = lb.multistart(prob, x_obs, n_starts=n_starts, seed=seed,
                             verbose=False)
        arr = np.array([f["theta"] for f in fits])
        starts = np.array([f["theta0"] for f in fits])

    ich, ihh = names.index("J_CH"), names.index("J_HH")
    span_hh = prob.prior_span()[ihh]

    # Both couplings on one axis, each centred on its own prior, in units of
    # its own prior width, so the two behaviours are directly comparable.
    def norm(v, i):
        return (v - prob.low[i]) / prob.prior_span()[i]

    r_hh = np.corrcoef(starts[:, ihh], arr[:, ihh])[0, 1]
    r_ch = np.corrcoef(starts[:, ich], arr[:, ich])[0, 1]

    # The truth, not the identity line, is the reference: the question is
    # whether the fit lands on the right answer, not whether it stays put.
    ax.axhline(norm(theta_true[ihh], ihh), color=C_LOCAL, lw=0.9, ls="--")
    ax.axhline(norm(theta_true[ich], ich), color=C_NPE, lw=0.9, ls="--")
    ax.plot(norm(starts[:, ihh], ihh), norm(arr[:, ihh], ihh), "o", ms=5,
            color=C_LOCAL, alpha=0.85,
            label=f"$J_{{\\rm HH}}$  (flat)   r = {r_hh:+.2f}")
    ax.plot(norm(starts[:, ich], ich), norm(arr[:, ich], ich), "s", ms=4.5,
            color=C_NPE, alpha=0.85,
            label=f"$J_{{\\rm CH}}$ (measured) r = {r_ch:+.2f}")
    ax.annotate("true $J_{\\rm HH}$", xy=(0.99, norm(theta_true[ihh], ihh)),
                ha="right", va="bottom", fontsize=7.2, color=C_LOCAL)
    ax.annotate("true $J_{\\rm CH}$", xy=(0.99, norm(theta_true[ich], ich)),
                ha="right", va="bottom", fontsize=7.2, color=C_NPE)

    drift = np.mean(np.abs(arr[:, ihh] - starts[:, ihh]))
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("starting guess  (fraction of prior)", fontsize=9)
    ax.set_ylabel("fitted value  (fraction of prior)", fontsize=9)
    ax.set_title("(b) For a direction the data cannot see,\n"
                 "the fit returns an arbitrary number", fontsize=9.5,
                 fontweight="bold")
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9)
    ax.text(0.03, 0.95,
            f"$[^{{13}}$C]-methanol, {len(arr)} starts\n"
            f"$J_{{\\rm HH}}$ fills its {span_hh:.0f} Hz prior;\n"
            f"mean drift from the start\n"
            f"  {drift:.1f} Hz ({drift / span_hh:.0%} of the prior)\n"
            f"reported as $\\sigma \\sim 10^6$ Hz",
            transform=ax.transAxes, va="top", fontsize=7.6,
            bbox=dict(fc="white", ec="#adb5bd", alpha=0.9, pad=4))
    ax.tick_params(labelsize=8)
    return dict(r_hh=float(r_hh), r_ch=float(r_ch), drift=float(drift))


def panel_c(ax, seed=SEED):
    """A genuinely bimodal posterior: the network holds both modes."""
    prob = zi.InferenceProblem(seed=seed, theta_max_deg=180.0)
    theta_true = np.array([prob.J_center + 0.7, 1.0, 55.0, 12.0])
    saved, prob.rng = prob.rng, np.random.default_rng(seed)
    x_obs = prob.simulate_one(theta_true)
    prob.rng = saved

    posterior, _ = zi.train_or_load(prob, tag="formic_acid_bimodal_50k",
                                    n_sims=50_000, seed=seed,
                                    max_num_epochs=150)
    m = zi.evaluate(prob, posterior, theta_true, seed=seed, n_post=20000)
    s, w = m["samples"], m["weights"]

    grid = np.linspace(0.0, 180.0, 900)
    dn = _density(s[:, 2], w, grid)
    ax.fill_between(grid, dn / dn.max(), color=C_NPE, alpha=0.20)
    ax.plot(grid, dn / dn.max(), color=C_NPE, lw=2.0,
            label="NPE posterior, reweighted")

    for k, start_ang in enumerate((70.0, 110.0)):
        t0 = theta_true.copy()
        t0[2] = start_ang
        f = lb.local_fit(prob, x_obs, t0)
        sg, _, _, _ = lb.curvature_errors(prob, x_obs, f["theta"])
        g = np.exp(-0.5 * ((grid - f["theta"][2]) / sg[2]) ** 2)
        ax.plot(grid, g, color=C_LOCAL, lw=1.8, ls="--",
                label="local fit (started at 70 and 110 deg)" if k == 0 else None)
        ax.annotate(f"{f['theta'][2]:.1f} $\\pm$ {1.96 * sg[2]:.1f}$^\\circ$",
                    xy=(f["theta"][2], 1.02), ha="center", fontsize=7.6,
                    color=C_LOCAL)
    ax.axvline(theta_true[2], color="#22333b", lw=1.0, ls=":")
    ax.annotate("truth", xy=(theta_true[2], 0.5), xytext=(-30, 0),
                textcoords="offset points", fontsize=7.6, color="#22333b")

    mass = w[s[:, 2] < 90].sum() / w.sum()
    ax.text(0.02, 0.62, f"posterior mass\n  below 90$^\\circ$: {mass:.0%}\n"
                        f"  above 90$^\\circ$: {1 - mass:.0%}\n"
                        f"(exactly degenerate,\n so 50/50 is correct)",
            transform=ax.transAxes, va="top", fontsize=7.6,
            bbox=dict(fc="white", ec="#adb5bd", alpha=0.9, pad=4))
    ax.set_ylim(0, 1.12)
    ax.set_xlim(0, 180)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xlabel(r"field angle $\theta_B$  [deg]", fontsize=9)
    ax.set_title(r"(c) $\theta_B \leftrightarrow 180^\circ-\theta_B$:"
                 "\nthe local fit sees one mode of two", fontsize=9.5,
                 fontweight="bold")
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
    ax.tick_params(labelsize=8)
    return dict(mass_below=float(mass))


def main():  # pragma: no cover - figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
    print("panel (a): formic acid, three methods ...", flush=True)
    a = panel_a(axes[0])
    print("panel (b): methanol, the flat direction ...", flush=True)
    b = panel_b(axes[1])
    print("panel (c): the bimodal angle ...", flush=True)
    c = panel_c(axes[2])

    fig.suptitle("Figure 4 — the trained network against exact-likelihood "
                 "sampling and against a least-squares fit",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT, dpi=150, facecolor="white", bbox_inches="tight")

    print(f"\n  (a) nested {a['nested']:.2f} / NPE {a['npe']:.2f} / "
          f"local {a['local']:.2f} mHz, floor {a['floor']:.2f}")
    print(f"  (b) corr(start, fit): J_HH {b['r_hh']:+.3f}, "
          f"J_CH {b['r_ch']:+.3f}; mean J_HH drift {b['drift']:.1f} Hz")
    print(f"  (c) posterior mass below 90 deg: {c['mass_below']:.1%}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":       # pragma: no cover
    main()
