#!/usr/bin/env python3
"""Figure 2: the first posterior, with its spread.

The second of the five figures the proposal commits to. It has one job -- show
what the network actually returns for a coupling, and how much narrower that is
than what went in.

The honest way to draw it is on zoomed axes with the prior width written on
each panel, not on the prior's own axes. On the prior's axes the J posterior is
a vertical line 1/2500th of the frame wide: dramatic, and it hides the
correlations between the nuisance parameters, which are the part of the figure
a referee will actually interrogate.

The inset carries the number that matters: prior 5.7 Hz, raw network proposal,
reweighted posterior 2.2 mHz, and the information floor the noise model allows.
Reaching the floor is the claim; the inset is where it is checked.

    python make_figure2.py        # writes figure2_posterior.png
"""

from __future__ import annotations

import numpy as np

import zulf_infer as zi

OUT = "figure2_posterior.png"
N_SIMS = 150_000
SEED = 0

_UNIT = {"J_CH": "Hz", "B_nT": "nT", "B_theta_deg": "deg", "T2_s": "s"}
_LABEL = {"J_CH": r"$J_{\rm CH}$", "B_nT": r"$|B|$",
          "B_theta_deg": r"$\theta_B$", "T2_s": r"$T_2$"}


def _wq(v, q, w):
    return zi.weighted_quantile(v, q, w)


def _wcov(s, w):
    w = w / w.sum()
    mu = w @ s
    d = s - mu
    return (d * w[:, None]).T @ d


def corner(prob, samples, weights, theta_true, raw=None, path=OUT):
    """Weighted corner plot on zoomed axes, with the prior width annotated."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = prob.param_names
    d = len(names)
    span = prob.prior_span()

    # Zoom each axis to the posterior, not the prior.
    lims = []
    for i in range(d):
        lo, hi = _wq(samples[:, i], [0.001, 0.999], weights)
        pad = 0.25 * (hi - lo) or 1e-6
        lims.append((max(lo - pad, prob.low[i]), min(hi + pad, prob.high[i])))

    fig, axes = plt.subplots(d, d, figsize=(9.6, 9.0))
    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                ax.hist(samples[:, i], bins=60, range=lims[i], weights=weights,
                        color="#2a9d8f", edgecolor="none")
                ax.axvline(theta_true[i], color="#e76f51", lw=1.4)
                lo, hi = _wq(samples[:, i], [0.025, 0.975], weights)
                ax.axvspan(lo, hi, color="#264653", alpha=0.10)
                u = _UNIT.get(names[i], "")
                shrink = 1.0 - (hi - lo) / (0.95 * span[i])
                ax.set_title(f"{_LABEL.get(names[i], names[i])}: "
                             f"{hi - lo:.3g} {u} wide\n"
                             f"prior {0.95 * span[i]:.3g} {u}  "
                             f"(shrinkage {shrink:.3f})", fontsize=8)
                ax.set_yticks([])
            else:
                ax.hist2d(samples[:, j], samples[:, i], bins=45,
                          range=[lims[j], lims[i]], weights=weights,
                          cmap="Greens")
                ax.plot(theta_true[j], theta_true[i], "o", ms=5,
                        mfc="none", mec="#e76f51", mew=1.6)
                r = _wcov(samples, weights)
                rho = r[i, j] / np.sqrt(r[i, i] * r[j, j])
                ax.text(0.04, 0.90, f"r = {rho:+.2f}", transform=ax.transAxes,
                        fontsize=7.5, color="#22333b")
            ax.set_xlim(*lims[j])
            if i != j:
                ax.set_ylim(*lims[i])
            ax.tick_params(labelsize=7)
            if i == d - 1:
                ax.set_xlabel(f"{_LABEL.get(names[j], names[j])} "
                              f"[{_UNIT.get(names[j], '')}]", fontsize=8.5)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(f"{_LABEL.get(names[i], names[i])} "
                              f"[{_UNIT.get(names[i], '')}]", fontsize=8.5)
            else:
                ax.set_yticklabels([])

    # ---- inset: how far J moved, prior -> proposal -> posterior ----------
    ax = fig.add_axes([0.60, 0.62, 0.345, 0.27])
    prior_w = 0.95 * span[0] * 1e3
    lo, hi = _wq(samples[:, 0], [0.025, 0.975], weights)
    post_w = (hi - lo) * 1e3
    floor = 2 * 1.96 * prob.sigma_f / np.sqrt(3) * 1e3
    bars = [("prior", prior_w, "#adb5bd")]
    if raw is not None:
        rlo, rhi = np.percentile(raw[:, 0], [2.5, 97.5])
        bars.append(("raw NPE", (rhi - rlo) * 1e3, "#8ab17d"))
    bars.append(("reweighted", post_w, "#2a9d8f"))
    bars.append(("floor", floor, "#e76f51"))
    ypos = np.arange(len(bars))[::-1]
    ax.barh(ypos, [b[1] for b in bars], color=[b[2] for b in bars], height=0.62)
    ax.set_yticks(ypos)
    ax.set_yticklabels([b[0] for b in bars], fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel(r"95% width on $J_{\rm CH}$  [mHz]", fontsize=8.5)
    for y, b in zip(ypos, bars):
        ax.text(b[1] * 1.35, y, f"{b[1]:.4g}", va="center", fontsize=7.5)
    ax.set_xlim(floor * 0.4, prior_w * 12)
    ax.tick_params(labelsize=7)
    ax.set_title(f"{post_w / floor:.2f}x the information floor", fontsize=8.5)

    fig.suptitle("Figure 2 — the posterior on a coupling, and its spread\n"
                 r"[$^{13}$C]-formic acid, simulated spectrum, "
                 f"single-round NPE on {N_SIMS:,} simulations, "
                 "reweighted by the exact likelihood",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    return path


def main():  # pragma: no cover - figure
    np.set_printoptions(suppress=True)
    prob = zi.InferenceProblem(seed=SEED)
    ref = np.load("nested_reference.npz")
    theta_true = ref["theta_true"]

    print(f"training/loading the network ({N_SIMS} simulations) ...", flush=True)
    posterior, _ = zi.train_or_load(prob, tag="formic_acid_150k", n_sims=N_SIMS,
                                    seed=SEED, max_num_epochs=150)
    m = zi.evaluate(prob, posterior, theta_true, seed=SEED, n_post=20000)
    s, w = m["samples"], m["weights"]

    print(f"\n  efficiency {m['efficiency']:.1%}, "
          f"ESS {m['efficiency'] * len(s):.0f}")
    for r in zi.shrinkage(prob, s, w):
        print(f"  {r['param']:>12}  width {r['post_width']:<12.5g} "
              f"prior {r['prior_width']:<10.4g} shrinkage {r['shrinkage']:.4f}")

    path = corner(prob, s, w, theta_true, raw=s)
    print(f"\nwrote {path}")


if __name__ == "__main__":       # pragma: no cover
    main()
