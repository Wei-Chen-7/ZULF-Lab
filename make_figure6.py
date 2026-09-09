#!/usr/bin/env python3
"""Figure 6 - what happens when the model is wrong, and whether anything warns you.

Two rows, one column per misspecification.

Top row is accuracy: how far the reported coupling moves, against the error bar
it is reported with. While the point stays inside the band the answer is still
defensible; once it leaves, the method is confidently wrong.

Bottom row is the warning. Two candidate detectors are drawn together: the
goodness of fit at the best-fit point, and the importance-sampling efficiency
relative to its value on clean data. The question the figure is built to answer
is whether the warning arrives *before* the answer goes wrong.

Run ``python misspecification.py`` first to write misspecification.npz.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "figure6_misspecification.png"

C_BIAS = "#c2543a"
C_BAND = "#d3e8e4"
C_CHI2 = "#1f7d72"
C_EFF = "#264653"

PANELS = [
    ("equivalence",
     "(a) Two couplings the model\ncalls equal, and are not",
     r"imbalance $\delta$ on $J_{\rm CH}$  [Hz]", False),
    ("drift",
     "(b) The field drifts\nduring acquisition",
     r"field excursion  [nT]", False),
    ("axis_scale",
     "(c) The frequency axis is\nmis-scaled by $1+\\epsilon$",
     r"$\epsilon$", True),
]


def load(path="misspecification.npz"):
    d = np.load(path, allow_pickle=True)
    kind = np.array([str(k) for k in d["kind"]])
    return d, kind


def _x(levels, logx):
    """Positions to plot at. A zero level cannot go on a log axis."""
    if not logx:
        return np.asarray(levels, float), None
    lv = np.asarray(levels, float)
    pos = lv.copy()
    nz = lv[lv > 0]
    pos[lv <= 0] = nz.min() / 3.0 if len(nz) else 1.0
    return pos, float(nz.min() / 3.0) if len(nz) else None


def main():
    d, kind = load()
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 7.2))

    for col, (name, title, xlabel, logx) in enumerate(PANELS):
        m = kind == name
        lv = d["level"][m]
        bias = np.abs(d["bias_mHz"][m])
        half = 0.5 * d["width_mHz"][m]
        chi2 = d["chi2_per_dof"][m]
        eff = d["efficiency"][m]
        pos, zero_at = _x(lv, logx)

        # ---- top: is the answer still right? ----------------------------
        ax = axes[0, col]
        ax.fill_between(pos, 0, half, color=C_BAND,
                        label="reported 95% half-width")
        ax.plot(pos, bias, "o-", color=C_BIAS, lw=2.0, ms=5.5,
                label=r"shift in $J_{\rm CH}$")
        bad = bias > half
        if bad.any():
            ax.plot(pos[bad], bias[bad], "o", ms=11, mfc="none",
                    mec=C_BIAS, mew=1.8)
        if logx:
            ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel("mHz", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=7.5, loc="upper left", framealpha=0.92)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25, lw=0.6)

        # ---- bottom: does anything warn you? ----------------------------
        ax = axes[1, col]
        ax.plot(pos, np.maximum(chi2, 1e-3), "s-", color=C_CHI2, lw=2.0,
                ms=5.0, label="goodness of fit  $\\chi^2/$dof")
        ax.axhline(1.0, color=C_CHI2, lw=0.9, ls=":")
        if logx:
            ax.set_xscale("log")
        ax.set_yscale("log")
        # One scale across all three panels. Left to autoscale, a dead-flat
        # chi2 gets stretched over a range of 0.06 and reads as a rising
        # trend, which is the opposite of what these panels show.
        ax.set_ylim(0.5, 60.0)
        ax.set_ylabel(r"$\chi^2/$dof", fontsize=9, color=C_CHI2)
        ax.tick_params(axis="y", labelcolor=C_CHI2, labelsize=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.grid(alpha=0.25, lw=0.6)

        ax2 = ax.twinx()
        ax2.plot(pos, eff / max(eff[0], 1e-12), "^--", color=C_EFF, lw=1.6,
                 ms=5.0, label="efficiency / clean")
        ax2.set_ylabel("efficiency, relative to clean", fontsize=9,
                       color=C_EFF)
        ax2.tick_params(axis="y", labelcolor=C_EFF, labelsize=8)
        ax2.set_ylim(0, 1.35)

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="upper left",
                  framealpha=0.92)

        if zero_at is not None:
            for a in (axes[0, col], axes[1, col]):
                a.set_xticks([zero_at] + [v for v in lv if v > 0])
                a.set_xticklabels(["0"] + [f"{v:g}" for v in lv if v > 0])

    fig.suptitle("Figure 6 — three ways the model can be wrong, and "
                 "whether the method notices", fontsize=12.5,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(OUT, dpi=150, facecolor="white", bbox_inches="tight")
    print(f"wrote {OUT}")

    for name, _, _, _ in PANELS:
        m = kind == name
        bias, half = np.abs(d["bias_mHz"][m]), 0.5 * d["width_mHz"][m]
        chi2 = d["chi2_per_dof"][m]
        caught = chi2 > 2.0
        wrong = bias > half
        print(f"  {name:>12}: wrong at {int(wrong.sum())}/{len(wrong)} levels, "
              f"chi2 fires at {int(caught.sum())}/{len(caught)}")


if __name__ == "__main__":
    raise SystemExit(main())
