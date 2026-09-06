#!/usr/bin/env python3
"""Methanol: a measured coupling and an unmeasurable one, reported side by side.

[13C]-methanol's methyl group is the sharp identifiability case. It has six
pairwise couplings but only two symmetry-distinct classes, and only one of those
is measurable: the Delta I_A = 0 selection rule means the proton-proton coupling
inside the equivalent group moves no line at all. A posterior over all six pairs
would report the prior back in five directions while looking like a result.

So the problem is parameterized by the two classes, both left free, and the
shrinkage from prior to posterior is reported for each. J_CH should be pinned;
J_HH should come back as its own prior, and be labelled as such rather than
quoted as a measurement.

This is a four-spin system, so it also exercises the inference layer past the
two-spin case.
"""

from __future__ import annotations

import numpy as np

import zulf_infer as zi

N_SIMS = 60_000
SEED = 0


def main():
    np.set_printoptions(suppress=True)
    prob = zi.InferenceProblem(system="methanol", seed=SEED)
    print(f"[13C]-methanol, {prob.sys.n} spins, "
          f"{len(prob.param_names)} free parameters: {prob.param_names}")
    print(f"summary vector length {prob.x_dim()}")
    print(f"training NPE on {N_SIMS} simulations ...", flush=True)
    posterior, _, _ = zi.train_npe(prob, n_sims=N_SIMS, seed=SEED,
                                   verbose=False, max_num_epochs=150)

    # A truth with a deliberately non-zero J_HH, to show it cannot be recovered.
    theta_true = np.array([141.0, -12.4, 1.0, 55.0, 12.0])
    m = zi.evaluate(prob, posterior, theta_true, seed=SEED, label="methanol")
    s, w = m["samples"], m["weights"]

    print("\n" + "=" * 84)
    print("Posterior after importance reweighting")
    print("=" * 84)
    print(f"{'parameter':>12} {'true':>9} {'mean':>10} {'95% interval':>24} "
          f"{'shrinkage':>10}")
    rows = zi.shrinkage(prob, s, w)
    for i, r in enumerate(rows):
        lo, hi = zi.weighted_quantile(s[:, i], [0.025, 0.975], w)
        mean = np.average(s[:, i], weights=w)
        print(f"{r['param']:>12} {theta_true[i]:>9.3f} {mean:>10.3f} "
              f"  [{lo:9.3f},{hi:9.3f}] {r['shrinkage']:>10.3f}")

    jch, jhh = rows[0], rows[1]
    print(f"\n  efficiency {m['efficiency']:.1%}")
    print("\n" + "-" * 84)
    print(f"  J_CH  shrinkage {jch['shrinkage']:.3f}  -> measured "
          f"({jch['post_width']*1e3:.2f} mHz from a {jch['prior_width']:.1f} Hz prior)")
    print(f"  J_HH  shrinkage {jhh['shrinkage']:.3f}  -> NOT measured; the "
          f"posterior is the prior")
    print("\n  The second line is the point. J_HH is a flat direction, and the")
    print("  posterior says so instead of returning a confident number. Quoting")
    print("  a mean and an error bar for it would be reporting the prior back.")

    np.savez("methanol_posterior.npz", samples=s, weights=w,
             theta_true=theta_true, param_names=np.array(prob.param_names))
    print("\n  wrote methanol_posterior.npz")


if __name__ == "__main__":
    main()
