#!/usr/bin/env python3
"""Calibration and efficiency spread for the tightened network, on one training run.

Two questions, both settled against the same trained proposal:

1. **Is it calibrated?** SBC ranks should be uniform. The failure that matters is
   ranks piling at the edges, meaning the posterior is too narrow and the network
   is confidently wrong.

2. **How much does sampling efficiency move between observations?** The failure
   criterion is stated as a single number, but efficiency is a property of the
   particular observation. The sweep produced 1.7% and 12% for the *same*
   configuration on two different noise realizations, so the spread needs
   measuring before a single-observation threshold can mean anything.

Also re-checks the reweighted posterior against the stored nested-sampling
reference, which is the only independent statement of what the right answer is.
"""

from __future__ import annotations

import numpy as np

import zulf_infer as zi
import sbc_check as sc

N_SIMS = 150_000            # the configuration that won the sweep
SEED = 0


def main():
    np.set_printoptions(suppress=True)
    prob = zi.InferenceProblem(seed=SEED)

    print(f"training NPE on {N_SIMS} simulations (the sweep winner) ...",
          flush=True)
    posterior, _, _ = zi.train_npe(prob, n_sims=N_SIMS, seed=SEED, verbose=False,
                                   max_num_epochs=150)

    # ---- 1. calibration -------------------------------------------------
    n_trials, n_post = 300, 99
    print(f"SBC: {n_trials} trials x {n_post} draws ...", flush=True)
    ranks = sc.run_sbc(prob, posterior, n_trials=n_trials, n_post=n_post,
                       seed=SEED, verbose=False)
    print("\n" + "=" * 88)
    print(f"SBC on the tightened network ({N_SIMS} sims)")
    print("=" * 88)
    print(f"{'param':>12} {'chi2 p':>9} {'KS p':>9} {'outer20%':>10} "
          f"{'centre20%':>10}  verdict")
    rows = sc.diagnose(ranks, n_post)
    for r in rows:
        print(f"{r['param']:>12} {r['p_chi2']:>9.3f} {r['p_ks']:>9.3f} "
              f"{r['outer']:>10.3f} {r['centre']:>10.3f}  {r['verdict']}")
    print("  (both fractions are 0.20 when calibrated)")
    sc.plot_ranks(ranks, n_post, path="sbc_ranks.png",
                  title=f"SBC ranks, NPE trained on {N_SIMS} simulations")
    np.save("sbc_ranks.npy", ranks)

    # ---- 2. efficiency spread -------------------------------------------
    print("\nmeasuring efficiency across observations ...", flush=True)
    effs, widths, _ = zi.efficiency_spread(prob, posterior, n_obs=40, seed=SEED)
    floor = 2 * 1.96 * prob.sigma_f / np.sqrt(3) * 1e3
    print("\n" + "=" * 88)
    print("Sampling efficiency across 40 observations drawn from the prior")
    print("=" * 88)
    q = np.percentile(effs, [0, 5, 25, 50, 75, 95, 100])
    print(f"  min {q[0]:.2%} | 5% {q[1]:.2%} | 25% {q[2]:.2%} | median {q[3]:.2%}"
          f" | 75% {q[4]:.2%} | 95% {q[5]:.2%} | max {q[6]:.2%}")
    print(f"  spread max/min      : {q[6]/max(q[0],1e-9):.0f}x")
    print(f"  fraction below 1%   : {np.mean(effs < 0.01):.1%}")
    print(f"\n  reweighted width on J: median {np.median(widths):.2f} mHz, "
          f"90% within [{np.percentile(widths,5):.2f}, {np.percentile(widths,95):.2f}] mHz")
    print(f"  information floor    : {floor:.2f} mHz")
    print(f"  fraction wider than 10 mHz (the criterion): "
          f"{np.mean(widths > 10):.1%}")

    # ---- 3. against the nested-sampling reference ------------------------
    try:
        ref = np.load("nested_reference.npz")
        theta_true = ref["theta_true"]
        m = zi.evaluate(prob, posterior, theta_true, seed=SEED, label="tightened")
        rs, rw = ref["samples"], ref["weights"]
        rlo, rhi = zi.weighted_quantile(rs[:, 0], [0.025, 0.975], rw)
        print("\n" + "=" * 88)
        print("Against the nested-sampling reference (same observation)")
        print("=" * 88)
        print(f"  nested sampling : {(rhi - rlo)*1e3:6.2f} mHz")
        print(f"  reweighted NPE  : {m['reweighted_mHz']:6.2f} mHz  "
              f"(efficiency {m['efficiency']:.1%})")
        print(f"  agreement       : {abs(m['reweighted_mHz']-(rhi-rlo)*1e3)/((rhi-rlo)*1e3):.1%}")
    except FileNotFoundError:
        print("\n(no nested_reference.npz; run nested_reference.py first)")

    np.savez("final_check.npz", ranks=ranks, effs=effs, widths=widths)
    print("\nwrote sbc_ranks.png, sbc_ranks.npy, final_check.npz")


if __name__ == "__main__":
    main()
