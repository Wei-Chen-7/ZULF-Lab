#!/usr/bin/env python3
"""Calibration and efficiency spread for the tightened network, on one training run.

Two questions, both settled against the same trained proposal:

1. **Is it calibrated?** SBC ranks should be uniform. The failure that matters is
   ranks piling at the edges, meaning the posterior is too narrow and the network
   is confidently wrong.

2. **How much does sampling efficiency move between observations?** The failure
   criterion is stated as a single number, but efficiency is a property of the
   particular observation, not just of the network. It ranges over 1.5% to 45%
   across draws from the same prior with the same network, so the spread needs
   measuring before a single-observation threshold can mean anything.

Also re-checks the reweighted posterior against the stored nested-sampling
reference, which is the only independent statement of what the right answer is.
"""

from __future__ import annotations

import numpy as np

import zulf_infer as zi
import sbc_check as sc
import results

# The nsf default. tighten_proposal.py finds the wide flow (96 features, 8
# transforms) reaches 37% efficiency against this one's 24%, at the same
# reweighted precision -- so this is not the best proposal, only the one every
# other result here was computed with.
N_SIMS = 150_000
SEED = 0


def _local_sigma_J(prob, ref):
    """The least-squares baseline's own error bar on J, and how independent it is.

    Returns ``(sigma_J, rel_diff)``. The second number matters for how the
    comparison is presented: for a Gaussian likelihood the observed information
    at the optimum IS the expected information, so the curvature error bar and
    the Cramer-Rao floor are the same quantity reached two ways, not two
    independent estimates. Quoting them as separate rows implies corroboration
    that is not there, so the size of the gap is worth recording.
    """
    import local_baseline as lb
    import numpy as np
    fit = lb.local_fit(prob, ref["x_obs"], 0.5 * (prob.low + prob.high))
    sigma, _, hess, _ = lb.curvature_errors(prob, ref["x_obs"], fit["theta"])
    fisher, _ = zi.fisher_matrix(prob, fit["theta"])
    rel = float(np.abs(hess - fisher).max() / np.abs(fisher).max())
    return float(sigma[0]), rel


def main():
    np.set_printoptions(suppress=True)
    prob = zi.InferenceProblem(seed=SEED)
    try:
        ref = np.load("nested_reference.npz")
    except FileNotFoundError:
        ref = None

    print(f"training/loading NPE on {N_SIMS} simulations ...", flush=True)
    posterior, _ = zi.train_or_load(prob, tag="formic_acid_150k", n_sims=N_SIMS,
                                    seed=SEED, max_num_epochs=150)

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
    rows = sc.diagnose(ranks, n_post, names=prob.param_names)
    for r in rows:
        print(f"{r['param']:>12} {r['p_chi2']:>9.3f} {r['p_ks']:>9.3f} "
              f"{r['outer']:>10.3f} {r['centre']:>10.3f}  {r['verdict']}")
    print("  (both fractions are 0.20 when calibrated)")
    sc.plot_ranks(ranks, n_post, path="sbc_ranks.png",
                  title=f"SBC ranks, NPE trained on {N_SIMS} simulations",
                  names=prob.param_names)
    np.save("sbc_ranks.npy", ranks)

    # ---- 2. efficiency spread -------------------------------------------
    print("\nmeasuring efficiency across observations ...", flush=True)
    effs, widths, _ = zi.efficiency_spread(prob, posterior, n_obs=40, seed=SEED)
    # The floor depends on where in the prior the observation sits, so quote it
    # at the reference point the widths below are compared against.
    anchor = ref["theta_true"] if ref is not None else 0.5 * (prob.low + prob.high)
    floor = zi.information_floor(prob, anchor)[0] * 1e3
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
    if ref is None:
        print("\n(no nested_reference.npz; run nested_reference.py first)")
    else:
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
        nested_mHz = (rhi - rlo) * 1e3
        sigma_J, curv_vs_fisher = _local_sigma_J(prob, ref)
        results.record("vs_nested", dict(
            nested_mHz=nested_mHz, npe_mHz=m["reweighted_mHz"],
            local_mHz=2 * 1.96 * sigma_J * 1e3,
            curvature_vs_fisher_rel=curv_vs_fisher,
            floor_mHz=zi.information_floor(prob, theta_true)[0] * 1e3,
            efficiency=m["efficiency"],
            agreement=abs(m["reweighted_mHz"] - nested_mHz) / nested_mHz))

    j = rows[0]
    results.record("sbc", dict(
        n_trials=n_trials, n_post=n_post,
        J_outer=j["outer"], J_centre=j["centre"], J_verdict=j["verdict"],
        calibrated=[r["param"] for r in rows[1:]
                    if "calibrated" in r["verdict"]]))
    results.record("efficiency", dict(
        n_obs=40, min=float(q[0]), median=float(q[3]), max=float(q[6]),
        spread=float(q[6] / max(q[0], 1e-9)),
        below_one_percent=float(np.mean(effs < 0.01)),
        median_width_mHz=float(np.median(widths)), floor_mHz=float(floor)))

    np.savez("final_check.npz", ranks=ranks, effs=effs, widths=widths)
    print("\nwrote sbc_ranks.png, sbc_ranks.npy, final_check.npz")


if __name__ == "__main__":
    main()
