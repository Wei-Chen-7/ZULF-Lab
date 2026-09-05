#!/usr/bin/env python3
"""Can the NPE proposal be tightened enough to lift the sampling efficiency?

The first run reached the information floor on J *after* importance reweighting,
but at only 1.7% sample efficiency against the project's 1% floor. That matters
for a reason beyond speed: efficiency is the misspecification detector. It is
supposed to fall when the model is wrong on real data, and a detector sitting at
1.7x its own threshold in the perfectly-specified simulated case cannot
distinguish "the model is wrong" from "the proposal was always loose".

Efficiency is governed by how well the network proposal q(theta|x) matches the
true posterior. The raw NPE posterior on J was about 5x wider than the true one,
so most samples land in the wings and carry negligible weight. This script
compares configurations to see what actually helps.
"""

from __future__ import annotations

import json
import time

import numpy as np

import zulf_infer as zi

# Capped at 150 epochs so the sweep stays bounded; the baseline converged in
# 123 epochs, so this is not a binding constraint for it.
CONFIGS = [
    dict(label="baseline: 50k, nsf default",
         n_sims=50_000, density_estimator="nsf", max_num_epochs=150),
    dict(label="150k sims, nsf default",
         n_sims=150_000, density_estimator="nsf", max_num_epochs=150),
    dict(label="150k sims, nsf wide (96 feat, 8 transforms)",
         n_sims=150_000, density_estimator="nsf",
         hidden_features=96, num_transforms=8, max_num_epochs=150),
]


def main():
    seed = 0
    theta_true = None
    rows = []
    for cfg in CONFIGS:
        label = cfg.pop("label")
        prob = zi.InferenceProblem(seed=seed)
        if theta_true is None:
            theta_true = np.array([prob.J_center + 0.7, 1.0, 55.0, 12.0])
        t0 = time.perf_counter()
        posterior, _, _ = zi.train_npe(prob, seed=seed, verbose=False, **cfg)
        train_s = time.perf_counter() - t0
        m = zi.evaluate(prob, posterior, theta_true, seed=seed, label=label)
        m["train_s"] = train_s
        rows.append(m)
        print(f"{label:48s} raw {m['raw_mHz']:7.2f} mHz  "
              f"rew {m['reweighted_mHz']:6.2f} mHz  "
              f"eff {m['efficiency']:6.2%}  "
              f"raw/floor {m['raw_over_floor']:5.1f}x  "
              f"({train_s/60:.1f} min)", flush=True)

    print("\n" + "=" * 92)
    print("Sample efficiency is set by how close the proposal is to the true")
    print("posterior; raw/floor is that mismatch measured in units of the")
    print("information floor. Efficiency should track it inversely.")
    print("=" * 92)
    best = max(rows, key=lambda r: r["efficiency"])
    base = rows[0]
    print(f"\n  baseline efficiency : {base['efficiency']:.2%}")
    print(f"  best efficiency     : {best['efficiency']:.2%}  ({best['label']})")
    print(f"  improvement         : {best['efficiency']/base['efficiency']:.1f}x")
    print(f"  headroom over 1%    : {best['efficiency']/0.01:.1f}x")

    out = "tighten_results.json"
    with open(out, "w") as fh:
        json.dump([{k: (float(v) if isinstance(v, (int, float, np.floating))
                        else v)
                    for k, v in r.items()
                    if k not in ("samples", "weights", "x_obs")}
                   for r in rows], fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
