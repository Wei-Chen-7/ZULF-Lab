#!/usr/bin/env python3
"""Train and cache a network per molecule, and report what each one measures.

One of the proposal's deliverables is "trained networks for a few small
molecules, ready to use". This produces them and, more importantly, produces
the honest accompanying statement: for each molecule, which couplings the
network actually measures and which come back as the prior.

That second column is not decoration. Three of the four molecules here have an
equivalent proton group, so their J_HH is a flat direction -- the Delta I_A = 0
selection rule means it moves no line. A library that shipped a posterior mean
for it without saying so would be shipping the prior back as a measurement.

Networks land in ``models/`` and are reused by every other script here, so this
is the expensive step and everything downstream is cheap.

    python train_library.py           # trains what is missing, reports all
    python train_library.py --force   # retrain everything
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

import zulf_infer as zi

MANIFEST = os.path.join(zi.MODEL_DIR, "manifest.json")

#: One entry per molecule: how many simulations, the cache tag, and a truth to
#: demonstrate on. Truths put the measurable coupling off the prior centre on
#: purpose, so recovering it is not an artefact of starting there.
LIBRARY = {
    "formic_acid":  dict(tag="formic_acid_150k", n_sims=150_000,
                         truth=[222.9, 1.0, 55.0, 12.0]),
    "formaldehyde": dict(tag="formaldehyde_60k", n_sims=60_000,
                         truth=[164.4, 8.0, 1.0, 55.0, 12.0]),
    "glycine":      dict(tag="glycine_60k", n_sims=60_000,
                         truth=[140.6, -5.0, 1.0, 55.0, 12.0]),
    "methanol":     dict(tag="methanol_60k", n_sims=60_000,
                         truth=[141.7, -12.4, 1.0, 55.0, 12.0]),
}

SEED = 0


def build(name, spec, seed=SEED, force=False, n_post=20000):
    prob = zi.InferenceProblem(system=name, seed=seed)
    t0 = time.perf_counter()
    posterior, meta = zi.train_or_load(prob, tag=spec["tag"],
                                       n_sims=spec["n_sims"], seed=seed,
                                       force=force, max_num_epochs=150)
    load_or_train_s = time.perf_counter() - t0
    truth = np.asarray(spec["truth"], float)
    if len(truth) != len(prob.param_names):
        raise ValueError(f"{name}: truth has {len(truth)} entries, "
                         f"expected {len(prob.param_names)}")

    t0 = time.perf_counter()
    m = zi.evaluate(prob, posterior, truth, seed=seed, n_post=n_post)
    per_spectrum_s = time.perf_counter() - t0
    rows = zi.shrinkage(prob, m["samples"], m["weights"])
    for i, r in enumerate(rows):
        r["true"] = float(truth[i])
        r["mean"] = float(np.average(m["samples"][:, i], weights=m["weights"]))
    return prob, meta, m, rows, load_or_train_s, per_spectrum_s


def main():  # pragma: no cover - study
    force = "--force" in sys.argv
    np.set_printoptions(suppress=True)
    manifest, summary = {}, []

    for name, spec in LIBRARY.items():
        print(f"\n{'=' * 84}\n{name}\n{'=' * 84}", flush=True)
        prob, meta, m, rows, t_build, t_obs = build(name, spec, force=force)
        trained = not meta.get("cached", False)
        print(f"  {prob.sys.n} spins, {len(prob.param_names)} parameters, "
              f"summary length {prob.x_dim()}, {spec['n_sims']:,} simulations")
        if trained:
            print(f"  trained in {meta['train_seconds']:.1f} s")
        else:
            print(f"  loaded from cache in {t_build:.2f} s "
                  f"(originally trained in {meta['train_seconds']:.1f} s)")
        print(f"\n{'parameter':>13} {'true':>10} {'mean':>11} {'95% width':>12} "
              f"{'shrinkage':>10}  reading")
        for r in rows:
            reading = "measured" if r["constrained"] else "NOT measured (prior)"
            print(f"{r['param']:>13} {r['true']:>10.4f} {r['mean']:>11.4f} "
                  f"{r['post_width']:>12.5g} {r['shrinkage']:>10.3f}  {reading}")
        print(f"\n  efficiency {m['efficiency']:.1%}, "
              f"{t_obs:.2f} s per new spectrum")

        couplings = rows[:prob.n_couplings]
        measured = [r for r in couplings if r["constrained"]]
        flat = [r for r in couplings if not r["constrained"]]
        manifest[name] = dict(
            tag=spec["tag"], n_sims=spec["n_sims"], n_spins=int(prob.sys.n),
            params=list(prob.param_names), x_dim=int(prob.x_dim()),
            merge_hz=prob.merge_hz, sigma_f=prob.sigma_f,
            efficiency=float(m["efficiency"]),
            seconds_per_spectrum=float(t_obs),
            measured=[r["param"] for r in measured],
            flat=[r["param"] for r in flat],
            widths_mHz={r["param"]: float(r["post_width"] * 1e3)
                        for r in couplings},
            shrinkage={r["param"]: float(r["shrinkage"]) for r in couplings})
        summary.append((name, prob, measured, flat, m))

    print(f"\n\n{'=' * 84}")
    print("LIBRARY SUMMARY -- what each network measures")
    print("=" * 84)
    print(f"{'molecule':>14} {'spins':>6} {'measured':>26} {'width':>11} "
          f"{'flat':>8} {'eff':>7}")
    for name, prob, measured, flat, m in summary:
        got = ", ".join(f"{r['param']}" for r in measured) or "-"
        wid = ", ".join(f"{r['post_width'] * 1e3:.2f}" for r in measured) or "-"
        lost = ", ".join(r["param"] for r in flat) or "none"
        print(f"{name:>14} {prob.sys.n:>6} {got:>26} {wid:>8} mHz "
              f"{lost:>8} {m['efficiency']:>6.1%}")

    n_flat = sum(len(f) for _, _, _, f, _ in summary)
    print(f"\n  {n_flat} of the coupling classes across this library are flat")
    print("  directions. Each is reported as such rather than quoted, which is")
    print("  the whole reason for parameterizing by symmetry class.")

    os.makedirs(zi.MODEL_DIR, exist_ok=True)
    with open(MANIFEST, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    print(f"\nwrote {MANIFEST} and {len(LIBRARY)} networks in {zi.MODEL_DIR}/")


if __name__ == "__main__":       # pragma: no cover
    main()
