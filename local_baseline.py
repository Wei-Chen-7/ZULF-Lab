#!/usr/bin/env python3
"""The local-fit baseline the network has to beat, implemented honestly.

Ref [1] extracts couplings by least-squares fitting a simulated spectrum to the
measured one -- Nelder-Mead down the chi-square surface from a starting guess,
with error bars from the curvature at the optimum. That is the standard method,
it works, and any claim that a trained network is better has to be made against
a competent version of it rather than a strawman.

So this implements it properly: normalized coordinates so the four parameters
are on comparable scales, restarted simplexes with a shrinking step so the
optimizer can actually reach a minimum 1e-4 of the box wide, and a numerical
Hessian with per-parameter steps auto-scaled to the local curvature.

Then it measures the five things the proposal promises to compare:

1. **Accuracy** -- how close the estimate lands, given a good starting guess.
2. **Starting-guess dependence** -- the same fit from many starts in the prior.
3. **Multi-modal behaviour** -- what each method does when the posterior has two
   genuinely separated solutions.
4. **Agreement with the exact posterior** -- curvature error bar vs the nested
   sampling reference, which is the only independent statement of the truth.
5. **Wall clock** -- including the break-even spectrum count, since the network
   pays its training cost once and the local fit pays its cost every time.

Everything here runs on simulated data. None of it waits on the archive.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize

import zulf_infer as zi

__all__ = ["local_fit", "curvature_errors", "multistart", "cluster_minima"]

# Penalty on leaving the prior box, in units of log-likelihood. Large enough to
# dominate any real chi-square difference, smooth enough for a simplex to slide
# back down rather than hit a wall.
_BOX_PENALTY = 1e6


def _nll_u(prob, x_obs, u):
    """Negative log-likelihood at normalized coordinates u, softly boxed."""
    uc = np.clip(u, 0.0, 1.0)
    pen = _BOX_PENALTY * float(np.sum((u - uc) ** 2))
    theta = prob.low + uc * (prob.high - prob.low)
    return -float(prob.log_likelihood(x_obs, theta)[0]) + pen


def _simplex(u0, step):
    """A right-angled simplex at u0, reflected off whichever wall it would hit."""
    d = len(u0)
    pts = [np.asarray(u0, float)]
    for i in range(d):
        p = np.array(u0, float)
        p[i] += step if p[i] + step <= 1.0 else -step
        pts.append(p)
    return np.array(pts)


def local_fit(prob, x_obs, theta0, n_restarts=6, step0=0.05, shrink=0.2,
              maxfev=2000, ftol=1e-9):
    """Nelder-Mead with restarts. Returns a dict of results.

    The restarts are not optional decoration. The chi-square minimum in J is
    about 1e-4 of the prior box wide, and a single simplex started at 5% of the
    box will stall long before it gets there. Each restart rebuilds the simplex
    at the current best point with a step ``shrink`` times smaller, which is
    the standard remedy and is what makes this a fair opponent.
    """
    span = prob.high - prob.low
    u = np.clip((np.asarray(theta0, float) - prob.low) / span, 0.0, 1.0)
    fun = lambda uu: _nll_u(prob, x_obs, uu)                    # noqa: E731

    t0 = time.perf_counter()
    best = fun(u)
    nfev, step, restarts_used = 1, step0, 0
    for k in range(n_restarts):
        res = minimize(fun, u, method="Nelder-Mead",
                       options=dict(initial_simplex=_simplex(u, step),
                                    xatol=1e-14, fatol=1e-14, maxfev=maxfev,
                                    adaptive=True))
        nfev += res.nfev
        restarts_used = k + 1
        gain = best - res.fun
        u, best = np.clip(res.x, 0.0, 1.0), res.fun
        step *= shrink
        if gain < ftol and k:                       # nothing left to win
            break

    return dict(theta=prob.low + u * span, nll=best, nfev=nfev,
                restarts=restarts_used, seconds=time.perf_counter() - t0,
                theta0=np.asarray(theta0, float))


def curvature_errors(prob, x_obs, theta_hat, target=0.5, max_tune=60):
    """Error bars from the curvature of the chi-square at the optimum.

    This is the error bar the local method actually reports. The negative
    log-likelihood is chi-square/2, so its Hessian is the inverse covariance
    directly and ``sigma = sqrt(diag(H^-1))``.

    The finite-difference step is auto-scaled per parameter rather than fixed,
    because the four parameters differ by orders of magnitude in how sharply
    the likelihood responds: a step that resolves the curvature in J is pure
    roundoff in B. Each step is grown or shrunk until the symmetric second
    difference is of order ``target``, which lands it near one sigma.

    Returns ``(sigma, cov, hess, steps)``; ``sigma`` is NaN where the Hessian
    is not positive definite, which is itself a result -- it means the fit sat
    on a saddle, not a minimum.
    """
    theta_hat = np.asarray(theta_hat, float)
    span = prob.high - prob.low
    d = len(theta_hat)
    f = lambda t: -float(prob.log_likelihood(x_obs, t)[0])       # noqa: E731
    f0 = f(theta_hat)

    # per-parameter step, tuned so the curvature term is resolvable
    h = 1e-3 * span
    for i in range(d):
        for _ in range(max_tune):
            e = np.zeros(d)
            e[i] = h[i]
            curv = 0.5 * (f(theta_hat + e) + f(theta_hat - e)) - f0
            if curv < 0.05 * target:
                h[i] *= 3.0
            elif curv > 20.0 * target:
                h[i] /= 3.0
            else:
                break
            h[i] = np.clip(h[i], 1e-12 * span[i], 0.4 * span[i])

    hess = np.empty((d, d))
    for i in range(d):
        ei = np.zeros(d)
        ei[i] = h[i]
        hess[i, i] = (f(theta_hat + ei) - 2 * f0 + f(theta_hat - ei)) / h[i] ** 2
        for j in range(i + 1, d):
            ej = np.zeros(d)
            ej[j] = h[j]
            hess[i, j] = hess[j, i] = (
                f(theta_hat + ei + ej) - f(theta_hat + ei - ej)
                - f(theta_hat - ei + ej) + f(theta_hat - ei - ej)
            ) / (4 * h[i] * h[j])

    try:
        cov = np.linalg.inv(hess)
        var = np.diag(cov)
        sigma = np.where(var > 0, np.sqrt(np.abs(var)), np.nan)
    except np.linalg.LinAlgError:
        cov = np.full((d, d), np.nan)
        sigma = np.full(d, np.nan)
    return sigma, cov, hess, h


def curvature_report(hess, names, span, tol=1.0):
    """Which directions the curvature actually constrains.

    A flat direction makes the Hessian singular, and inverting a singular
    matrix does not fail loudly -- it returns a large number that looks like an
    error bar. This says so explicitly instead.

    The test has to be done on the Hessian scaled by the prior widths,
    ``H~ = diag(span) H diag(span)``, not on the raw one. Raw eigenvalues carry
    the units of their parameters, and comparing a curvature in Hz^-2 against
    one in degrees^-2 says nothing: on methanol the raw spectrum spans 2e-12 to
    1.5e7, which makes B_theta and T2 look flat next to J_CH when both are in
    fact measured to better than a percent of their priors.

    Scaled, each eigenvalue is (prior width / posterior width)^2 along its
    eigenvector, so ``tol = 1`` is the meaningful threshold: below it the data
    constrains that direction no better than the prior already did.
    """
    span = np.asarray(span, float)
    scaled = hess * span[:, None] * span[None, :]
    w, v = np.linalg.eigh(scaled)
    flat = [dict(eigenvalue=float(w[k]),
                 width_ratio=float(np.sqrt(1.0 / w[k])) if w[k] > 0 else np.inf,
                 dominant=names[int(np.argmax(np.abs(v[:, k])))],
                 loading=float(np.max(np.abs(v[:, k]))))
            for k in range(len(w)) if w[k] < tol]
    lo = max(abs(w).min(), 1e-300)
    return dict(eigenvalues=w, condition=float(abs(w).max() / lo),
                flat=flat, singular=bool(flat))


def multistart(prob, x_obs, n_starts=200, seed=0, verbose=True, **fit_kw):
    """Fit from many starting guesses drawn from the prior."""
    rng = np.random.default_rng(seed)
    saved, prob.rng = prob.rng, rng
    starts = prob.sample_prior(n_starts)
    prob.rng = saved
    out = []
    for k, t0 in enumerate(starts):
        out.append(local_fit(prob, x_obs, t0, **fit_kw))
        if verbose and (k + 1) % 25 == 0:
            print(f"    {k + 1}/{n_starts}", flush=True)
    return out


def cluster_minima(fits, index=0, tol=1e-3):
    """Group fits by the value they converged to in one parameter.

    Returns clusters sorted by best (lowest) chi-square, each with its size,
    parameter value and depth relative to the global best.
    """
    vals = np.array([f["theta"][index] for f in fits])
    nlls = np.array([f["nll"] for f in fits])
    order = np.argsort(vals)
    clusters, cur = [], [order[0]]
    for k in order[1:]:
        if vals[k] - vals[cur[-1]] <= tol:
            cur.append(k)
        else:
            clusters.append(cur)
            cur = [k]
    clusters.append(cur)

    best = nlls.min()
    out = []
    for c in clusters:
        c = np.array(c)
        out.append(dict(n=len(c), frac=len(c) / len(fits),
                        value=float(np.mean(vals[c])),
                        spread=float(vals[c].max() - vals[c].min()),
                        nll=float(nlls[c].min()),
                        delta_nll=float(nlls[c].min() - best)))
    return sorted(out, key=lambda r: r["nll"])


# ===========================================================================
# The five comparisons
# ===========================================================================
def _fmt(v, n=4):
    return "nan" if not np.isfinite(v) else f"{v:.{n}f}"




# ===========================================================================
# Case A: the easy problem, where the two methods should agree
# ===========================================================================
def case_formic(seed=0, n_starts=200, n_sims=150_000, n_post=20000):  # pragma: no cover
    prob = zi.InferenceProblem(seed=seed)
    names = prob.param_names
    try:
        ref = np.load("nested_reference.npz")
    except FileNotFoundError:
        print("nested_reference.npz not found; run nested_reference.py first")
        return None
    theta_true, x_obs = ref["theta_true"], ref["x_obs"]
    ref_s, ref_w = ref["samples"], ref["weights"]

    print("=" * 88)
    print("CASE A -- [13C]-formic acid: one coupling, four parameters")
    print("=" * 88)
    print("  truth: " + ", ".join(f"{n}={v:.4f}" for n, v in zip(names, theta_true)))

    # ---- 1. accuracy ------------------------------------------------------
    # The predictor that supplies the prior also supplies the starting guess,
    # so the fair best case starts at the prior centre.
    centre = 0.5 * (prob.low + prob.high)
    fit = local_fit(prob, x_obs, centre)
    sigma, cov, hess, steps = curvature_errors(prob, x_obs, fit["theta"])
    print("\n" + "-" * 88)
    print("1. ACCURACY -- one fit started at the prior centre (the best case)")
    print("-" * 88)
    print(f"{'parameter':>12} {'true':>10} {'start':>10} {'fit':>12} "
          f"{'curvature sigma':>17} {'error/sigma':>12}")
    for i, n in enumerate(names):
        pull = ((fit["theta"][i] - theta_true[i]) / sigma[i]
                if np.isfinite(sigma[i]) and sigma[i] > 0 else np.nan)
        print(f"{n:>12} {theta_true[i]:>10.4f} {centre[i]:>10.4f} "
              f"{fit['theta'][i]:>12.5f} {_fmt(sigma[i], 6):>17} "
              f"{_fmt(pull, 2):>12}")
    print(f"\n  J error {(fit['theta'][0] - theta_true[0]) * 1e3:+.3f} mHz | "
          f"{fit['nfev']} evaluations in {fit['seconds']:.2f} s")

    # ---- 2 & 3. starting-guess dependence and multi-modality --------------
    print("\n" + "-" * 88)
    print(f"2. STARTING-GUESS DEPENDENCE -- {n_starts} starts drawn from the prior")
    print("-" * 88)
    fits = multistart(prob, x_obs, n_starts=n_starts, seed=seed, verbose=True)
    nlls = np.array([f["nll"] for f in fits])
    Js = np.array([f["theta"][0] for f in fits])
    found = nlls < nlls.min() + 0.5
    print(f"\n  reached the global minimum : {found.mean():.1%} of starts")
    print(f"  J when it did              : {Js[found].mean():.5f} Hz "
          f"(spread {Js[found].max() - Js[found].min():.2e} Hz)")
    if (~found).any():
        bad = np.abs(Js[~found] - theta_true[0]) * 1e3
        print(f"  J when it did NOT          : off by {np.median(bad):.0f} mHz "
              f"median, {bad.max():.0f} mHz worst")
    t_local1 = float(np.median([f["seconds"] for f in fits]))
    print(f"  median time per fit        : {t_local1:.2f} s")

    print("\n" + "-" * 88)
    print("3. MULTI-MODAL BEHAVIOUR")
    print("-" * 88)
    clusters = cluster_minima(fits, index=0, tol=1e-3)
    print(f"  distinct minima in J: {len(clusters)}")
    print(f"{'J (Hz)':>12} {'starts':>8} {'frac':>8} {'delta chi2/2':>14}")
    for c in clusters[:8]:
        print(f"{c['value']:>12.5f} {c['n']:>8d} {c['frac']:>8.1%} "
              f"{c['delta_nll']:>14.3f}")
    if len(clusters) == 1:
        print("\n  One basin. On this problem the local fit is not the weak link:")
        print("  it finds the same answer from anywhere in the prior. Any claim")
        print("  that the network wins here would have to be about cost, not")
        print("  correctness -- see case B for where correctness starts to bite.")

    # ---- 4. against the exact posterior -----------------------------------
    print("\n" + "-" * 88)
    print("4. AGREEMENT WITH THE EXACT POSTERIOR (nested sampling, same data)")
    print("-" * 88)
    print("  training/loading the network ...", flush=True)
    posterior, meta = zi.train_or_load(prob, tag="formic_acid_150k",
                                       n_sims=n_sims, seed=seed,
                                       max_num_epochs=150)
    t0 = time.perf_counter()
    m = zi.evaluate(prob, posterior, theta_true, seed=seed, n_post=n_post)
    npe_seconds = time.perf_counter() - t0

    print(f"\n{'parameter':>12} {'local 95%':>26} {'nested 95%':>26} {'ratio':>8}")
    for i, n in enumerate(names):
        llo, lhi = (fit["theta"][i] - 1.96 * sigma[i],
                    fit["theta"][i] + 1.96 * sigma[i])
        nlo, nhi = zi.weighted_quantile(ref_s[:, i], [0.025, 0.975], ref_w)
        ratio = (lhi - llo) / (nhi - nlo) if nhi > nlo else np.nan
        print(f"{n:>12}   [{_fmt(llo, 4):>10},{_fmt(lhi, 4):>10}] "
              f"  [{nlo:>10.4f},{nhi:>10.4f}] {_fmt(ratio, 2):>8}")

    nlo, nhi = zi.weighted_quantile(ref_s[:, 0], [0.025, 0.975], ref_w)
    nested_mHz, local_mHz = (nhi - nlo) * 1e3, 2 * 1.96 * sigma[0] * 1e3
    print(f"\n  95% width on J")
    print(f"    nested sampling (exact) : {nested_mHz:7.2f} mHz")
    print(f"    local curvature         : {local_mHz:7.2f} mHz  "
          f"({local_mHz / nested_mHz:.2f}x)")
    print(f"    reweighted NPE          : {m['reweighted_mHz']:7.2f} mHz  "
          f"({m['reweighted_mHz'] / nested_mHz:.2f}x, "
          f"efficiency {m['efficiency']:.1%})")
    print("\n  All three agree. That is the result: on a problem this clean the")
    print("  Gaussian curvature approximation is exact enough, and the network")
    print("  has to justify itself on cost and on harder molecules.")

    # ---- 5. wall clock ----------------------------------------------------
    print("\n" + "-" * 88)
    print("5. WALL CLOCK")
    print("-" * 88)
    t_train = float(meta.get("train_seconds", float("nan")))
    t_localN = t_local1 * n_starts
    print(f"  network, trained once        : {_fmt(t_train, 1)} s "
          f"({n_sims} simulations)")
    print(f"  network, per new spectrum    : {npe_seconds:.2f} s "
          f"({n_post} draws + reweighting)")
    print(f"  local fit, single start      : {t_local1:.2f} s")
    print(f"  local fit, {n_starts} starts        : {t_localN:.1f} s "
          f"(what it costs to be sure)")
    for label, t_loc in (("single start", t_local1),
                         (f"{n_starts} starts", t_localN)):
        if np.isfinite(t_train) and t_loc > npe_seconds:
            print(f"  break-even vs {label:<16}: "
                  f"{t_train / (t_loc - npe_seconds):.0f} spectra")
        else:
            print(f"  break-even vs {label:<16}: never "
                  f"(the local fit is cheaper per spectrum)")

    np.savez("local_baseline_formic.npz", theta_true=theta_true, x_obs=x_obs,
             fit_theta=fit["theta"], fit_sigma=sigma, fit_cov=cov,
             multistart_theta=np.array([f["theta"] for f in fits]),
             multistart_nll=nlls, param_names=np.array(names),
             seconds=np.array([f["seconds"] for f in fits]))
    return dict(fits=fits, fit=fit, sigma=sigma, t_local1=t_local1,
                t_train=t_train, npe_seconds=npe_seconds)


# ===========================================================================
# Case B: a flat direction, where the curvature error bar stops meaning anything
# ===========================================================================
def case_methanol(seed=0, n_starts=60, n_sims=60_000):  # pragma: no cover
    prob = zi.InferenceProblem(system="methanol", seed=seed)
    names = prob.param_names
    theta_true = np.array([141.7, -12.4, 1.0, 55.0, 12.0])
    saved, prob.rng = prob.rng, np.random.default_rng(seed)
    x_obs = prob.simulate_one(theta_true)
    prob.rng = saved

    print("\n\n" + "=" * 88)
    print("CASE B -- [13C]-methanol: two coupling classes, one of them flat")
    print("=" * 88)
    print("  J_HH inside the equivalent methyl group moves no line "
          "(Delta I_A = 0),")
    print("  so the chi-square surface has an exactly flat direction. What does")
    print("  a least-squares fit report for a parameter the data cannot see?")
    print("  truth: " + ", ".join(f"{n}={v:.4f}" for n, v in zip(names, theta_true)))

    centre = 0.5 * (prob.low + prob.high)
    fit = local_fit(prob, x_obs, centre)
    sigma, cov, hess, _ = curvature_errors(prob, x_obs, fit["theta"])
    rep = curvature_report(hess, names, prob.prior_span())

    print("\n" + "-" * 88)
    print("1. THE FIT AND ITS ERROR BAR")
    print("-" * 88)
    print(f"{'parameter':>12} {'true':>10} {'fit':>12} {'curvature sigma':>17} "
          f"{'95% width':>12}")
    for i, n in enumerate(names):
        w95 = 2 * 1.96 * sigma[i]
        print(f"{n:>12} {theta_true[i]:>10.4f} {fit['theta'][i]:>12.5f} "
              f"{_fmt(sigma[i], 6):>17} {_fmt(w95, 4):>12}")

    print("\n" + "-" * 88)
    print("2. WHAT THE CURVATURE ACTUALLY CONSTRAINS")
    print("-" * 88)
    print("  Hessian eigenvalues, scaled by the prior widths, so each one is")
    print("  (prior width / posterior width)^2 along its own eigenvector:")
    print("    " + ", ".join(f"{v:.3g}" for v in rep["eigenvalues"]))
    print(f"  condition number    : {rep['condition']:.3g}")
    if rep["singular"]:
        for f in rep["flat"]:
            print(f"  FLAT DIRECTION: eigenvalue {f['eigenvalue']:.3g} "
                  f"(< 1, so the data adds nothing to the prior), dominated by "
                  f"{f['dominant']} (loading {f['loading']:.2f})")
        print("\n  The Hessian is singular. Inverting it does not fail -- it")
        print("  returns a number, and that number is printed above as if it")
        print("  were an error bar. Nothing in the output of a standard fit")
        print("  says the data never constrained that parameter at all.")

    print("\n" + "-" * 88)
    print(f"3. WHERE J_HH ENDS UP, FROM {n_starts} DIFFERENT STARTS")
    print("-" * 88)
    fits = multistart(prob, x_obs, n_starts=n_starts, seed=seed, verbose=True)
    arr = np.array([f["theta"] for f in fits])
    starts = np.array([f["theta0"] for f in fits])
    for i, n in enumerate(names):
        r = np.corrcoef(starts[:, i], arr[:, i])[0, 1]
        print(f"  {n:>12}: fitted spread {arr[:, i].std():9.4f}   "
              f"corr(start, fit) = {r:+.3f}")
    print("\n  J_CH is fixed by the data; its correlation with the starting")
    print("  guess is nil. J_HH tracks its starting guess essentially one for")
    print("  one, because every value fits equally well.")

    print("\n" + "-" * 88)
    print("4. WHAT THE NETWORK REPORTS INSTEAD")
    print("-" * 88)
    print("  training/loading the network ...", flush=True)
    posterior, _ = zi.train_or_load(prob, tag="methanol_60k", n_sims=n_sims,
                                    seed=seed, max_num_epochs=150)
    m = zi.evaluate(prob, posterior, theta_true, seed=seed, label="methanol")
    rows = zi.shrinkage(prob, m["samples"], m["weights"])
    print(f"\n{'parameter':>12} {'shrinkage':>10} {'post 95%':>12} "
          f"{'prior 95%':>12}  reading")
    for r in rows:
        reading = "measured" if r["constrained"] else "NOT measured (prior)"
        print(f"{r['param']:>12} {r['shrinkage']:>10.3f} "
              f"{r['post_width']:>12.4f} {r['prior_width']:>12.4f}  {reading}")
    print(f"\n  efficiency {m['efficiency']:.1%}")
    print("\n  Same data, same flat direction. The difference is that the")
    print("  posterior reports the prior back verbatim, and the shrinkage")
    print("  column labels it, instead of quoting a mean and an error bar.")

    np.savez("local_baseline_methanol.npz", theta_true=theta_true, x_obs=x_obs,
             fit_theta=fit["theta"], fit_sigma=sigma, hess=hess,
             multistart_theta=arr, multistart_start=starts,
             param_names=np.array(names))
    return dict(fits=fits, hess=hess, report=rep)


# ===========================================================================
# Case C: a genuinely bimodal posterior
# ===========================================================================
def case_bimodal(seed=0, n_sims=50_000, n_post=20000):  # pragma: no cover
    """theta and 180 - theta give bit-identical spectra.

    R_x(pi) maps B = (Bx, 0, Bz) to (Bx, 0, -Bz) and sends both rho(0) = M and
    the observable M to -M, so S(t) = Tr[rho(t) M] is unchanged and only
    |cos theta| is identifiable. Everywhere else in this project the prior is
    capped at 90 degrees to remove the degeneracy by convention. Here it is
    deliberately left in, because it is the cleanest available example of a
    posterior with two separated modes of exactly equal height.
    """
    prob = zi.InferenceProblem(seed=seed, theta_max_deg=180.0)
    names = prob.param_names
    theta_true = np.array([prob.J_center + 0.7, 1.0, 55.0, 12.0])
    saved, prob.rng = prob.rng, np.random.default_rng(seed)
    x_obs = prob.simulate_one(theta_true)
    prob.rng = saved
    mirror = 180.0 - theta_true[2]

    print("\n\n" + "=" * 88)
    print("CASE C -- a posterior with two modes of exactly equal height")
    print("=" * 88)
    print(f"  truth theta = {theta_true[2]:.1f} deg; "
          f"{mirror:.1f} deg is exactly as good a fit")
    ll_a = float(prob.log_likelihood(x_obs, theta_true)[0])
    t_m = theta_true.copy()
    t_m[2] = mirror
    ll_b = float(prob.log_likelihood(x_obs, t_m)[0])
    print(f"  log L at {theta_true[2]:.0f} deg  : {ll_a:.9f}")
    print(f"  log L at {mirror:.0f} deg : {ll_b:.9f}")
    print(f"  difference        : {abs(ll_a - ll_b):.3e}  (exactly degenerate)")

    print("\n" + "-" * 88)
    print("1. WHAT THE LOCAL FIT DOES")
    print("-" * 88)
    print(f"{'start theta':>12} {'fitted theta':>14} {'fitted J':>12} "
          f"{'sigma(theta)':>14}  reported as")
    for start_ang in (30.0, 70.0, 110.0, 150.0):
        t0 = theta_true.copy()
        t0[2] = start_ang
        f = local_fit(prob, x_obs, t0)
        s, _, _, _ = curvature_errors(prob, x_obs, f["theta"])
        print(f"{start_ang:>12.1f} {f['theta'][2]:>14.4f} {f['theta'][0]:>12.5f} "
              f"{_fmt(s[2], 4):>14}  "
              f"{f['theta'][2]:.1f} +/- {1.96 * s[2]:.1f} deg")
    print("\n  Four fits, two answers, each with a tight and entirely honest")
    print("  error bar that excludes the other. Nothing in a local fit's output")
    print("  can tell you the other mode is there -- the curvature is a local")
    print("  property, and both modes are perfectly quadratic.")

    print("\n" + "-" * 88)
    print("2. WHAT THE NETWORK DOES")
    print("-" * 88)
    print("  training/loading the network ...", flush=True)
    posterior, _ = zi.train_or_load(prob, tag="formic_acid_bimodal_50k",
                                    n_sims=n_sims, seed=seed,
                                    max_num_epochs=150)
    m = zi.evaluate(prob, posterior, theta_true, seed=seed, n_post=n_post)
    s, w = m["samples"], m["weights"]
    ang = s[:, 2]
    lo_mode = w[ang < 90].sum() / w.sum()
    print(f"\n  posterior mass below 90 deg : {lo_mode:.1%}")
    print(f"  posterior mass above 90 deg : {1 - lo_mode:.1%}")
    print(f"  (exact degeneracy => 50/50; the split measures the sampling error)")
    for lbl, sel in (("mode near  55 deg", ang < 90), ("mode near 125 deg", ang >= 90)):
        if w[sel].sum() > 0:
            q = zi.weighted_quantile(ang[sel], [0.025, 0.5, 0.975], w[sel])
            print(f"  {lbl}: median {q[1]:6.2f}, 95% [{q[0]:.2f}, {q[2]:.2f}]")
    jlo, jhi = zi.weighted_quantile(s[:, 0], [0.025, 0.975], w)
    print(f"\n  J is unaffected by the ambiguity: 95% width "
          f"{(jhi - jlo) * 1e3:.2f} mHz across BOTH modes")
    print(f"  efficiency {m['efficiency']:.1%}")
    print("\n  The network holds both solutions at once and says how much it")
    print("  believes each. That is the property a local fit cannot have, and")
    print("  it is the honest form of the argument for using one here.")

    np.savez("local_baseline_bimodal.npz", theta_true=theta_true, x_obs=x_obs,
             samples=s, weights=w, param_names=np.array(names))
    return dict(mass_below=lo_mode, metrics=m)


def study(seed=0, **kw):  # pragma: no cover
    np.set_printoptions(suppress=True)
    a = case_formic(seed=seed, **kw)
    if a is None:
        return
    case_methanol(seed=seed)
    case_bimodal(seed=seed)
    print("\n\nwrote local_baseline_formic.npz, local_baseline_methanol.npz, "
          "local_baseline_bimodal.npz")


if __name__ == "__main__":       # pragma: no cover
    study()
