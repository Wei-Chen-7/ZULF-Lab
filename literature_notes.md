# What the neighbouring literature does about degeneracy

Written after reading five papers in full, prompted by one question: does the
closest existing work already report non-identifiable couplings, and if so what
is left to claim?

**Short answer.** Nobody produces a posterior over parameters given one
spectrum. Two of the five *do* identify unmeasurable coupling combinations, and
one resolves the residual ambiguity by an explicit ordering convention — which
is exactly the "nuisance broken by hand" the proposal contrasts against, so it
is citable support rather than a threat. But three things the proposal implies
are novel are not, and the claim needs narrowing before submission.

Quotes below are verbatim; whitespace is normalised because the PDF text layer
drops spaces in places. Extracted text is in the scratchpad, not committed.

---

## The five papers

| # | Paper | Degeneracy? | Uncertainty? |
|---|---|---|---|
| [20] | Cobas, *Artif. Intell. Chem.* **4**, 100127 (2026) — PINN for ABC/ABCD | excluded from training | none |
| — | Cobas, *J. Magn. Reson.* **387**, 108061 (2026) — symmetric four-spin | **identified, broken by convention** | none |
| — | 2D-JCOG, *Anal. Chem.* (2026) | equivalent couplings curated out | deep ensemble |
| — | MolDeTr, *Anal. Chem.* **98**, 23399 (2026) | ill-posedness named; equivalents grouped away | ensemble over 5 runs |
| [28] | Neural net analysis of strongly-coupled spin systems | no mention | none |

`[28]` returned **zero** hits across every search term — degeneracy, ambiguity,
uncertainty, error bar, posterior, equivalent nuclei, permutation.

---

## Ref [20] — Cobas, ABC and ABCD spin systems

The closest existing work, and the proposal's citation for it is wrong twice
over (see *Corrections* below).

**It removes degeneracy rather than reporting it.**

> "Exact degeneracies are excluded by the minimum shift-separation filter
> Δν_min ≥ 0.5 Hz used during data generation."

**It breaks the permutation by canonical ordering — on chemical shift.**

> "The dataset stores ground-truth couplings in canonical order after ranking
> the four proton sites by descending chemical-shift frequency."

This sentence is the single most useful thing in the whole literature for us.
The standard fix for the labelling degeneracy *is the chemical shift*, and at
zero field there isn't one. Our scoping to ZULF is not a convenient hedge; it
removes the tool everyone else uses.

**Near-degeneracy makes it fail quietly.**

> "at 500 MHz, A and B differ by about only 41 Hz (8.59 vs 8.51 ppm) and the
> model occasionally returns shifts in swapped slot order"

> "Only 0.42% of samples (24 of 5692) exhibit a physically significant
> misalignment with MAE drop above 0.5 Hz; these cases are concentrated in the
> strong-coupling regime ... where two of the four protons can be
> near-degenerate in chemical shift."

It does audit the labelling — all 24 relabellings of S₄ per sample, identity
optimal in 94.8% — but returns one labelling, with no statement of how much
better it is than the alternative.

**No uncertainty of any kind.** Zero occurrences of posterior, credible
interval, confidence interval, or error bar. Accuracy is reported as MAE:
sub-0.1 Hz on the ABC experimental benchmark, MAE(J) ≈ 0.13 Hz synthetic and
≈ 0.08 Hz experimental for ABCD, in tens of milliseconds per spectrum.

---

## The other Cobas paper — symmetric four-spin systems

Not cited in the proposal, and closer to our territory than ref [20] is. It
does an honest identifiability analysis and then breaks the tie by fiat:

> "Their assignment to J_AA′ and J_BB′ respectively (or vice versa) cannot be
> determined from the spectrum alone and requires additional chemical
> knowledge."

> "In the absence of such prior knowledge, the model adopts the ordering
> convention: J_large = max[(K+M)/2, (K−M)/2], J_small = min[...]"

> "In para-disubstituted benzenes, by contrast, J_AA′ and J_XX′ both correspond
> to meta coupling pathways, making their individual assignment inherently
> ambiguous from the one-dimensional proton spectrum alone. For such systems,
> only the composite parameters K and |M|, together with the inter-group
> couplings J_AX and J_AX′, are reliably extractable."

So: a leading commercial NMR group, in 2026, states which combinations are
extractable and resolves the rest by convention. **Cite this.** It is the
documented instance of the practice we are proposing to replace, and a referee
who finds it first will assume we missed it.

---

## 2D-JCOG and MolDeTr — uncertainty, but not the kind we mean

Both report uncertainty, and both mean **spread across independently trained
networks**, not a posterior over parameters given the data.

2D-JCOG:

> "the distribution of predictions across ensemble members provides a measure
> of uncertainty"

> "the median standard deviation across the five models is effectively zero for
> all complexity categories"

A near-zero ensemble spread says the *networks* agree. It says nothing about
what the *spectrum* constrains — a perfectly flat direction would give five
identical confident answers. That distinction is our strongest reply when a
referee points at these papers, and it is one sentence long.

MolDeTr names the problem squarely:

> "Nonuniqueness: multiple molecular structures may yield indistinguishable or
> nearly identical spectral signatures"

> "Spectra with extreme congestion reflect the ill-posed nature of the inverse
> problem and cannot in general be uniquely resolved"

but its error bars are also ensemble spread — "the mean prediction across five
runs, with error bars denoting the 95% confidence intervals ... across these
runs".

**Both dispose of the unobservable coupling by exclusion.** 2D-JCOG: "Couplings
between magnetically equivalent protons ... were not considered in the analysis
because they fall outside the scope of applicability of the proposed method."
MolDeTr: "we reduce the parameter space by grouping magnetically equivalent
protons". Nobody fits it and reports it as unconstrained, which is what we do.

---

## What we can no longer claim

Three things the proposal implies are ours, and are not:

1. **"No starting guess."** Ref [20]: "It operates on an isolated peak list of
   the target spin system without requiring initial parameters or manual
   transition assignment."
2. **"A front end that seeds a classical fitter."** That is ref [20]'s own
   architecture (neural stage → QM refinement), and MolDeTr proposes the same:
   fitting algorithms "could exploit MolDeTr predictions as priors."
3. **"Nobody quantifies uncertainty."** Two of the five do. The rebuttal is
   about the *kind* of uncertainty, not its absence.

Also worth retiring: that couplings between magnetically equivalent nuclei are
unobservable is textbook, not a discovery. Every one of these papers knows it.
What none of them does is *fit it anyway and let the posterior report it as
unconstrained* — which is what you need when you do not know in advance which
combinations are flat.

## The claim that survives

> Simulation-based inference applied to ZULF NMR, producing a calibrated
> posterior over couplings from a single spectrum, in a regime where the
> standard remedy for the labelling degeneracy — canonical ordering by chemical
> shift — does not exist. Flat directions are fitted and reported as posterior
> shrinkage rather than excluded by curation or resolved by convention.

Every clause is now defensible against a specific paper rather than against a
literature search that found nothing.

## Corrections to make in the proposal

- **Ref [20] has no author.** It is Carlos Cobas, Mestrelab Research S.L.U.
- **Ref [20] has the wrong journal.** It is *Artificial Intelligence Chemistry*
  **4** (2026) 100127, not *Journal of Magnetic Resonance Open*. The PII prefix
  S2949-7477 is that journal's ISSN.
- **Add** Cobas, *J. Magn. Reson.* **387** (2026) 108061, and discuss it — it is
  the nearest prior art on reporting unmeasurable couplings.
- **Add** 2D-JCOG and MolDeTr as the uncertainty-aware comparison, with the
  ensemble-vs-posterior distinction stated explicitly.
