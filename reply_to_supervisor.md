# Draft: reply to Dima on scope

Send the text below the line. Nothing to fill in.

Three things this draft deliberately does.

It **concedes the point** in the first two sentences. He asked what the project
is for, and the honest answer is that there was no settled answer. Arguing
would be worse than useless with someone who has seen a hundred students do it.

It **answers with his own data**, not with a pitch. Every number below comes
from Wilzewski et al., the paper the forward model has been validated against.
That turns "here is why my method is good" into "here is a problem visible in
our own published table", which is a much harder thing to wave away.

It **gives up the precision claim**. This is the part worth being deliberate
about: the 1 to 2 mHz widths are the most impressive-looking numbers in the
whole project, and they are not the point. The existing fit already reports
1 mHz. Leading with precision would invite exactly the reply that the field is
not statistics-limited, and the rest of the email would not get read.

The Wilzewski numbers below are from Tables I and II of that paper and were
checked against the paper text on 2026-09-09. The misspecification numbers
(28x, 12 mHz, 22 mHz, nine times the error bar) come from
`python misspecification.py` and are recorded in `results.json`; Figure 6 of
`readout.pdf` plots them, so attach the report again if you send this.

---

**Subject:** Re: report, and what the project is actually for

Dear Dima,

Thank you, and you are asking the right question. I did not have a settled
answer, so let me say where I now think the destination is and ask which
version is worth steering towards.

When I went looking, I found the answer in Wilzewski et al., the paper I have
been validating the forward model against. In the benzene table, ¹J_CH is
fitted to 158.363(1) Hz against a literature value of 158.354(1), a nine sigma
difference. In benzaldehyde, ³J_HH(H3,H4) comes out 72 mHz from the literature
value with a 7 mHz error bar. The paper says so plainly, that the disagreement
is larger than the quadrature sum of the uncertainties, and attributes it to
sample preparation, temperature, and residual dipolar couplings that are hard
to tell apart from J-couplings.

That tells me two things. First, ZULF J-coupling measurement is not limited by
statistical precision, so a more precise fit is not a goal worth having. My
networks reach a 95% interval of roughly 1 to 2 mHz on simulated spectra, which
is the same order the existing least-squares fit already achieves on real ones.
Second, the quoted uncertainties do not capture what actually separates two
measurements of the same quantity.

There is a related detail in the same paper. The authors say they chose fit
parameters that do not co-vary, because their uncertainties come from the
diagonal of the inverse covariance matrix. That is a careful workaround, done
by hand, for something a posterior handles on its own. They also read "zero
within error" as evidence for the fast-exchange limit, which holds only if the
parameter was constrained to zero rather than simply unconstrained by the data.
Telling those two cases apart is the one thing my method does that a
least-squares fit structurally cannot.

So, three candidate destinations. I would value your view on which is actually
useful to the group.

1. **Metrology.** The product is an uncertainty that survives parameter
   correlations, separates "unmeasured" from "measured to be zero", and folds
   systematics such as residual dipolar couplings and temperature into the model
   as parameters rather than leaving them in a caveat.
2. **A tool for new-physics searches.** The product is a calibrated joint
   uncertainty over the conventional parameters, so a bound on an anomalous
   interaction is not corrupted by a degeneracy with them.
3. **Throughput.** Many spectra, no starting guesses, no supervision per
   spectrum. Honest, but the weakest of the three, since comparable tools
   already exist at high field.

On noise, drifts and nonlinearities you are right, and they are not in the
model. What I have is an idealized noise level applied to fitted peak
positions, not something derived from a real FID. The method does compute a
consistency check that is supposed to collapse when the model cannot explain
the data, and I had been relying on that. Your email made me notice I had never
actually tested it, because every spectrum it had ever seen came from the same
simulator it fits with. So I built three spectra it cannot explain. The result
is that it mostly does not notice.

Field drift during acquisition is caught, loudly. The fit quality degrades by a
factor of 28, the consistency check falls to a third of its clean value, and
the coupling stays inside its error bar because the error bar widens to match.
That is the behaviour I had been assuming throughout.

The other two are silent failures. If two protons the model treats as
equivalent actually differ by 2 Hz, which is the scale of the residual dipolar
couplings you would expect in a partially aligned aromatic, the reported
coupling moves by 12 mHz, about nine times its own error bar, and every
diagnostic sits exactly where it was on clean data. I traced the reason.
Breaking the equivalence does produce a signature, namely lines near J and J/2
that the selection rule forbids in the correct model, so nothing else could put
them there. But they come out four orders of magnitude weaker than the main
multiplet, and my peak-list summary keeps only the strongest few lines. I am
throwing the evidence away before the fit ever sees it. That one is fixable.

The third is not fixable by better statistics. A frequency axis mis-scaled by
one part in 10^4 shifts the coupling by 22 mHz with no change in any
diagnostic, and that is provable rather than merely observed: scaling every
frequency by 1+eps is reproduced exactly by scaling J and the field up and T2
down. The two predictions agree to under a millionth of the measurement noise,
so there is no residual left for any test to find. The frequency axis has to be
calibrated independently and its uncertainty carried through by hand.

The honest summary is that the method catches model errors pointing away from
its parameters and is blind to those pointing along them. That is the wrong way
round, because the second kind is the dangerous kind: they get absorbed into a
plausible-looking parameter value instead of showing up as a bad fit.

Which is also why there is no hurry on the archived spectra. I would rather fix
that and agree on where this is going first.

Best,
Wei
