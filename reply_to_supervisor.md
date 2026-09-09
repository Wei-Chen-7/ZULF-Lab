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

The numbers quoted below are from Tables I and II of Wilzewski et al. and were
checked against the paper text on 2026-09-09, not from `results.json`.

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

On noise, drifts and nonlinearities, you are right, and they are not in the
model. What I have is an idealized noise level applied to fitted peak
positions, not something derived from a real FID. The plan has two halves: put
the effects that can be parameterized into the model, and for the rest rely on
a consistency check the method already computes, which measures how well the
data actually support the model and is supposed to collapse when the model is
wrong. I have never tested that check against deliberately wrong data, so I am
setting that up now: spectra carrying an extra coupling the model does not
know about, spectra with field drift during acquisition, and spectra with a
mis-scaled frequency axis. If the check does not fire, the method is not ready
for real data, and I would rather find that out before asking you for any.

Which is why there is no hurry on the archived spectra. I would rather agree on
where this is going first.

Best,
Wei
