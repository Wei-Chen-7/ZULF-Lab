# Draft: archived-data request

Edit and send the text below the line. Two things to fill in: the supervisor's
name, and the deadline in the closing paragraph if you want one. **Attach
`readout.pdf`**, since the body refers to it.

Unlike the README and `readout.tex`, this draft is *not* generated from
`results.json`. It is a one-off you will edit before sending, and regenerating
it would fight that. So if it sits unsent for a while, re-check the three
figures it quotes (7.2 mHz, 1.0 to 2.3 mHz, 2%) against
`python results.py --show` before it goes out. They were last verified against
the store on 2026-09-09 and were all current.

Two things about how it is written. The jargon is explained where it appears,
because the terms that matter here (information floor, nested sampling,
calibration) are inference words rather than NMR words, and there is no reason
to make a reader guess. And the split between Essential and Helpful matters:
Essential is what Figure 5 cannot be made without, Helpful is what stops us
having to assume something and then defend the assumption in the paper. Ask for
both, but make clear that the first three are the blocking ones. A request that
looks like seven demands is easier to postpone than one that looks like three.

---

**Subject:** Archived ZULF spectra and pulse sequences for the J-coupling project

Dear [supervisor],

The simulation half of the J-coupling project is finished, and the next step
needs real spectra. Could I get access to the group's archived ZULF data?

I have attached a short report with the figures, so I will keep this brief.
Where things stand:

**The forward model reproduces a real spectrum.** With nothing fitted, it
predicts the benzene-¹³C₁ multiplet published by Wilzewski et al. from their own
couplings, with a scatter of 7.2 mHz. The entire remaining residual is a single
offset that shifts every line by the same amount, which is what reading a
figure's frequency axis slightly wrong produces. An error in the physics would
shift the lines by different amounts.

**The trained networks measure the coupling as precisely as the noise allows.**
For formic acid, formaldehyde, glycine and methanol, each network pins ¹J_CH
down to a 95% interval between 1.0 and 2.3 mHz wide. There is a hard limit on
how narrow that interval can be, set by the noise, and I compute it from the
simulator itself. Every network sits on that limit. As an independent check I
also ran a slow but exact calculation on the same data, and the two intervals
differ by 2%, so the fast method is not cutting corners.

**The error bars are honest.** I test this by simulating spectra from known
couplings and checking how often the true value falls where the network says it
should. The field and T₂ come out right. The coupling comes out slightly
cautious: its interval is a little wider than it strictly needs to be. That is
the safe direction to be wrong in.

**What is missing is real data.** Four of the five figures I promised are made.
The fifth is the method applied to a measured spectrum.

**Essential: Figure 5 cannot be made without these**

1. **The spectra themselves.** Whichever molecules the archive has is fine, and
   nothing depends on my four. Setting up a new molecule is a table of couplings
   and a training run, so please pick by what is easiest to find and best
   documented, not by what I have already done. Raw FIDs if they exist. If only
   processed spectra survive, then I need to know what apodization, zero-filling
   and phasing were applied, because those change the line shape my model has to
   match.
2. **How the sample was prepared** for each dataset: thermal prepolarization
   with a sudden drop, an adiabatic ramp, or a pulse sequence. If pulses, the
   flip angles, the axes, and which nucleus. This sets the initial state, which
   is what fixes the *height* of each line. Line positions on their own are not
   enough to pin down the couplings.
3. **Acquisition parameters**: sample rate, acquisition time, number of points.

**Helpful: each of these removes a guess or adds a check**

4. **The resolution of the peak-fitting step**, meaning how close two lines can
   be before the software reports them as a single peak. This matters more than
   it sounds. It has to be a fixed property of the measurement, decided by the
   acquisition rather than by the fit. I originally let it follow the fitted
   T₂ instead, and that broke things badly: a 0.1 s change in T₂ merged three
   lines into one and moved the log-likelihood, which is the number measuring
   how well the model matches the data, by 5×10⁵. I would rather use the
   instrument's real number than guess.
5. **The residual field** during acquisition, size and direction, if it was
   measured or logged. If not, I fit it as an unknown, which works, since I
   deliberately allow it a wide range. But a measurement would tighten the
   result.
6. **The magnetometer response**: bandwidth, and any hardware filter in the
   chain. The 6th-order 500 Hz low-pass in the documentation is nearly flat in
   amplitude below 300 Hz, but it shifts the phase by about 67° between the J
   and 2J lines, so it affects the line heights even where it looks harmless.
7. **Any couplings already fitted** for these datasets. Not to assume them, but
   as a check: it would let me compare my method against the existing fitter on
   the same spectra, rather than only against simulation.

One thing is worth flagging about what the method returns, because it differs
from a standard fit. It reports which couplings the spectrum can pin down and
which it cannot, instead of returning a number for all of them. In a methyl or
methylene group, the proton-proton coupling inside the equivalent group moves no
line at all, so the spectrum carries no information about it, and my method
reports it as unmeasured. A least-squares fit on the same data returns an
arbitrary value there with an error bar attached, which looks like a result and
is not one. Worth knowing before anyone reads a J_HH off one of our outputs.

I have also now read the five closest papers properly, which narrowed what I
should be claiming. Cobas at Mestrelab does the analytic version of this for
AA′BB′ systems: he works out which combinations can be extracted, and settles
the rest by convention. His ABC/ABCD network already runs without a starting
guess. So neither of those is ours. What is left, and what the measurements
support, is a calibrated uncertainty from a single spectrum, in a regime where
the usual fix for the labelling ambiguity does not exist. Everyone else tells
the nuclei apart by chemical shift, and at zero field there is no chemical
shift. I would rather find that out now than from a referee.

Happy to do the digging myself if you can point me at where the archive lives
and who to ask about the acquisition details. I do not want to make this your
job. Even one or two well-documented datasets would be enough to make the last
figure real, and I would like to have them by [date] to keep the write-up on
schedule.

Thanks,
Wei
