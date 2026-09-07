# Draft: archived-data request

Edit and send the text below the line. Two things to fill in: the supervisor's
name, and the deadline in the closing paragraph if you want one.

The split matters. **Essential** is what Figure 5 cannot be made without.
**Helpful** is what stops us having to assume something and defend the
assumption in the paper. Ask for both, but make clear that the first three are
the blocking ones — a request that looks like seven demands is easier to
postpone than one that looks like three.

---

**Subject:** Archived ZULF spectra + pulse sequences for the J-coupling inference project

Dear [supervisor],

The simulation side of the J-coupling inference project is finished, and the
only thing left needs real spectra. Could I get access to the group's archived
ZULF data?

Where things stand: with nothing fitted, the forward model reproduces
Wilzewski et al.'s published benzene-¹³C₁ multiplet from their own couplings to
a scatter of 7.2 mHz — the residual is a single uniform offset that traces to
the axis anchor I read off their figure, not to the physics. Trained networks
for formic acid, formaldehyde, glycine and methanol each reach their
information floor on ¹J_CH (1.0–2.3 mHz), agree with nested sampling on the
exact likelihood to about 2%, and pass simulation-based calibration. The
comparison against a least-squares fit is done on simulated data. Four of the
five figures I promised are made; the fifth is the answer on real data.

**Essential — Figure 5 cannot be made without these**

1. **The spectra themselves.** Whichever molecules the archive actually has is
   fine; nothing depends on my four. Setting up a new molecule is a table of
   couplings and a training run, so please pick by what is easiest to find and
   best documented, not by what I have already done. Raw FIDs (time domain) if
   they exist — if only processed spectra survive, then whatever apodization,
   zero-filling and phasing was applied, since those change the lineshape the
   model has to match.
2. **The preparation protocol** for each dataset: thermal prepolarization with a
   sudden drop, an adiabatic ramp, or a pulse sequence. If pulses, the flip
   angles, axes and which nucleus. This sets ρ(0), which fixes the line
   *amplitudes* — line positions alone will not pin the couplings.
3. **Acquisition parameters**: sample rate, acquisition time, number of points.

**Helpful — each of these removes an assumption or adds a check**

4. **The resolution of the peak-fitting step** — how close two lines can be
   before they are reported as one peak. This matters more than it sounds. It
   has to be a fixed property of the acquisition: when I let it depend on the
   fitted T₂ instead, the likelihood went discontinuous and the log-likelihood
   jumped by 5×10⁵ across a 0.1 s change in T₂. I would rather have the
   instrument's number than a guess.
5. **Residual field** during acquisition, magnitude and direction, if it was
   logged or measured. Otherwise it stays a nuisance parameter, which works —
   the priors are deliberately wide — but a measurement would tighten things.
6. **The magnetometer response**: bandwidth, and any hardware filter in the
   chain. The 6th-order 500 Hz low-pass in the documentation is nearly flat in
   amplitude below 300 Hz but contributes about −67° of differential phase
   across the J/2J band, so it matters for amplitudes even where it looks
   harmless.
7. **Any previously fitted couplings** for these datasets. Not an assumption —
   a check: it would let me compare the network against the existing fitter on
   the same spectra rather than only against simulation.

One thing worth flagging about what the method returns. It reports which
couplings a spectrum can and cannot constrain, instead of giving a number for
all of them. In a methyl or methylene group the proton–proton coupling inside
the equivalent group moves no line at all, so it comes back as its prior and is
labelled unmeasured. A least-squares fit on the same data returns an arbitrary
value there with an error bar attached — that is the specific failure this is
meant to avoid, and it is worth knowing before anyone reads a J_HH off one of
our outputs.

Happy to do the digging myself if you can point me at where the archive lives
and who to ask about the acquisition details — I do not want to make this your
job. Even one or two well-documented datasets would be enough to make the last
figure real, and I would like to have it in hand by [date] to keep the write-up
on schedule.

Thanks,
Wei
