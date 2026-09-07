# Draft: archived-data request

Short version to send. Everything in "Essential" is needed to produce the
fifth figure; everything in "Helpful" removes an assumption we would otherwise
have to make and state.

---

**Subject:** Archived ZULF spectra + pulse sequences for the J-coupling inference project

Dear [supervisor],

The simulation side of the J-coupling inference project is finished, and I am
at the point where the only thing left needs real spectra. Could I get access
to the group's archived ZULF data?

Where things stand: the forward model reproduces ref. [1]'s benzene-¹³C₁
multiplet to 7.2 mHz RMS from their published couplings, with nothing fitted.
Trained networks for formic acid, formaldehyde, glycine and methanol each reach
their information floor (1.0–2.2 mHz on ¹J_CH), agree with nested sampling on
the exact likelihood to 1.5%, and pass simulation-based calibration. The
comparison against a least-squares fit is done on simulated data. Four of the
five figures I promised are made; the fifth is the answer on real data.

**Essential**

1. Archived spectra for any of the molecules above, or others you would rather
   I start with. **Raw FIDs (time domain) if they exist** — if only processed
   spectra survive, then whatever apodization, zero-filling and phasing was
   applied, since those change the lineshape the model has to match.
2. The **preparation protocol** for each dataset: thermal prepolarization with a
   sudden drop, an adiabatic ramp, or a pulse sequence. If pulses, the flip
   angles, axes and which nucleus. This sets ρ(0), which fixes the line
   *amplitudes* — the positions alone will not pin the couplings.
3. **Acquisition parameters**: sample rate, acquisition time, number of points.

**Helpful, and each one removes an assumption**

4. The **frequency resolution of the peak-fitting step** — how close two lines
   can be before they are reported as one peak. This turned out to matter more
   than it sounds: I had let that threshold depend on the fitted T₂, which made
   the likelihood discontinuous and moved it by 5×10⁵ across a 0.1 s change in
   T₂. It has to be an instrument constant, so I need the instrument's number
   rather than a guess.
5. **Residual field** during acquisition, magnitude and direction, if it was
   logged or measured. Otherwise it stays a nuisance parameter, which is fine —
   the priors are deliberately wide — but a measurement would tighten things.
6. The **magnetometer response**: bandwidth, and any hardware filter in the
   chain. The 6th-order 500 Hz low-pass in the documentation is nearly flat in
   amplitude below 300 Hz but contributes about −67° of differential phase
   across the J/2J band, so it matters for amplitudes even where it looks
   harmless.
7. Any **previously fitted couplings** for these datasets, so the network's
   answers can be compared against the existing fitter on the same spectra
   rather than only against simulation.

One thing worth flagging: the method reports which couplings a spectrum can and
cannot constrain, rather than returning a number for all of them. For a methyl
or methylene group the proton–proton coupling inside the equivalent group moves
no line at all, so it comes back as its prior and is labelled as unmeasured. A
least-squares fit on the same data returns an arbitrary value there with an
error bar attached, which is the specific failure the approach is meant to
avoid.

Happy to work with whatever is easiest to pull out of the archive — even one or
two datasets would be enough to make the last figure real.

Thanks,
Wei
