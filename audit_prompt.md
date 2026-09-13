# Prompt for an independent audit

Paste everything below the line into a fresh session. It is written to make
"stop" a genuinely available answer, and to stop the auditor simply agreeing
with the repository's own account of itself.

Two things to know before you read the verdict you get back.

The five papers behind `literature_notes.md` were uploaded as PDFs in the
session that wrote it. They are not in the repository, so a fresh session cannot
re-check those quotes without finding the papers again. The prompt says so and
asks for them to be marked unverified rather than assumed. If you still have the
PDFs, attach them and delete that caveat.

The prompt asks the auditor to judge the project, not you. Keep those separate
when you read the answer. A verdict of "this is worth someone continuing" is
compatible with "it is not the right thing for me right now", and the second is
your call, not the auditor's.

---

I want an independent audit of a research project to decide whether it is worth
continuing. Give me a verdict, not encouragement. "Stop" is an acceptable and
useful answer, and so is "continue, but only this part of it".

The repository is github.com/Wei-Chen-7/ZULF-Lab, branch
`claude/zulf-nmr-simulator-n8cjlh`. It is a zero- to ultralow-field NMR
simulator plus a simulation-based-inference stack that takes one spectrum and
returns a posterior over the scalar J-couplings, including a statement of which
couplings the spectrum cannot constrain at all.

Read `README.md`, `literature_notes.md` and `readout.tex` first. Treat all three
as claims made by an interested party rather than as evidence: they were written
by the same process that did the work. The git log is unusually detailed and is
a useful audit trail, including for mistakes that were found and corrected.

Then verify, in this order:

1. **Do the numbers come from the code?** Run `pytest -q` and
   `python results.py --check`. Pick three quantitative claims from the README
   and trace each one back to the script that computed it and the entry in
   `results.json`. Report anything you cannot trace.

2. **Does the central negative result hold?** Section 7 of `readout.tex` reports
   that the method's own misspecification check catches one of three tested
   failure modes, and that a frequency-axis scale error is exactly degenerate
   with rescaling (J, |B|, 1/T2). Re-derive that degeneracy yourself from the
   forward model instead of trusting the claim, and re-run
   `python misspecification.py`. If it holds, say plainly what it costs the
   project. If it does not hold, that is the most important thing you can tell
   me.

3. **Is the surviving claim actually novel?** `literature_notes.md` argues that
   what is left is a calibrated posterior from a single ZULF spectrum, in a
   regime where the standard remedy for the labelling degeneracy, ordering
   nuclei by chemical shift, does not exist. I do not have the five PDFs in this
   session, so treat every quotation in that file as unverified. Find the papers
   yourself and say whether the claim survives. The closest is Cobas,
   *Artificial Intelligence Chemistry* 4 (2026) 100127.

4. **What stands between here and a publishable result?** List the specific
   remaining work with a rough size for each item. Include repairing the two
   failure modes that currently go undetected, and say which of them you think
   is actually repairable.

5. **Could someone learn this?** Judge whether the repository is self-documenting
   enough that a motivated physics student could reconstruct and defend its key
   claims within a few months, or whether it depends on understanding that
   exists nowhere in it.

Then give me, in this order:

- a verdict: continue, continue in reduced scope, or stop, with your confidence
  and the main thing that confidence rests on;
- the strongest argument against your own verdict;
- if continue: the single first thing to do, and why that one before anything
  else;
- if stop: what should still be written up and published, if anything, and where.

Constraints. Do not take the repository's self-assessment at face value
anywhere. Where you cannot verify something, write "unverified" rather than
assuming it is fine. Judge the project on its own merits, and tell me
explicitly if your verdict depends on who is doing the work rather than on the
work itself. Do not soften the answer to be encouraging.
