# Draft: reply to Dima, with the test results and a decision

Send the text below the line. Nothing to fill in. **Attach `readout.pdf`**, in
which Section 7 and Figure 6 are the test described here.

This replaces the earlier unsent draft, which proposed three possible
destinations and asked him to pick one. That framing no longer fits, since it
asked him to invest in a direction you are stepping back from. The part of it
worth keeping, the observation about Wilzewski's error bars, is kept below in
short form and is written up in full in `literature_notes.md` and Section 8 of
the report, so nothing is lost if anyone picks this up later.

Two deliberate choices about how this is written, both worth checking against
what you actually want to say.

It **leads with the result, not the decision.** The misspecification test is
real work that found a real problem, and it earns the paragraphs that follow.
Opening with "I think I should stop" would make everything after it read as
justification.

It **separates "I do not yet understand this" from "this is not worth doing".**
The draft states the first plainly and gives the second as your judgement about
your own position rather than about the project. Those are different claims,
and a supervisor can respond to the first with help. If you want the stronger
version, that the project itself is not worth pursuing by anyone, say so and I
will change it. The numbers below do not settle that question either way.

The results quoted are from `python misspecification.py`, recorded in
`results.json` and plotted in Figure 6. The Wilzewski values are from Table I of
that paper.

---

Subject: Re: report, test results and where I have got to

Dear Dima,

Thank you for those questions. They were the right ones, and trying to answer
the second properly is what produced the result below.

You asked how the method would cope with real noise, drifts and nonlinearities.
It computes a consistency check that is supposed to collapse when the model
cannot explain the data, and I had been relying on that throughout. I had never
tested it, because every spectrum it had ever seen came from the same simulator
it fits with. So I built three spectra that no parameter value can reproduce.
It catches one of the three.

1. Field drift during acquisition of 1 nT. Noticed. The fit quality measure
goes from about 1 on clean data to 28, the consistency check drops to a third
of its clean value, and the reported coupling stays inside its error bar.

2. Two protons 2 Hz away from equivalent. Not noticed. The reported coupling
moves by 12 mHz, which is about nine times its own error bar, and every check
reads the same as it does on clean data.

3. Frequency axis off by one part in 10,000. Not noticed. The reported coupling
moves by 22 mHz, and again every check reads the same as on clean data.

Figure 6 of the attached report shows all three, and the section next to it
explains why two of them slip through.

One other thing worth passing on, whatever happens to this project. Reading
Wilzewski et al. properly, their benzene 1J_CH is 158.363(1) Hz against a
literature value of 158.354(1), and the paper notes that the disagreement
exceeds the quoted uncertainties. They also say they chose fit parameters that
do not co-vary, because the uncertainties come from the diagonal of the inverse
covariance matrix. That suggests the limit on this kind of measurement is not
statistical precision but what the error bars leave out, which seems worth
someone looking at.

Now the harder part. Working through this has made me realise the project has
got ahead of my own understanding. I set out to use a trained network to read
J-couplings out of a zero-field spectrum more precisely, and what I have learned
is that precision was never the difficulty. The difficulty is in
identifiability, systematics and model error, and those rest on foundations I do
not have yet. There are parts of what is now in the repository that I could not
derive or defend on my own, and I am not comfortable building further on ground
I cannot stand on. So I do not think this is the right thing for me to be
working on at the moment, at least not in this form.

That connects to the other thing I wanted to raise. I no longer think I should
put the group's real data through this. Recent discussion about AI in
mathematics research has made me more careful about it. These tools produce work
that looks rigorous and is sometimes confidently wrong, and the test above is a
small and concrete example: a coupling wrong by nine times its own error bar
with every check reporting clean, which I only found because you asked a
question that made me go looking. I would rather not take that risk with your
group's unpublished data.

Thank you for your support, and for your patience with a project that changed
shape several times. The questions in your last email did more for it than
anything I managed in the weeks before, and I would not have found that failure
without them. I am grateful for it.

Best,
Wei
