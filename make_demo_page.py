#!/usr/bin/env python3
"""Build the one-page project summary, with the figures embedded.

A published page cannot load images from disk, so every figure goes in as a
base64 data URI and the result is one self-contained HTML file. Regenerate the
figures first if they are stale -- this script only packages them.

    python make_figure1.py && python make_figure2.py
    python final_check.py && python make_figure4.py
    python make_demo_page.py          # writes zulf_demo_page.html

The page is the artifact sent to the group: what is done, what each figure
shows, the negative results, and what the archive still has to supply.
"""

import base64
import pathlib

import results

FIGURES = {
    "fig1": "figure1_model_over_published.png",
    "fig2": "figure2_posterior.png",
    "fig3": "sbc_ranks.png",
    "fig4": "figure4_comparison.png",
}


LABELS = {"formic_acid": "[<sup>13</sup>C]-formic acid",
          "formaldehyde": "[<sup>13</sup>C]-formaldehyde",
          "glycine": "[<sup>13</sup>C]-glycine",
          "methanol": "[<sup>13</sup>C]-methanol"}
ORDER = ["formic_acid", "formaldehyde", "glycine", "methanol"]


def data_uri(path):
    return "data:image/png;base64," + base64.b64encode(
        pathlib.Path(path).read_bytes()).decode()


def library_rows(lib):
    """The results table, rendered from results.json rather than typed."""
    out = []
    for name in ORDER:
        m = lib.get(name)
        if not m:
            continue
        meas = ", ".join(f"J<sub>{p.split('_')[1]}</sub>" for p in m["measured"])
        flat = ", ".join(f"J<sub>{p.split('_')[1]}</sub>" for p in m["flat"])
        flat_cell = (f'<td class="flat">{flat}</td>' if flat
                     else '<td class="none">&mdash;</td>')
        out.append(
            f'        <tr><td>{LABELS.get(name, name)}</td>'
            f'<td>{m["n_spins"]}</td><td>{meas}</td>\n'
            f'            <td>{m["width_mHz"]:.2f} <span class="unit">mHz</span></td>'
            f'<td>{m["floor_mHz"]:.3f}</td><td>{m["ratio"]:.2f}</td>\n'
            f'            {flat_cell}<td>{100 * m["efficiency"]:.1f}%</td></tr>')
    return "\n".join(out)

HTML = r"""<title>Zero-Field J Readout</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&display=swap">
<style>
:root{
  --ground:#f2f5f5;
  --surface:#ffffff;
  --surface-2:#e8edee;
  --ink:#14232a;
  --ink-2:#3d5158;
  --ink-3:#6b8189;
  --rule:#d3dcde;
  --rule-2:#bcc9cc;
  --teal:#1f7d72;
  --teal-soft:#d3e8e4;
  --deep:#264653;
  --coral:#c2543a;
  --coral-soft:#f6e0d9;
  --plate:#ffffff;
  --shadow:0 1px 2px rgba(20,35,42,.06),0 8px 24px -16px rgba(20,35,42,.28);
  --serif:"Source Serif 4",Georgia,"Times New Roman",serif;
  --sans:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,"SF Mono",Menlo,monospace;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --ground:#0e171c;
    --surface:#152329;
    --surface-2:#1c2d34;
    --ink:#e4eded;
    --ink-2:#b0c2c6;
    --ink-3:#7e969c;
    --rule:#26383f;
    --rule-2:#33474f;
    --teal:#5cc4b4;
    --teal-soft:#183833;
    --deep:#8fb3bf;
    --coral:#ea8567;
    --coral-soft:#3a2119;
    --plate:#f4f6f6;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 28px -18px rgba(0,0,0,.8);
  }
}
:root[data-theme="dark"]{
  --ground:#0e171c;
  --surface:#152329;
  --surface-2:#1c2d34;
  --ink:#e4eded;
  --ink-2:#b0c2c6;
  --ink-3:#7e969c;
  --rule:#26383f;
  --rule-2:#33474f;
  --teal:#5cc4b4;
  --teal-soft:#183833;
  --deep:#8fb3bf;
  --coral:#ea8567;
  --coral-soft:#3a2119;
  --plate:#f4f6f6;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 28px -18px rgba(0,0,0,.8);
}

*{box-sizing:border-box}
body{
  background:var(--ground);
  color:var(--ink);
  font:400 16px/1.62 var(--sans);
  -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1080px;margin:0 auto;padding:0 28px 96px}
.col{max-width:68ch}
h1,h2,h3{font-family:var(--serif);text-wrap:balance;margin:0}
h1{font-size:clamp(2.1rem,4.4vw,3.1rem);font-weight:700;line-height:1.08;letter-spacing:-.015em}
h2{font-size:clamp(1.45rem,2.6vw,1.85rem);font-weight:600;line-height:1.2}
h3{font-size:1.06rem;font-weight:600;line-height:1.3}
p{margin:0}
a{color:var(--teal)}
strong{font-weight:600}
.mono{font-family:var(--mono);font-variant-numeric:tabular-nums}
.eyebrow{
  font:500 .72rem/1 var(--mono);
  letter-spacing:.13em;text-transform:uppercase;color:var(--ink-3);
}

/* ---------- masthead ---------- */
header{
  border-bottom:1px solid var(--rule);
  padding:52px 0 34px;
  display:flex;flex-direction:column;gap:20px;
}
.sub{font-size:1.1rem;color:var(--ink-2);max-width:60ch}
.byline{font:400 .84rem/1.5 var(--mono);color:var(--ink-3)}
.chips{display:flex;flex-wrap:wrap;gap:8px}
.chip{
  font:500 .76rem/1 var(--mono);
  padding:7px 11px;border-radius:3px;border:1px solid var(--rule-2);
  color:var(--ink-2);background:var(--surface);
}
.chip.on{border-color:var(--teal);color:var(--teal);background:var(--teal-soft)}
.chip.off{border-color:var(--coral);color:var(--coral);background:var(--coral-soft)}

/* ---------- sections ---------- */
section{padding:52px 0;border-bottom:1px solid var(--rule)}
section:last-of-type{border-bottom:0}
.head{display:flex;flex-direction:column;gap:9px;margin-bottom:26px}
.stack{display:flex;flex-direction:column;gap:16px}
.lede{font-size:1.04rem;color:var(--ink-2)}

/* ---------- data table ---------- */
.scroll{overflow-x:auto;margin:26px 0 0}
table{border-collapse:collapse;width:100%;min-width:620px;font-size:.9rem}
caption{
  text-align:left;font:500 .74rem/1.5 var(--mono);letter-spacing:.09em;
  text-transform:uppercase;color:var(--ink-3);padding-bottom:11px;
}
th,td{padding:10px 14px;text-align:right;border-bottom:1px solid var(--rule)}
th:first-child,td:first-child{text-align:left}
thead th{
  font:500 .74rem/1.4 var(--mono);color:var(--ink-3);
  border-bottom:1px solid var(--rule-2);white-space:nowrap;
}
tbody td{font-family:var(--mono);font-variant-numeric:tabular-nums;color:var(--ink)}
tbody td:first-child{font-family:var(--sans);font-weight:500}
tbody tr:last-child td{border-bottom:0}
td .unit{color:var(--ink-3);font-size:.82em}
.flat{color:var(--coral)}
.none{color:var(--ink-3)}

/* ---------- shrinkage bars ---------- */
.bars{display:flex;flex-direction:column;gap:3px;margin-top:26px}
.barrow{
  display:grid;grid-template-columns:8.5rem 1fr 5.4rem;gap:14px;align-items:center;
  padding:7px 0;border-bottom:1px solid var(--rule);
}
.barrow:last-child{border-bottom:0}
.barrow .who{font:500 .84rem/1.3 var(--sans)}
.barrow .who em{display:block;font:400 .72rem/1.3 var(--mono);color:var(--ink-3);font-style:normal}
.track{height:13px;background:var(--surface-2);border-radius:2px;overflow:hidden}
.fill{height:100%;background:var(--teal);border-radius:2px 0 0 2px}
.fill.thin{background:var(--coral)}
.barrow .val{
  font:500 .84rem/1 var(--mono);text-align:right;font-variant-numeric:tabular-nums;
}
.barnote{font:400 .78rem/1.5 var(--mono);color:var(--ink-3);margin-top:12px}

/* ---------- figure spine ---------- */
.figs{display:flex;flex-direction:column;gap:34px;margin-top:8px}
figure{margin:0;display:flex;flex-direction:column;gap:13px}
.figtag{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap}
.num{
  font:600 .74rem/1 var(--mono);letter-spacing:.1em;text-transform:uppercase;
  color:var(--teal);border:1px solid var(--teal);border-radius:2px;padding:6px 9px;
  white-space:nowrap;
}
.num.waiting{color:var(--coral);border-color:var(--coral)}
.plate{
  background:var(--plate);border:1px solid var(--rule-2);border-radius:4px;
  padding:12px;box-shadow:var(--shadow);overflow-x:auto;
}
.plate img{display:block;width:100%;min-width:560px;height:auto}
figcaption{font-size:.92rem;color:var(--ink-2);max-width:74ch}
figcaption b{color:var(--ink);font-weight:600}
.empty{
  border:1px dashed var(--rule-2);border-radius:4px;padding:38px 26px;
  background:var(--surface);display:flex;flex-direction:column;gap:9px;
}
.empty .big{font-family:var(--serif);font-size:1.16rem;font-weight:600;color:var(--ink-2)}

/* ---------- callouts ---------- */
.notes{display:grid;grid-template-columns:repeat(auto-fit,minmax(268px,1fr));gap:18px;margin-top:24px}
.note{
  background:var(--surface);border:1px solid var(--rule);border-radius:4px;
  padding:20px;display:flex;flex-direction:column;gap:9px;
}
.note .eyebrow{color:var(--teal)}
.note.warn .eyebrow{color:var(--coral)}
.note p{font-size:.92rem;color:var(--ink-2)}
.note h3{font-size:1rem}
.note .mono{color:var(--ink);font-size:.86rem}

/* ---------- pull quote / key figure ---------- */
.pull{
  border-left:3px solid var(--coral);padding:4px 0 4px 20px;margin:26px 0 0;
  display:flex;flex-direction:column;gap:7px;
}
.pull .big{font-family:var(--serif);font-size:1.28rem;font-weight:600;line-height:1.32}
.pull p{font-size:.94rem;color:var(--ink-2)}

/* ---------- ask list ---------- */
ol.ask{margin:22px 0 0;padding:0;list-style:none;counter-reset:a;display:flex;flex-direction:column;gap:0}
ol.ask li{
  counter-increment:a;display:grid;grid-template-columns:2.1rem 1fr;gap:14px;
  padding:15px 0;border-bottom:1px solid var(--rule);
}
ol.ask li:last-child{border-bottom:0}
ol.ask li::before{
  content:counter(a,decimal-leading-zero);
  font:500 .78rem/1.7 var(--mono);color:var(--ink-3);
}
ol.ask h3{margin-bottom:4px}
ol.ask p{font-size:.92rem;color:var(--ink-2)}
footer{padding:34px 0 0;font:400 .82rem/1.6 var(--mono);color:var(--ink-3)}
@media (max-width:640px){
  .wrap{padding:0 18px 64px}
  .barrow{grid-template-columns:6.6rem 1fr 4.2rem;gap:10px}
}
</style>

<div class="wrap">

<header>
  <div class="eyebrow">Simulation-based inference &middot; zero- to ultralow-field NMR</div>
  <h1>Reading J&#8209;couplings out of a zero&#8209;field spectrum</h1>
  <p class="sub">A trained network that returns a posterior over scalar couplings with no starting guess &mdash; and states which couplings the spectrum cannot constrain instead of quoting a number for them.</p>
  <p class="byline">Wei Chen &middot; Wabash College / Helmholtz&#8209;Institut Mainz &middot; status as of the last simulation run</p>
  <div class="chips">
    <span class="chip on">Forward model validated &middot; 7.2 mHz RMS</span>
    <span class="chip on">4 trained networks, all at their information floor</span>
    <span class="chip on">Calibrated (SBC) &middot; exact-likelihood reference</span>
    <span class="chip on">203 tests</span>
    <span class="chip off">Figure 5 &mdash; needs archived spectra</span>
  </div>
</header>

<section>
  <div class="head">
    <div class="eyebrow">The result</div>
    <h2>Every network reaches the precision the noise allows</h2>
  </div>
  <div class="col stack">
    <p class="lede">Four molecules, each network trained once and cached. The <b>floor</b> column is the narrowest 95% interval the noise model permits, computed from the Fisher information of the simulator. Every measured coupling sits on it.</p>
  </div>

  <div class="scroll">
    <table>
      <caption>Reweighted posteriors on <span class="mono">&sigma;<sub>f</sub></span> = 1 mHz
      peak-position noise &middot; efficiency = effective sample size / draws, after reweighting</caption>
      <thead>
        <tr>
          <th>Molecule</th><th>Spins</th><th>Measured</th>
          <th>95% width</th><th>Floor</th><th>Ratio</th>
          <th>Unmeasurable</th><th>Efficiency</th>
        </tr>
      </thead>
      <tbody>
__LIBROWS__
      </tbody>
    </table>
  </div>

  <div class="col stack" style="margin-top:26px">
    <p>Efficiency is the fraction of the network&rsquo;s draws that survive reweighting, and it is
    also the misspecification detector: on real data it is meant to collapse when the model is
    wrong. It varies by ~30&times; between observations of the <em>same</em> molecule with the
    <em>same</em> network, which is why the project&rsquo;s pass/fail threshold on it should be read
    over several spectra rather than one.</p>
    <p>These are not the best figures available, and it is worth saying so: a wider flow reaches
    <b>37%</b> against the 24% above on formic acid, at the same reweighted precision. Every result
    here uses the narrower one, because the precision is set by the exact likelihood and only the
    efficiency moves &mdash; but there is roughly 1.5&times; of headroom on the detector before any
    real data arrives.</p>
    <p>Precision <em>improves</em> across this series, which is not noise. The floor is not
    <span class="mono">&sigma;<sub>f</sub>/&radic;n</span>: XA<sub>2</sub> puts its line at
    <span class="mono">3/2&nbsp;J</span> and XA<sub>3</sub> at <span class="mono">J</span> and
    <span class="mono">2J</span>, so the lines move <em>faster</em> than J and a given peak-position
    error buys a tighter coupling. Finding that corrected a constant in the code that had only ever
    been right for two spins.</p>
  </div>
</section>

<section>
  <div class="head">
    <div class="eyebrow">How it works</div>
    <h2>Spectrum to posterior, in five steps</h2>
  </div>
  <div class="col stack">
    <p class="lede">Nothing here is exotic. The choices that matter are what the network is shown
    and what it is asked to produce.</p>
  </div>
  <ol class="ask">
    <li><div><h3>Peaks, not the spectrum</h3>
      <p>The observation is a list of line positions, amplitudes and a width. Handing a flow the raw
      spectrum instead is the known failure mode: a 10 mHz linewidth across a 500 Hz band needs
      ~200&#8239;000 bins, and resolving one part in 10<sup>6</sup> of that vector is not something a
      density estimator does.</p></div></li>
    <li><div><h3>A fixed-length summary</h3>
      <p>A low-frequency group (Larmor precession, near 0 Hz) and the J multiplet, with multiplet
      positions written as offsets from the prior centre &mdash; so the network sees numbers of order
      0.1 Hz rather than 200 Hz.</p></div></li>
    <li><div><h3>A prior that is not flat</h3>
      <p>DFT and ML predictors already fix <sup>1</sup>J<sub>CH</sub> to about 1 Hz. That is useless
      as a measurement and excellent as a prior, so the coupling prior is a few Hz wide rather than
      the full band. The nuisances &mdash; residual field magnitude and angle, T<sub>2</sub> &mdash;
      get deliberately wide priors: a network trained on a messier world than the real one transfers,
      and one trained on a tidy world does not.</p></div></li>
    <li><div><h3>One round of amortized NPE</h3>
      <p>Sequential methods tune the proposal to a single observation and lose amortization. A single
      round keeps it, at the cost of more simulations &mdash; and simulations are cheap here, 0.2 ms
      for a four-spin line list, because one diagonalization gives exact frequencies and amplitudes
      with no FID to build and transform.</p></div></li>
    <li><div><h3>Reweighting against the exact likelihood</h3>
      <p>The network&rsquo;s samples are reweighted by likelihood &times; prior / network density. This
      is what makes the reported posterior exact rather than approximate, and it is why the widths in
      the table above land on the information floor instead of merely near it.</p></div></li>
  </ol>

  <div class="pull">
    <div class="big">The forward model is deterministic with additive Gaussian noise, so the
      likelihood is available in closed form. The case for a trained network here is not
      intractability.</div>
    <p>It is that the network searches globally and returns <em>every</em> solution consistent with
    the spectrum rather than the one nearest a starting guess; that having the likelihood is exactly
    what lets its samples be reweighted into an exact answer; and that the efficiency of that
    reweighting doubles as a misspecification detector &mdash; it is meant to collapse when the model
    is wrong about real data, which is the check no simulation study can run on itself.</p>
  </div>
</section>

<section>
  <div class="head">
    <div class="eyebrow">Why this and not a least-squares fit</div>
    <h2>The spectrum cannot see every coupling, and the report should say so</h2>
  </div>
  <div class="col stack">
    <p class="lede">A methyl or methylene group has a proton&ndash;proton coupling inside it that moves
    no line at all &mdash; the <span class="mono">&Delta;I<sub>A</sub> = 0</span> selection rule.
    Sweeping J<sub>HH</sub> across its entire prior changes the summary vector by less than
    10<sup>&minus;12</sup>. Parameterizing by symmetry-distinct classes and reporting the
    <em>shrinkage</em> makes that visible instead of burying it.</p>
  </div>

  <div class="bars">
    <div class="barrow">
      <div class="who">J<sub>CH</sub><em>formic acid</em></div>
      <div class="track"><div class="fill" style="width:100%"></div></div>
      <div class="val">1.000</div>
    </div>
    <div class="barrow">
      <div class="who">J<sub>CH</sub><em>formaldehyde</em></div>
      <div class="track"><div class="fill" style="width:100%"></div></div>
      <div class="val">1.000</div>
    </div>
    <div class="barrow">
      <div class="who">J<sub>HH</sub><em>formaldehyde</em></div>
      <div class="track"><div class="fill thin" style="width:0.5%"></div></div>
      <div class="val">&minus;0.001</div>
    </div>
    <div class="barrow">
      <div class="who">J<sub>CH</sub><em>methanol</em></div>
      <div class="track"><div class="fill" style="width:100%"></div></div>
      <div class="val">1.000</div>
    </div>
    <div class="barrow">
      <div class="who">J<sub>HH</sub><em>glycine</em></div>
      <div class="track"><div class="fill thin" style="width:0.5%"></div></div>
      <div class="val">&minus;0.003</div>
    </div>
    <div class="barrow">
      <div class="who">J<sub>HH</sub><em>methanol</em></div>
      <div class="track"><div class="fill thin" style="width:1.1%"></div></div>
      <div class="val">0.011</div>
    </div>
  </div>
  <p class="barnote">shrinkage = 1 &minus; posterior width / prior width &nbsp;&middot;&nbsp;
  1.000 = pinned down. The J<sub>HH</sub> values scatter either side of zero
  (&minus;0.003 to +0.011) because a posterior identical to the prior estimates a width that
  fluctuates around it &mdash; a slightly negative shrinkage is what no information looks like.</p>

  <div class="pull">
    <div class="big">Given the same methanol spectrum, a least-squares fit reports
      J<sub>HH</sub> = &minus;3.5 Hz with a 95% interval of 1&#8239;788&#8239;122 Hz &mdash; on a 30 Hz prior.</div>
    <p>It does not fail. The Hessian is singular, inverting it returns a number, and the number is
    printed as an error bar. Across 60 random starts the fitted value fills the whole prior with a
    spread of 11 Hz &mdash; wider than the prior&rsquo;s own 8.7 Hz &mdash; because the simplex drifts
    along the flat direction until the box stops it. From the same 60 fits J<sub>CH</sub> lands within
    6&times;10<sup>&minus;5</sup> Hz of the truth every time.</p>
  </div>
</section>

<section>
  <div class="head">
    <div class="eyebrow">The five figures the proposal commits to</div>
    <h2>Four are made; the fifth is the one that needs the lab</h2>
  </div>

  <div class="figs">

    <figure>
      <div class="figtag"><span class="num">Figure 1</span>
        <h3>The forward model over a published spectrum</h3></div>
      <div class="plate"><img src="__FIG1__" alt="Forward model line positions overlaid on the measured benzene-13C1 zero-field spectrum from the literature, with residuals of a few millihertz."></div>
      <figcaption>The measured benzene-<sup>13</sup>C<sub>1</sub> multiplet, recovered exactly from the
      published PDF &mdash; the plot is vector art, so the trace is an 8515-point polyline in the content
      stream, not a screenshot. The overlay uses the paper&rsquo;s own fitted couplings and preparation;
      nothing is fitted. All 13 strong peaks are offset the same way, by &minus;22.6 mHz, and
      <b>the scatter about that offset is 7.2 mHz</b>. The uniformity is the point: a residual field
      would shift lines differentially, and regressing offset against frequency gives no trend
      (+0.26 &plusmn; 0.84 mHz/Hz, p = 0.76). It is the axis anchor, read off a left-anchored text
      label &mdash; 0.58 pt out of 25.5 pt/Hz accounts for the whole of it.</figcaption>
    </figure>

    <figure>
      <div class="figtag"><span class="num">Figure 2</span>
        <h3>The posterior, and its spread</h3></div>
      <div class="plate"><img src="__FIG2__" alt="Corner plot of the four-parameter posterior for formic acid with an inset bar chart comparing prior, raw network proposal, reweighted posterior and information floor."></div>
      <figcaption>Prior 5700 mHz &rarr; raw network proposal &rarr; reweighted 2.289 &rarr; floor
      2.263. <b>The posterior lands at 1.01&times; the information floor</b>, at 24.2% sampling
      efficiency (effective sample size 4838 out of 20&#8239;000). Drawn on zoomed axes with the
      prior width written on each panel: on the prior&rsquo;s own axes the J posterior is a vertical line
      1/2500th of the frame wide, which looks impressive and hides the nuisance correlations a referee
      would actually interrogate. They come back below 0.02.</figcaption>
    </figure>

    <figure>
      <div class="figtag"><span class="num">Figure 3</span>
        <h3>The calibration check</h3></div>
      <div class="plate"><img src="__FIG3__" alt="Simulation-based calibration rank histograms for the four parameters, with the uniform expectation band."></div>
      <figcaption>Simulation-based calibration over 300 trials. The failure that matters is ranks piling
      at the <em>edges</em>, meaning the posterior is too narrow and the network is confidently wrong.
      <b>The opposite happens:</b> on J the outer 20% holds 0.057 of the mass against 0.20 expected &mdash;
      too wide, the safe direction, and precisely why importance reweighting has good coverage. The three
      nuisance parameters come back calibrated.</figcaption>
    </figure>

    <figure>
      <div class="figtag"><span class="num">Figure 4</span>
        <h3>Against exact sampling, and against the standard method</h3></div>
      <div class="plate"><img src="__FIG4__" alt="Three panels: overlapping posteriors from nested sampling, the network and a local fit; a scatter of fitted versus starting values for methanol; and a bimodal angle posterior with the local fit's two answers."></div>
      <figcaption>Three panels because the comparison has three answers, and showing only the flattering
      one would be dishonest. <b>(a)</b> On formic acid all three agree to about 2% &mdash; nested
      sampling 2.24 mHz, network 2.29, local curvature 2.26 &mdash; and 200 of 200 random starts find
      the same minimum, agreeing on J to 2&times;10<sup>&minus;7</sup> Hz, so the
      local fit is not the weak link here. <b>(b)</b> On methanol the fit returns an arbitrary number for
      the flat direction. <b>(c)</b> &theta;<sub>B</sub> and 180&deg;&minus;&theta;<sub>B</sub> give
      log-likelihoods identical to 5&times;10<sup>&minus;13</sup>; the network holds both modes at
      51/49, the fit returns 54.4 &plusmn; 2.0&deg; or 125.6 &plusmn; 2.0&deg; depending only on where it
      started.</figcaption>
    </figure>

    <figure>
      <div class="figtag"><span class="num waiting">Figure 5</span>
        <h3>The answer on real data</h3></div>
      <div class="empty">
        <div class="big">Waiting on the group&rsquo;s archived spectra.</div>
        <p style="color:var(--ink-2);font-size:.94rem">Everything above runs on simulated data and is
        reproducible from the repository. This is the only remaining external input &mdash; the archived
        spectra, the pulse sequences that produced them, and the acquisition resolution.</p>
      </div>
    </figure>

  </div>
</section>

<section>
  <div class="head">
    <div class="eyebrow">Things that did not go the way I expected</div>
    <h2>Negative results, kept rather than dropped</h2>
  </div>
  <div class="col stack">
    <p class="lede">A method paper is only worth anything if the checks that failed are in it too.</p>
  </div>

  <div class="notes">
    <div class="note warn">
      <div class="eyebrow">Refuted</div>
      <h3>Resolution cliffs do not drive the efficiency spread</h3>
      <p>A peak list is a discontinuous observable, so the obvious explanation for the ~31&times; spread in
      sampling efficiency was proximity to a jump. It is wrong: Spearman
      <span class="mono">&rho; = &minus;0.02</span> (p = 0.87) over 60 observations. Screening seven
      candidates, the only survivor of a Bonferroni correction is the true J<sub>CH</sub> itself
      (<span class="mono">&rho; = &minus;0.41</span>, p<sub>adj</sub> = 0.009) &mdash; a property of the
      flow, not the physics.</p>
    </div>
    <div class="note warn">
      <div class="eyebrow">Does not win</div>
      <h3>Amortization loses on raw speed</h3>
      <p>Training costs <span class="mono">1973 s</span> once, then <span class="mono">4.98 s</span> per
      spectrum &mdash; against <span class="mono">1.56 s</span> for a single well-started local fit. The
      network only wins against the 200-start protocol you would need to be sure, where break-even is
      <b>6 spectra</b>. The honest argument for it is global search and multimodality, not throughput.</p>
    </div>
    <div class="note warn">
      <div class="eyebrow">My own bug</div>
      <h3>A discontinuity in T<sub>2</sub> that should never have been there</h3>
      <p>The merge threshold was the model&rsquo;s own linewidth, so it moved with the fitted T<sub>2</sub>.
      The multiplet gap is 26.644 mHz and the linewidth at T<sub>2</sub> = 12 s is 26.526, so a
      <b>0.1 s change in T<sub>2</sub> merged three lines into one and moved the log-likelihood by
      5&times;10<sup>5</sup></b>. It has to be an instrument constant &mdash; which is why the archive
      request asks for the real one.</p>
    </div>
  </div>
</section>

<section>
  <div class="head">
    <div class="eyebrow">Checked against the primary literature</div>
    <h2>Three corrections to the physics in my own proposal</h2>
  </div>
  <div class="col stack">
    <p>Each was confirmed against published measurements, not just re-derived.</p>
  </div>
  <div class="notes">
    <div class="note">
      <div class="eyebrow">Corrected</div>
      <h3>XA<sub>2</sub> gives one line, not two</h3>
      <p>I had written that an XA<sub>2</sub> system produces lines at J and 2J. It produces a single line
      at <span class="mono">3/2&nbsp;J</span> &mdash; confirmed analytically three ways and against the
      measured formaldehyde value of 245.85 Hz.</p>
    </div>
    <div class="note">
      <div class="eyebrow">Corrected</div>
      <h3>A transverse field gives a doublet, not a triplet</h3>
      <p>Two lines about J, split by the <em>sum</em> of the Larmor frequencies, and nothing at J
      itself. The centre line&rsquo;s intensity falls as
      <span class="mono">cos&sup2;&theta;</span> and vanishes outright at 90&deg;, where
      <span class="mono">&lang;T&#8320;|M&#770;<sub>z</sub>|S&rang; = 0</span> forbids it for every
      preparation. The remaining line sits at the <em>mean</em> of the Larmor frequencies &mdash;
      down near 0 Hz, not inside the doublet.</p>
    </div>
    <div class="note">
      <div class="eyebrow">New</div>
      <h3>A third exact degeneracy</h3>
      <p><span class="mono">&theta;<sub>B</sub></span> and
      <span class="mono">180&deg; &minus; &theta;<sub>B</sub></span> give bit-identical spectra, so only
      <span class="mono">|cos&nbsp;&theta;<sub>B</sub>|</span> is identifiable. It sits alongside the
      global sign flip and the equal-&gamma; permutations, and is broken by convention in the prior.</p>
    </div>
  </div>
</section>

<section>
  <div class="head">
    <div class="eyebrow">The ask</div>
    <h2>What the archive needs to supply</h2>
  </div>
  <div class="col stack">
    <p class="lede">In rough order of how much each one unblocks. The first three are needed to make
    Figure 5 at all; the rest each remove an assumption that would otherwise have to be stated and
    defended.</p>
  </div>
  <ol class="ask">
    <li><div><h3>The spectra &mdash; raw FIDs if they exist</h3>
      <p>If only processed spectra survive, then the apodization, zero-filling and phasing that were
      applied, since those change the lineshape the model has to match.</p></div></li>
    <li><div><h3>The preparation protocol for each dataset</h3>
      <p>Thermal prepolarization with a sudden drop, an adiabatic ramp, or a pulse sequence &mdash; and if
      pulses, the flip angles, axes and nucleus. This sets &rho;(0), which fixes the line
      <em>amplitudes</em>; positions alone will not pin the couplings.</p></div></li>
    <li><div><h3>Acquisition parameters</h3>
      <p>Sample rate, acquisition time, number of points.</p></div></li>
    <li><div><h3>The resolution of the peak-fitting step</h3>
      <p>How close two lines can be before they are reported as one peak. This is the instrument constant
      the T<sub>2</sub> bug above turned on, so a measured number beats a guess.</p></div></li>
    <li><div><h3>Residual field during acquisition, if logged</h3>
      <p>Magnitude and direction. Otherwise it stays a nuisance parameter with a deliberately wide prior,
      which works &mdash; but a measurement would tighten it.</p></div></li>
    <li><div><h3>The magnetometer response</h3>
      <p>Bandwidth and any hardware filter. The documented 6th-order 500 Hz low-pass is nearly flat in
      amplitude below 300 Hz but contributes about &minus;67&deg; of differential phase across the J/2J
      band, so it matters for amplitudes even where it looks harmless.</p></div></li>
    <li><div><h3>Any previously fitted couplings</h3>
      <p>So the network can be compared against the existing fitter on the same spectra, rather than only
      against simulation.</p></div></li>
  </ol>
</section>

<footer>
  Reproducible from the repository: <span class="mono">pytest -q</span> &rarr; 203 passing &middot;
  <span class="mono">python train_library.py</span> trains and caches the four networks &middot;
  <span class="mono">python local_baseline.py</span> runs the three-case comparison &middot;
  figures from <span class="mono">make_figure1.py</span>, <span class="mono">make_figure2.py</span>,
  <span class="mono">final_check.py</span>, <span class="mono">make_figure4.py</span>.
</footer>

</div>
"""

def main():
    data = results.load()
    lib = data.get("library")
    if not lib:
        raise SystemExit("results.json has no library section; run "
                         "train_library.py first")
    html = HTML.replace("__LIBROWS__", library_rows(lib))
    for key, name in FIGURES.items():
        if not pathlib.Path(name).exists():
            raise SystemExit(f"missing {name}; regenerate the figures first")
        html = html.replace(f"__{key.upper()}__", data_uri(name))
    out = pathlib.Path("zulf_demo_page.html")
    out.write_text(html)
    print(f"wrote {out}  ({len(html) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
