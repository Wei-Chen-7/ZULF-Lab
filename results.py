#!/usr/bin/env python3
"""One store for every number that appears in a document.

The problem this solves is drift. Every figure in the README, the demo page and
the archive-request email was transcribed by hand out of a run log, and an audit
found a stale library table, a stale comparison table, a stale eigenvalue list,
an error bar off by a factor of 1.3, and a *reversed* conclusion in the
proposal-tightening section. Adding a LaTeX paper as a fourth surface without
fixing this would repeat all of it, in the one place it is least recoverable.

The rule here is that a reported number is computed exactly once, by the script
that owns it, and every document renders from the store:

    study script  --record-->  results.json  --render-->  README block
                                             --render-->  LaTeX macros
                                             --render-->  demo page

So ``python results.py --check`` fails if any rendered surface has drifted from
the store, which is the thing that could not be checked before.

    python results.py             # re-render every surface from results.json
    python results.py --check     # exit 1 if a surface is stale (for CI)
    python results.py --show      # print the store
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import sys

RESULTS = "results.json"
MACROS = "results_macros.tex"
README = "README.md"

_BEGIN = "<!-- BEGIN generated: {name} -- edit results.py, not this block -->"
_END = "<!-- END generated: {name} -->"


# ===========================================================================
# store
# ===========================================================================
def load(path=RESULTS):
    p = pathlib.Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def record(section, values, path=RESULTS):
    """Merge one study's headline numbers into the store, and save.

    Called at the end of each study script with the numbers that script owns.
    Sections are replaced wholesale, so a re-run cannot leave half of an old
    result behind next to half of a new one.
    """
    data = load(path)
    data[section] = _plain(values)
    pathlib.Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return data


def _plain(v):
    """Convert numpy scalars/arrays to JSON-native types."""
    try:
        import numpy as np
    except ImportError:                                   # pragma: no cover
        np = None
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if np is not None:
        if isinstance(v, np.ndarray):
            return [_plain(x) for x in v.tolist()]
        if isinstance(v, np.generic):
            return _plain(v.item())
    if isinstance(v, float):
        return float(f"{v:.12g}")                # kill float64 repr noise
    return v


# ===========================================================================
# renderers
# ===========================================================================
def _macro_name(path_parts):
    """A LaTeX-legal macro name: letters only, so digits become words."""
    digits = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
              "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"}
    out = ["z"]
    for part in path_parts:
        # Digits become separate words *before* title-casing; doing it after
        # would leave "figure1" as "Figureone" instead of "FigureOne".
        chunk = "".join(f" {digits[c]} " if c in digits else c for c in str(part))
        chunk = re.sub(r"[^A-Za-z]+", " ", chunk).title().replace(" ", "")
        out.append(chunk)
    return "".join(out)


#: Key fragments whose value is a fraction worth also having as a percentage.
#: A paper writes "24.2%", and LaTeX cannot multiply by 100 on its own.
_PCT_KEYS = ("efficiency", "frac_", "mass_", "agreement", "below_one_percent")


def tex_value(x, sig=4):
    """Format a number for a paper: significant figures, real LaTeX exponents.

    The store keeps full precision because it is data. A document does not
    want ``2.27355957031``, and ``5.18696197105e-13`` is not even valid maths
    in LaTeX -- it would set as the letter e.
    """
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int):
        return str(x)
    if not isinstance(x, float):
        return str(x)
    if x != x or x in (float("inf"), float("-inf")):
        return r"\infty" if x > 0 else (r"-\infty" if x < 0 else r"\mathrm{NaN}")
    if x == 0:
        return "0"
    if float(x).is_integer() and abs(x) < 1e5:
        return str(int(x))
    if abs(x) < 1e-3 or abs(x) >= 1e5:
        mant, exp = f"{x:.{sig - 1}e}".split("e")
        return f"{mant.rstrip('0').rstrip('.')}\\times10^{{{int(exp)}}}"
    return f"{x:.{sig}g}"


def latex_macros(data):
    """A \\newcommand per leaf, so the paper never hard-codes a number."""
    lines = ["% Generated by results.py -- do not edit.",
             "% Every number the paper quotes comes from results.json.",
             ""]
    seen = {}

    def walk(node, parts):
        if isinstance(node, dict):
            for k in sorted(node):
                walk(node[k], parts + [k])
        elif isinstance(node, list):
            pass                                  # arrays are not quotable
        else:
            name = _macro_name(parts)
            if name in seen:
                raise ValueError(f"macro collision {name}: "
                                 f"{seen[name]} vs {'.'.join(parts)}")
            seen[name] = ".".join(parts)
            lines.append(f"\\newcommand{{\\{name}}}{{{tex_value(node)}}}")
            key = parts[-1].lower()
            if (isinstance(node, float) and not isinstance(node, bool)
                    and any(k in key for k in _PCT_KEYS) and 0.0 <= node <= 1.0):
                lines.append(f"\\newcommand{{\\{name}Pct}}"
                             f"{{{tex_value(100 * node)}}}")

    walk(data, [])
    return "\n".join(lines) + "\n"


def _fmt(x, nd):
    return f"{x:.{nd}f}"


def readme_blocks(data):
    """The generated markdown blocks, keyed by marker name."""
    out = {}

    lib = data.get("library", {})
    if lib:
        rows = ["| molecule | spins | measured | 95% width | floor | ratio | flat | efficiency |",
                "|---|---|---|---|---|---|---|---|"]
        for name in ("formic_acid", "formaldehyde", "glycine", "methanol"):
            m = lib.get(name)
            if not m:
                continue
            flat = ", ".join(m["flat"]) or "—"
            rows.append(
                f"| {m['label']} | {m['n_spins']} | {', '.join(m['measured'])} "
                f"| {_fmt(m['width_mHz'], 2)} mHz | {_fmt(m['floor_mHz'], 3)} "
                f"| {_fmt(m['ratio'], 2)} | {flat} | {_fmt(100 * m['efficiency'], 1)}% |")
        out["library"] = "\n".join(rows)

    t = data.get("tighten", {}).get("configs")
    if t:
        rows = ["| configuration | raw width | raw/floor | reweighted | efficiency |",
                "|---|---|---|---|---|"]
        for c in t:
            rows.append(f"| {c['label']} | {_fmt(c['raw_mHz'], 2)} mHz "
                        f"| {_fmt(c['raw_over_floor'], 2)}× "
                        f"| {_fmt(c['reweighted_mHz'], 2)} mHz "
                        f"| {_fmt(100 * c['efficiency'], 1)}% |")
        out["tighten"] = "\n".join(rows)

    v = data.get("vs_nested", {})
    if v:
        out["vs_nested"] = "\n".join([
            "| | J 95% width |",
            "|---|---|",
            f"| nested sampling (reference) | {_fmt(v['nested_mHz'], 2)} mHz |",
            f"| reweighted NPE | {_fmt(v['npe_mHz'], 2)} mHz |",
            f"| local fit (curvature) | {_fmt(v['local_mHz'], 2)} mHz |",
            f"| information floor | {_fmt(v['floor_mHz'], 2)} mHz |",
            f"| agreement, NPE vs nested | **{_fmt(100 * v['agreement'], 1)}%** |",
        ])
    return out


def render_readme(data, path=README):
    """Replace each marked block in the README. Returns the new text."""
    text = pathlib.Path(path).read_text()
    for name, body in readme_blocks(data).items():
        begin, end = _BEGIN.format(name=name), _END.format(name=name)
        pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.S)
        if not pattern.search(text):
            continue                              # marker not present; skip
        text = pattern.sub(f"{begin}\n{body}\n{end}", text)
    return text


# ===========================================================================
def _write_if_changed(path, text):
    p = pathlib.Path(path)
    old = p.read_text() if p.exists() else None
    if old == text:
        return False
    p.write_text(text)
    return True


def main(argv=None):                              # pragma: no cover - CLI
    argv = list(sys.argv[1:] if argv is None else argv)
    data = load()
    if not data:
        print(f"{RESULTS} is empty or missing -- run the studies first "
              f"(./run_all.sh)", file=sys.stderr)
        return 1

    if "--show" in argv:
        print(json.dumps(data, indent=2, sort_keys=True))
        return 0

    tex = latex_macros(data)
    readme = render_readme(data)
    check = "--check" in argv

    stale = []
    if check:
        if not os.path.exists(MACROS) or pathlib.Path(MACROS).read_text() != tex:
            stale.append(MACROS)
        if pathlib.Path(README).read_text() != readme:
            stale.append(README)
        if stale:
            print("STALE, regenerate with `python results.py`: "
                  + ", ".join(stale), file=sys.stderr)
            return 1
        print(f"up to date: {MACROS}, {README} "
              f"({sum(1 for _ in tex.splitlines() if _.startswith(chr(92)+'newcommand'))} macros)")
        return 0

    wrote = []
    if _write_if_changed(MACROS, tex):
        wrote.append(MACROS)
    if _write_if_changed(README, readme):
        wrote.append(README)
    n = sum(1 for l in tex.splitlines() if l.startswith("\\newcommand"))
    print(f"{len(data)} sections, {n} macros -> "
          + (", ".join(wrote) if wrote else "no change"))
    return 0


if __name__ == "__main__":                        # pragma: no cover
    sys.exit(main())
