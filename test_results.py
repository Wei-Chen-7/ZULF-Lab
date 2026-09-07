"""Tests for the single-source results store.

The store exists because numbers drifted: an audit found a stale library table,
a stale comparison table, a stale eigenvalue list, an error bar off by a factor
of 1.3, and a reversed conclusion, all because four documents transcribed the
same run logs by hand. These tests pin the properties that make drift
*detectable* rather than merely less likely.
"""

import json
import pathlib

import pytest

import results


@pytest.fixture
def store(tmp_path):
    return str(tmp_path / "results.json")


# -- the store ---------------------------------------------------------------
def test_record_then_load_round_trips(store):
    results.record("a", {"x": 1.5, "y": "text"}, path=store)
    assert results.load(store) == {"a": {"x": 1.5, "y": "text"}}


def test_sections_are_replaced_not_merged(store):
    """Half an old result next to half a new one is the failure to avoid."""
    results.record("a", {"old": 1, "shared": 1}, path=store)
    results.record("a", {"shared": 2}, path=store)
    assert results.load(store)["a"] == {"shared": 2}


def test_other_sections_survive_a_record(store):
    results.record("a", {"x": 1}, path=store)
    results.record("b", {"y": 2}, path=store)
    assert set(results.load(store)) == {"a", "b"}


def test_load_of_a_missing_file_is_empty(tmp_path):
    assert results.load(str(tmp_path / "nope.json")) == {}


def test_numpy_values_are_stored_as_json_natives(store):
    np = pytest.importorskip("numpy")
    results.record("a", {"f": np.float64(2.5), "i": np.int64(3),
                         "arr": np.array([1.0, 2.0])}, path=store)
    raw = json.loads(pathlib.Path(store).read_text())
    assert raw["a"] == {"f": 2.5, "i": 3, "arr": [1.0, 2.0]}


def test_float_repr_noise_is_trimmed(store):
    results.record("a", {"x": 0.1 + 0.2}, path=store)
    assert "0.30000000000000004" not in pathlib.Path(store).read_text()


# -- LaTeX macros ------------------------------------------------------------
def test_macro_names_are_letters_only():
    """LaTeX macro names cannot contain digits or underscores."""
    tex = results.latex_macros({"lib": {"formic_acid": {"width_mHz": 2.29,
                                                        "n_spins": 2}}})
    names = [l.split("{")[1].rstrip("}") for l in tex.splitlines()
             if l.startswith("\\newcommand")]
    assert names, tex
    for n in names:
        assert n.startswith("\\")
        assert n[1:].isalpha(), n


def test_digits_in_keys_become_words():
    tex = results.latex_macros({"figure1": {"n": 1}})
    assert "zFigureOneN" in tex


def test_every_leaf_becomes_a_macro():
    tex = results.latex_macros({"a": {"x": 1, "y": 2}, "b": {"z": 3}})
    assert tex.count("\\newcommand") == 3


def test_arrays_are_not_quotable():
    """A list has no single value, so it must not become a macro."""
    tex = results.latex_macros({"a": {"xs": [1, 2, 3], "n": 3}})
    assert tex.count("\\newcommand") == 1
    assert "zAN" in tex


def test_macro_collisions_are_refused():
    """Two different keys must never render to the same macro name."""
    with pytest.raises(ValueError, match="collision"):
        results.latex_macros({"a": {"b_c": 1, "b__c": 2}})


# -- README rendering --------------------------------------------------------
_LIB = {"library": {"formic_acid": dict(
    label="formic acid", n_spins=2, measured=["J_CH"], flat=[],
    width_mHz=2.2888, floor_mHz=2.2633, ratio=1.0113, efficiency=0.2419,
    shrinkage={"J_CH": 0.9996})}}


def test_readme_block_is_rendered_between_markers(tmp_path):
    p = tmp_path / "R.md"
    begin = results._BEGIN.format(name="library")
    end = results._END.format(name="library")
    p.write_text(f"before\n{begin}\nSTALE TABLE\n{end}\nafter\n")
    out = results.render_readme(_LIB, path=str(p))
    assert "STALE TABLE" not in out
    assert "formic acid" in out and "2.29 mHz" in out
    assert out.startswith("before\n") and out.rstrip().endswith("after")


def test_rendering_is_idempotent(tmp_path):
    """Re-rendering an up-to-date README must change nothing, or --check lies."""
    p = tmp_path / "R.md"
    begin = results._BEGIN.format(name="library")
    end = results._END.format(name="library")
    p.write_text(f"{begin}\nx\n{end}\n")
    once = results.render_readme(_LIB, path=str(p))
    p.write_text(once)
    assert results.render_readme(_LIB, path=str(p)) == once


def test_text_outside_the_markers_is_untouched(tmp_path):
    p = tmp_path / "R.md"
    begin = results._BEGIN.format(name="library")
    end = results._END.format(name="library")
    p.write_text(f"KEEP ME\n{begin}\nold\n{end}\nKEEP ME TOO\n")
    out = results.render_readme(_LIB, path=str(p))
    assert "KEEP ME" in out and "KEEP ME TOO" in out


def test_a_missing_marker_is_skipped_not_an_error(tmp_path):
    p = tmp_path / "R.md"
    p.write_text("no markers here\n")
    assert results.render_readme(_LIB, path=str(p)) == "no markers here\n"


def test_flat_column_shows_an_em_dash_when_nothing_is_flat():
    block = results.readme_blocks(_LIB)["library"]
    assert "| — |" in block


def test_flat_column_names_the_unmeasurable_coupling():
    data = {"library": {"methanol": dict(
        label="methanol", n_spins=4, measured=["J_CH"], flat=["J_HH"],
        width_mHz=1.04, floor_mHz=1.012, ratio=1.03, efficiency=0.098,
        shrinkage={"J_CH": 0.9998, "J_HH": 0.011})}}
    assert "J_HH" in results.readme_blocks(data)["library"]


# -- the real repository -----------------------------------------------------
def test_the_committed_store_renders_without_error():
    data = results.load()
    if not data:
        pytest.skip("results.json not populated yet")
    results.latex_macros(data)
    results.readme_blocks(data)


def test_the_committed_readme_is_not_stale():
    """This is the check the old hand-transcribed documents could not do."""
    data = results.load()
    if not data or not pathlib.Path(results.README).exists():
        pytest.skip("nothing to check yet")
    current = pathlib.Path(results.README).read_text()
    if results._BEGIN.format(name="library") not in current:
        pytest.skip("markers not installed yet")
    assert results.render_readme(data) == current, (
        "README generated blocks are stale -- run `python results.py`")
