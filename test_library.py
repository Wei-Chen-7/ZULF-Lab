"""Tests for the trained-network library.

No training happens here -- these check the table itself, which is the part
that can be quietly wrong. A truth vector of the wrong length or a molecule
whose flat directions are mislabelled would produce a library that looks fine
and ships the prior back as a measurement.
"""

import numpy as np
import pytest

import zulf_infer as zi
import train_library as tl


@pytest.mark.parametrize("name", sorted(tl.LIBRARY))
def test_truth_matches_the_parameter_count(name):
    prob = zi.InferenceProblem(system=name, seed=0)
    assert len(tl.LIBRARY[name]["truth"]) == len(prob.param_names)


@pytest.mark.parametrize("name", sorted(tl.LIBRARY))
def test_truth_lies_inside_the_prior(name):
    prob = zi.InferenceProblem(system=name, seed=0)
    truth = np.asarray(tl.LIBRARY[name]["truth"], float)
    assert prob.in_prior(truth[None, :])[0], f"{name}: {truth} outside the prior"


@pytest.mark.parametrize("name", sorted(tl.LIBRARY))
def test_measurable_coupling_is_off_the_prior_centre(name):
    """Recovering a coupling you started at the centre of proves nothing."""
    prob = zi.InferenceProblem(system=name, seed=0)
    truth = np.asarray(tl.LIBRARY[name]["truth"], float)
    centre = 0.5 * (prob.low[0] + prob.high[0])
    assert abs(truth[0] - centre) > 0.05 * prob.prior_span()[0]


@pytest.mark.parametrize("name", sorted(tl.LIBRARY))
def test_the_simulator_runs_at_the_stated_truth(name):
    prob = zi.InferenceProblem(system=name, seed=0)
    x = prob.simulate_one(tl.LIBRARY[name]["truth"], noisy=False)
    assert x.shape == (prob.x_dim(),)
    assert np.all(np.isfinite(x))


@pytest.mark.parametrize("name", sorted(tl.LIBRARY))
def test_tags_are_unique_and_name_their_size(name):
    tags = [s["tag"] for s in tl.LIBRARY.values()]
    assert len(set(tags)) == len(tags)
    spec = tl.LIBRARY[name]
    assert spec["tag"].startswith(name)
    assert f"{spec['n_sims'] // 1000}k" in spec["tag"]


def test_every_equivalent_group_molecule_has_a_flat_coupling_class():
    """Three of the four have equivalent protons, so J_HH must be a class.

    Keeping the flat class in the parameterization is deliberate: it is only
    reported as unmeasured because it is there to be reported on. Dropping it
    would hide the same fact behind a smaller parameter vector.
    """
    for name in ("formaldehyde", "glycine", "methanol"):
        classes = [c[0] for c in zi.COUPLING_CLASSES[name]]
        assert "J_HH" in classes, name
    assert [c[0] for c in zi.COUPLING_CLASSES["formic_acid"]] == ["J_CH"]


@pytest.mark.parametrize("name", ["formaldehyde", "glycine", "methanol"])
def test_J_HH_moves_no_line_for_every_library_molecule(name):
    """The physical claim the 'not measured' label rests on."""
    prob = zi.InferenceProblem(system=name, seed=0)
    base = np.asarray(tl.LIBRARY[name]["truth"], float)
    ihh = list(prob.param_names).index("J_HH")
    x0 = prob.simulate_one(base, noisy=False)
    for v in (prob.low[ihh], 0.0, prob.high[ihh]):
        t = base.copy()
        t[ihh] = v
        assert np.max(np.abs(prob.simulate_one(t, noisy=False) - x0)) < 1e-9
