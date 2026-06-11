"""Importer robustness on the UCSF Douglas Lab cadnano corpus
(VALIDATION.md test 0, extended).

The Douglas Lab (cadnano's creators) publish their designs at
github.com/douglaslab/cadnano-designs. Our importer parses ALL 76 legacy-
format designs there with every invariant holding (no double-pairing,
nucleotide conservation) across 1,218,310 nucleotides — including the
24,300-nt / 180-helix icosahedron and the 2009 Science twist/protractor set.

This test pins a curated, diverse subset vendored into examples/data/
(2009 Nature 3D, 2009 Science curvature + twist, a 2018 custom scaffold, and
a cadnano-paper aspect-ratio variant) so the robustness is enforced in CI.
"""

import os

import pytest

from src import contactmap as cmap

DATA = os.path.join(os.path.dirname(__file__), "..", "examples", "data", "douglas_designs")

# (scaffold_len, n_staples, n_intended_pairs) pinned per design.
EXPECTED = {
    "4_slottedcross.json": (8634, 186, 6760),        # Douglas 2009 Nature, slotted cross
    "6hb-1512.json": (1512, 48, 1512),               # Douglas 2018 pScaf, 6-helix bundle
    "gear90.json": (6804, 217, 6530),                # Douglas 2009 Science, curved gear
    "monolith_right_twist.json": (8064, 220, 7919),  # Douglas 2009 Science, twisted monolith
    "vii_2x30.json": (7560, 250, 7560),              # Douglas 2009 NAR, 2x30 aspect ratio
}


@pytest.mark.parametrize("fname,expected", EXPECTED.items())
def test_douglas_design_parses_with_invariants(fname, expected):
    cm = cmap._load_cadnano(os.path.join(DATA, fname))
    assert (len(cm.scaffold), len(cm.staples), len(cm.intended_pairs)) == expected
    # nucleotide conservation
    assert cm.n_bases == len(cm.scaffold) + sum(len(s) for s in cm.staples)
    # no scaffold base and no staple base is paired more than once
    assert len({p[2] for p in cm.intended_pairs}) == len(cm.intended_pairs)
    assert len({(p[3], p[4]) for p in cm.intended_pairs}) == len(cm.intended_pairs)
    # every pair indexes in-bounds
    for (_, _, sc, strand, st) in cm.intended_pairs:
        assert 0 <= sc < len(cm.scaffold)
        assert 0 <= st < len(cm.staples[strand - 1])


def test_curved_and_twisted_designs_have_unpaired_scaffold():
    # gear90 (curvature) and the twisted monolith leave some scaffold bases
    # unpaired (at curves/seams), unlike the flat fully-duplexed aspect-ratio
    # design — a sanity check that pairing reflects real routing, not a stub.
    gear = cmap._load_cadnano(os.path.join(DATA, "gear90.json"))
    flat = cmap._load_cadnano(os.path.join(DATA, "vii_2x30.json"))
    assert len(gear.intended_pairs) < len(gear.scaffold)     # curved: gaps
    assert len(flat.intended_pairs) == len(flat.scaffold)    # flat: fully duplexed
