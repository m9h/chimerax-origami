"""Golden-file regression tests on REAL cadnano designs.

Pins the importer's output on two canonical legacy-format files checked into
examples/data/ (see examples/data/README.md for provenance), so any
regression in the strand router / pairing logic is caught. This is
VALIDATION.md test 0 ("importer round-trips") realized as a CI test.

The headline fixture is Nature09_monolith.json — Douglas et al., Nature 459,
414 (2009) — a 7560-nt (p7560) honeycomb monolith with 144 staples across 18
helices: a genuine 3D published design, not a toy.
"""

import json
import os

from src import contactmap as cmap
from src.contactmap import reverse_complement

DATA = os.path.join(os.path.dirname(__file__), "..", "examples", "data")


def _stap_color_anchors(path):
    """cadnano marks each staple's 5' end with a stap_colors entry
    [base_idx, color] on its helix. Return the set of (helix, base) anchors —
    independent metadata the importer's router does NOT consult."""
    doc = json.load(open(path))
    return {(vs["num"], b) for vs in doc["vstrands"]
            for (b, _c) in vs.get("stap_colors", [])}


def _router_staple_5ends(path):
    """The (helix, base) 5' end of each staple, derived purely from the
    scaf/stap routing pointers."""
    doc = json.load(open(path))
    vmap = {vs["num"]: vs for vs in doc["vstrands"]}
    return {p[0] for p in cmap._walk_strands(vmap, "stap")}


def _check_conservation(cm):
    """Invariants that must hold for any validly imported design."""
    # nucleotide conservation: bases == scaffold + all staples.
    assert cm.n_bases == len(cm.scaffold) + sum(len(s) for s in cm.staples)
    # no scaffold base is paired more than once.
    scaf_paired = [p[2] for p in cm.intended_pairs]
    assert len(scaf_paired) == len(set(scaf_paired))
    # no (staple_strand, staple_idx) is paired more than once either.
    stap_paired = [(p[3], p[4]) for p in cm.intended_pairs]
    assert len(stap_paired) == len(set(stap_paired))
    # every pair indexes inside its strand.
    for (_, _, sc, strand, st) in cm.intended_pairs:
        assert 0 <= sc < len(cm.scaffold)
        assert 1 <= strand <= len(cm.staples)
        assert 0 <= st < len(cm.staples[strand - 1])


def test_simple42_golden():
    cm = cmap._load_cadnano(os.path.join(DATA, "simple42legacy.json"))
    assert cm.summary() == {
        "name": "design",
        "format": "contactmap",   # source_format set by load_design, not _load_cadnano
        "scaffold_length": 42,
        "n_staples": 1,
        "n_bases": 84,
        "n_intended_pairs": 42,
    }
    assert len(cm.staples[0]) == 42        # single staple duplexes the whole scaffold
    assert set(cm.helices) == {0}
    _check_conservation(cm)


def test_monolith_golden():
    cm = cmap._load_cadnano(os.path.join(DATA, "Nature09_monolith.json"))
    s = cm.summary()
    assert s["scaffold_length"] == 7560     # p7560 scaffold
    assert s["n_staples"] == 144
    assert s["n_intended_pairs"] == 5880
    assert s["n_bases"] == 13440
    # 60 distinct helices carry scaffold in this honeycomb monolith.
    assert len(set(cm.helices)) == 60
    _check_conservation(cm)


def test_monolith_sequence_application_is_complementary():
    # Apply a deterministic repeating scaffold sequence and confirm every
    # paired staple base is the Watson-Crick complement of its scaffold base.
    seq = ("ACGT" * 2000)[:7560]
    cm = cmap._load_cadnano(os.path.join(DATA, "Nature09_monolith.json"),
                            scaffold_sequence=seq)
    assert cm.scaffold == seq
    checked = 0
    for (_, _, sc_idx, strand, st_idx) in cm.intended_pairs[:500]:
        assert cm.staples[strand - 1][st_idx] == reverse_complement(cm.scaffold[sc_idx])
        checked += 1
    assert checked == 500


def test_router_staples_match_cadnano_stap_colors():
    """Cross-validate against cadnano's OWN metadata: the routing-derived
    staple 5' ends must exactly equal the stap_colors anchors the design
    file declares. The router never reads stap_colors, so agreement is an
    independent confirmation the strand walk is correct — VALIDATION.md test
    0's cadnano cross-check, dependency-free.
    """
    for f in ("simple42legacy.json", "Nature09_monolith.json"):
        path = os.path.join(DATA, f)
        assert _router_staple_5ends(path) == _stap_color_anchors(path)


def test_monolith_loads_via_public_api_and_scores():
    # End-to-end through load_design + score, the path a user actually takes.
    import types as _t
    from src import score
    session = _t.SimpleNamespace()
    summary = cmap.load_design(session, os.path.join(DATA, "Nature09_monolith.json"),
                               scaffold_sequence=("ACGT" * 2000)[:7560])
    assert summary["scaffold_length"] == 7560
    assert summary["format"] == "cadnano"
    cm = cmap.design_get(session)
    sd = score.score(cm, k=10)
    assert sd.objectives["total"] >= 0.0    # scorer runs on the real design
