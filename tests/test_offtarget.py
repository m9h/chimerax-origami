"""Off-target scorer + Pareto selector smoke tests.

Mirrors chimerax-vampnet's tests/test_random_walk.py: a synthetic input
with a known answer exercises the whole featurize -> score -> optimize
wiring. Here the synthetic input is a scaffold deliberately seeded with a
self-complementary repeat, which MUST register as scaffold<->scaffold
(j2) frustration.
"""

from src.contactmap import ContactMap, reverse_complement
from src import score as score_mod
from src import optimize as opt_mod


def test_reverse_complement():
    assert reverse_complement("AAAA") == "TTTT"
    assert reverse_complement("GATC") == "GATC"  # palindrome
    assert reverse_complement("ACGT") == "ACGT"


def test_self_complementary_scaffold_is_frustrated():
    # A scaffold containing a segment and its reverse complement downstream
    # must light up the j2 (scaffold<->scaffold) channel.
    seg = "GGGGCCCCAAAATTTT"
    spacer = "ACACACACACAC"
    scaffold = seg + spacer + reverse_complement(seg)
    cm = ContactMap(scaffold=scaffold, name="selfcomp")
    sd = score_mod.score(cm, k=8)
    assert sd.objectives["j2_scaffold_scaffold"] > 0
    assert sd.objectives["total"] >= sd.objectives["j2_scaffold_scaffold"]


def test_clean_scaffold_scores_lower():
    clean = "ACGTACGAGTCAGTCAGGATCAGATCAGCATCAGCATCAGCAACGACTACG"
    seg = "GGGGCCCCAAAATTTT"
    dirty = seg + "ACAC" + reverse_complement(seg)
    s_clean = score_mod.score(ContactMap(scaffold=clean, name="clean"), k=8)
    s_dirty = score_mod.score(ContactMap(scaffold=dirty, name="dirty"), k=8)
    assert s_clean.objectives["total"] < s_dirty.objectives["total"]


def test_pareto_front_includes_dominant_best():
    cands = [
        ContactMap(scaffold="ACGTACGAGTCAGTCAGGATCAGATCAGCATCAGCATCAGCAACG", name="a"),
        ContactMap(scaffold="GGGGCCCCAAAATTTTACACGGGGCCCCAAAATTTT", name="b"),
        ContactMap(scaffold="TACGATCGATCAGCATCGATCAGCTAGCATCGATCAGCATCG", name="c"),
    ]
    res = opt_mod.optimize(cands, k=8)
    assert res["n_candidates"] == 3
    assert len(res["pareto_front"]) >= 1
    assert res["best_compromise"] in range(3)


def test_dominates():
    assert opt_mod.dominates([1, 1, 1, 1], [2, 2, 2, 2])
    assert not opt_mod.dominates([1, 3, 1, 1], [2, 2, 2, 2])  # worse on obj2
    assert not opt_mod.dominates([2, 2, 2, 2], [2, 2, 2, 2])  # equal, not strict
