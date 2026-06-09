"""Recursive-improvement loop tests.

Mirrors the intent of chimerax-vampnet's adaptive-sampling demo: a short
deterministic run of the evolutionary loop must (a) be reproducible under a
fixed seed and (b) not make the design worse than the seed (the archive
always retains at least the seed, so best_total <= seed_total).
"""

from src.contactmap import ContactMap, reverse_complement
from src import evolve as ev


def _seed_design():
    seg = "GGGGCCCCAAAATTTT"
    scaffold = (seg + "ACAC" + reverse_complement(seg)) * 3
    return ContactMap(scaffold=scaffold, name="evolve_seed")


def test_evolve_is_reproducible():
    cm = _seed_design()
    r1 = ev.evolve(cm, generations=60, seed=7)
    r2 = ev.evolve(cm, generations=60, seed=7)
    assert r1["best_total"] == r2["best_total"]
    assert r1["best"]["scaffold"] == r2["best"]["scaffold"]


def test_evolve_never_regresses_below_seed():
    cm = _seed_design()
    r = ev.evolve(cm, generations=120, seed=1)
    assert r["best_total"] <= r["seed_total"]
    assert r["improvement"] >= 0.0


def test_evolve_keeps_diverse_archive():
    # Open-endedness: the MAP-Elites archive should hold more than one niche
    # after enough generations (stepping stones, not a single champion).
    cm = _seed_design()
    r = ev.evolve(cm, generations=200, seed=3, point_rate=0.05)
    assert r["archive_size"] >= 2
    assert len(r["best_lineage"]) >= 1
    assert r["best_lineage"][0]["generation"] == 0  # lineage roots at the seed


def test_improvement_curve_is_monotone_nonincreasing():
    cm = _seed_design()
    r = ev.evolve(cm, generations=100, seed=2)
    curve = r["improvement_curve"]
    assert all(curve[i + 1] <= curve[i] + 1e-9 for i in range(len(curve) - 1))
