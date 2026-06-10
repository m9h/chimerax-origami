"""Evo2-guided vs random mutation ablation (VALIDATION.md test 4).

GPU-free: uses a deterministic FakeEvo2 surrogate (the real Evo 2 plausibility
check — natural >> shuffled log-likelihood — is checkpoint-pending). Validates
(a) that score/FM-guided mutation with worst-window targeting reaches lower
off-target than random mutation in a fixed generation budget, and (b) the
targeting mechanism itself.
"""

import random

from src import evolve as ev
from src.contactmap import ContactMap, reverse_complement


class FakeEvo2:
    def score(self, seqs):
        # rewards balanced GC (a cheap stand-in for a sequence-quality prior).
        return [-abs(((s.count("G") + s.count("C")) / max(len(s), 1)) - 0.5) for s in seqs]


def _hard_seed():
    rng = random.Random(0)
    blocks = []
    for _ in range(8):
        A = "".join(rng.choice("ACGT") for _ in range(30))
        blocks.append(A + "AC" * 5 + reverse_complement(A))   # A ... RC(A): off-target
    return ContactMap(scaffold="".join(blocks),
                      staples=["".join(rng.choice("ACGT") for _ in range(20))], name="hard")


def test_guided_beats_random_at_fixed_budget():
    seed = _hard_seed()
    budget = 20
    rand_best, guided_best = [], []
    for sd in range(4):
        rand_best.append(ev.evolve(seed, generations=budget, seed=sd)["best_total"])
        guided_best.append(ev.evolve(
            seed, generations=budget, seed=sd,
            mutator=ev.Evo2Mutator(backend=FakeEvo2(), mode="score", fm_weight=20.0),
        )["best_total"])
    # guided should reach lower off-target on average within the budget.
    assert sum(guided_best) / 4 < sum(rand_best) / 4
    assert all(g <= r for g, r in zip(guided_best, rand_best))


def test_guided_targets_worst_window():
    # off-target concentrated in the MIDDLE (a region + its RC); with
    # explore=0 the mutator must select a window overlapping it.
    flanks = "ACAGTCAGTCAGGATCAGATCAGCATCAG"
    A = "GGGGCCCCAAAATTTTGGGGCCCC"
    seq = flanks + A + "ACAC" + reverse_complement(A) + flanks
    mid_start = len(flanks)
    mid_end = len(seq) - len(flanks)
    mut = ev.Evo2Mutator(backend=FakeEvo2(), mode="score", window=24, explore=0.0, k=8)
    s, e = mut._window_bounds(seq, random.Random(0))
    # the chosen window overlaps the frustrated middle region.
    assert s < mid_end and e > mid_start


def test_explore_can_pick_random_window():
    # with explore=1 the window is random (not score-based) — keeps exploration.
    seq = "ACGT" * 50
    mut = ev.Evo2Mutator(backend=FakeEvo2(), window=24, explore=1.0)
    s, e = mut._window_bounds(seq, random.Random(3))
    assert e - s == 24 and 0 <= s <= len(seq) - 24
