"""Evo2Mutator integration tests.

These never touch Evo 2 / Modal — they use a deterministic FakeBackend so
CI stays dependency-light, exactly as vampnet's adapter-smoke tests import
the adapters without running the GPU models. We verify (a) the pluggable
mutator hook, (b) FM-guided proposals fuse the off-target objective with
the backend's likelihood, (c) graceful fallback when no backend is given,
and (d) reproducibility under a fixed seed.
"""

from src.contactmap import ContactMap, reverse_complement
from src import evolve as ev


def _seed_design():
    seg = "GGGGCCCCAAAATTTT"
    scaffold = (seg + "ACAC" + reverse_complement(seg)) * 3
    return ContactMap(scaffold=scaffold, staples=["TTTTGGGGCCCCAAAA"], name="mut_seed")


class FakeEvo2:
    """Deterministic stand-in for the Evo 2 backend. `score` rewards a
    balanced GC content (a cheap, monotone, no-dep proxy for 'natural-like'),
    so the test can assert the FM prior actually influences selection.
    `generate` returns simple deterministic infills.
    """

    def __init__(self):
        self.n_score_calls = 0

    def score(self, seqs):
        self.n_score_calls += 1
        out = []
        for s in seqs:
            gc = (s.count("G") + s.count("C")) / max(len(s), 1)
            out.append(-abs(gc - 0.5))  # 0 at GC=0.5, negative otherwise
        return out

    def generate(self, prefix, n_tokens, n, temperature=0.7):
        return [("ACGT" * (n_tokens // 4 + 1))[:n_tokens] for _ in range(n)]


def test_random_mutator_is_default_and_matches_legacy():
    cm = _seed_design()
    r = ev.evolve(cm, generations=80, seed=5)
    assert r["mutator"] == "RandomMutator"
    assert r["fm_guided"] is False
    assert r["best_total"] <= r["seed_total"]


def test_evo2_mutator_runs_and_improves():
    cm = _seed_design()
    backend = FakeEvo2()
    mut = ev.Evo2Mutator(backend=backend, mode="score", n_candidates=5, fm_weight=20.0)
    r = ev.evolve(cm, generations=80, seed=5, mutator=mut)
    assert r["mutator"] == "Evo2Mutator"
    assert r["fm_guided"] is True
    assert r["best_total"] <= r["seed_total"]
    assert backend.n_score_calls > 0          # the FM was actually consulted
    assert mut.template is cm                  # evolve() injected the design


def test_evo2_mutator_generate_mode():
    cm = _seed_design()
    mut = ev.Evo2Mutator(backend=FakeEvo2(), mode="generate", n_candidates=4)
    r = ev.evolve(cm, generations=40, seed=1, mutator=mut)
    assert r["fm_guided"] is True
    assert r["best_total"] <= r["seed_total"]


def test_evo2_mutator_falls_back_without_backend():
    # No backend -> behaves like random mutation, never raises.
    mut = ev.Evo2Mutator(backend=None)
    cm = _seed_design()
    out = mut(cm.scaffold, __import__("random").Random(0))
    assert isinstance(out, str) and len(out) == len(cm.scaffold)


def test_evo2_mutator_survives_broken_backend():
    class Broken:
        def score(self, seqs):
            raise RuntimeError("simulated OOM")

    cm = _seed_design()
    mut = ev.Evo2Mutator(backend=Broken(), mode="score")
    # Must not raise — degrades to random mutation per generation.
    r = ev.evolve(cm, generations=30, seed=2, mutator=mut)
    assert r["best_total"] <= r["seed_total"]


def test_evo2_guided_is_reproducible():
    cm = _seed_design()
    r1 = ev.evolve(cm, generations=60, seed=9, mutator=ev.Evo2Mutator(backend=FakeEvo2()))
    r2 = ev.evolve(cm, generations=60, seed=9, mutator=ev.Evo2Mutator(backend=FakeEvo2()))
    assert r1["best"]["scaffold"] == r2["best"]["scaffold"]
