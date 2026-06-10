"""Generative-routing tests (diffusion generator + parametric fallback).

GPU-free: exercises the fallback router and a FakeDiffusion backend, then
confirms a generated routing flows into the rest of the stack (apply
sequence -> score -> shape). Mirrors the Evo2 / DGNN adapter test pattern.
"""

import numpy as np

from src import generate as gen
from src import score as sc
from src import shape as sh
from src.contactmap import reverse_complement


def test_fallback_routing_is_valid():
    cm = gen.generate_routing(gen.Target(n_helices=6, length=64))
    assert len(cm.scaffold) == 6 * 64
    assert len(cm.staples) == 6
    assert all(len(s) == 64 for s in cm.staples)
    assert len(cm.intended_pairs) == 6 * 64
    # every scaffold base paired exactly once; every staple base paired once.
    assert len({p[2] for p in cm.intended_pairs}) == 6 * 64
    assert len({(p[3], p[4]) for p in cm.intended_pairs}) == 6 * 64
    assert len(cm.helices) == 6 * 64 and set(cm.helices) == set(range(6))


def test_apply_scaffold_derives_complementary_staples():
    cm = gen.generate_routing(gen.Target(n_helices=2, length=16))
    rng = np.random.default_rng(0)
    seq = "".join("ACGT"[i] for i in rng.integers(0, 4, 32))
    cm2 = gen.apply_scaffold(cm, seq)
    assert cm2.scaffold == seq
    # staple h pairs scaffold[h*16:(h+1)*16] antiparallel -> reverse complement.
    assert cm2.staples[0] == reverse_complement(seq[0:16])
    assert cm2.staples[1] == reverse_complement(seq[16:32])
    # and every recovered pair is genuinely complementary (revcomp of a single
    # base is its complement).
    for (_, _, sc_idx, strand, st_idx) in cm2.intended_pairs:
        assert cm2.staples[strand - 1][st_idx] == reverse_complement(cm2.scaffold[sc_idx])


def test_generated_routing_flows_into_score_and_shape():
    cm = gen.generate_routing(gen.Target(n_helices=4, length=48))
    cm = gen.apply_scaffold(cm, "ACGT" * 48)
    sd = sc.score(cm, k=8)
    assert sd.objectives["total"] >= 0.0
    res = sh.predict_shape(cm, n_ensemble=4)
    assert res.n_nodes == len(cm.scaffold) and res.radius_of_gyration > 0


def test_backend_used_when_provided():
    class FakeDiffusion:
        def sample(self, target, seed):
            # a tiny 1-helix routing dict in the documented schema.
            return {"scaffold_len": 10, "staples": [10],
                    "intended_pairs": [("sc-st", 0, i, 1, 9 - i) for i in range(10)],
                    "helices": [0] * 10, "name": "fake"}
    cm = gen.generate_routing(gen.Target(), backend=FakeDiffusion())
    assert cm.name == "fake"
    assert len(cm.scaffold) == 10 and len(cm.intended_pairs) == 10


def test_broken_backend_falls_back():
    class Broken:
        def sample(self, target, seed):
            raise RuntimeError("no checkpoint")
    cm = gen.generate_routing(gen.Target(n_helices=3, length=20), backend=Broken())
    assert len(cm.scaffold) == 60          # parametric fallback used, no raise


def test_generate_then_evolve_pipeline():
    # The intended composition: diffusion proposes routing, evolve optimizes
    # the scaffold sequence on it (staples re-derived via apply_scaffold).
    from src import evolve as ev
    cm = gen.generate_routing(gen.Target(n_helices=3, length=32))
    cm = gen.apply_scaffold(cm, "AAAAGGGGCCCCTTTT" * 6)
    result = ev.evolve(cm, generations=40, seed=0)
    assert result["best_total"] <= result["seed_total"]
