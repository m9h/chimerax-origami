"""Off-target scorer validation (VALIDATION.md test 1).

Validates the scorer on real M13mp18 and a real published routing, and pins
the fix for the diagonal-skip bug that previously masked designed self-
complementarity. The ViennaRNA cross-check is skipped if RNA isn't installed.
"""

import os
import random

import pytest

from src.contactmap import ContactMap, reverse_complement
from src import score as sc
from src import optimize as opt

DATA = os.path.join(os.path.dirname(__file__), "..", "examples", "data")


def _j2(seq, k=8):
    return sc.score(ContactMap(scaffold=seq), k=k).objectives["j2_scaffold_scaffold"]


def _spearman(a, b):
    def rank(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
                j += 1
            for kk in range(i, j + 1):
                r[order[kk]] = (i + j) / 2
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    return 1 - 6 * sum((ra[i] - rb[i]) ** 2 for i in range(n)) / (n * (n * n - 1))


def test_designed_self_complementarity_is_detected():
    # Regression for the diagonal-skip bug: a region and its reverse
    # complement downstream MUST register as scaffold<->scaffold off-target.
    rng = random.Random(1)
    A = "".join(rng.choice("ACGT") for _ in range(150))
    structured = A + "".join(rng.choice("ACGT") for _ in range(100)) + reverse_complement(A)
    shuffled = "".join(rng.sample(structured, len(structured)))
    assert _j2(structured) > 50
    assert _j2(structured) > 10 * max(_j2(shuffled), 1)   # structure >> its shuffle


def test_detection_population():
    rng = random.Random(0)
    R = lambda n: "".join(rng.choice("ACGT") for _ in range(n))
    randoms = [R(400) for _ in range(20)]
    structured = [(lambda A: A + R(100) + reverse_complement(A))(R(150)) for _ in range(20)]
    mr = sum(_j2(s) for s in randoms) / 20
    ms = sum(_j2(s) for s in structured) / 20
    assert ms > 20 * max(mr, 1)


def test_viennarna_cross_tool_correlation():
    RNA = pytest.importorskip("RNA")
    rng = random.Random(0)
    R = lambda n: "".join(rng.choice("ACGT") for _ in range(n))
    seqs = [R(rng.randint(80, 120)) for _ in range(150)]
    js = [_j2(s, k=5) for s in seqs]
    mfes = [-RNA.fold(s.replace("T", "U"))[1] for s in seqs]
    # our cheap k-mer self-complementarity tracks ViennaRNA's MFE.
    assert _spearman(js, mfes) > 0.3


def test_scaffold_choice_changes_offtarget_on_real_routing():
    # The Krasnogor claim: for a FIXED routing (real Douglas 2009 monolith),
    # the scaffold sequence determines off-target.
    from src import contactmap as cmap
    mono = os.path.join(DATA, "Nature09_monolith.json")
    m13 = "".join(c for c in open(os.path.join(DATA, "m13mp18.txt")).read()
                  if c in "ACGT")
    N = 7560
    m13_tiled = (m13 * 2)[:N]
    shuffled = "".join(random.Random(0).sample(m13_tiled, N))
    t_m13 = sc.score(cmap._load_cadnano(mono, scaffold_sequence=m13_tiled), k=10).objectives["total"]
    t_shuf = sc.score(cmap._load_cadnano(mono, scaffold_sequence=shuffled), k=10).objectives["total"]
    assert t_m13 != t_shuf            # sequence choice matters for the same routing
    assert t_m13 > 0 and t_shuf > 0


def test_optimize_rejects_pathological_scaffold():
    # optimize() must not pick a low-complexity (high self-complementarity)
    # scaffold over balanced ones.
    rng = random.Random(2)
    good1 = ContactMap(scaffold="".join(rng.choice("ACGT") for _ in range(400)), name="good1")
    good2 = ContactMap(scaffold="".join(rng.choice("ACGT") for _ in range(400)), name="good2")
    bad = ContactMap(scaffold="ACGT" * 100, name="pathological")
    res = opt.optimize([good1, good2, bad], k=8)
    assert res["nodes"][res["best_compromise"]]["name"] != "pathological"
    # the pathological scaffold has the highest total off-target.
    totals = {n["name"]: n["total"] for n in res["nodes"]}
    assert totals["pathological"] == max(totals.values())
