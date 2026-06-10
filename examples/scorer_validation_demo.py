#!/usr/bin/env python3
"""Validation of the off-target scorer (VALIDATION.md test 1).

Three checks, on the REAL M13mp18 scaffold where relevant:
  A. Detection   — designed self-complementarity scores far above random.
  B. Cross-tool  — our scaffold-self channel (j2) correlates with ViennaRNA's
                   thermodynamic MFE (an independent, established tool).
  C. Krasnogor   — for a FIXED routing (the real Douglas 2009 monolith), the
                   scaffold SEQUENCE choice changes off-target, and `optimize`
                   selects against the pathological one. This is the paper's
                   central, falsifiable claim.

    python examples/scorer_validation_demo.py     # ViennaRNA optional
"""

import os
import sys
import types
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if "chimerax" not in sys.modules:
    for n in ["chimerax", "chimerax.core", "chimerax.core.toolshed",
              "chimerax.core.commands", "chimerax.core.errors"]:
        sys.modules[n] = types.ModuleType(n)
    sys.modules["chimerax.core.toolshed"].BundleAPI = object
    sys.modules["chimerax.core.errors"].UserError = type("UserError", (Exception,), {})

from src.contactmap import ContactMap, reverse_complement
from src import score as sc
from src import optimize as opt

DATA = os.path.join(os.path.dirname(__file__), "data")


def j2(seq, k=6):
    return sc.score(ContactMap(scaffold=seq), k=k).objectives["j2_scaffold_scaffold"]


def spearman(a, b):
    def rank(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    d2 = sum((ra[i] - rb[i]) ** 2 for i in range(n))
    return 1 - 6 * d2 / (n * (n * n - 1))


def main():
    rng = random.Random(0)
    R = lambda n: "".join(rng.choice("ACGT") for _ in range(n))
    print("Off-target scorer validation")
    print("=" * 64)

    # A. Detection ------------------------------------------------------
    randoms = [R(400) for _ in range(20)]
    structured = []
    for _ in range(20):
        A = R(150)
        structured.append(A + R(100) + reverse_complement(A))   # A ... RC(A)
    mr = sum(j2(s, 8) for s in randoms) / 20
    ms = sum(j2(s, 8) for s in structured) / 20
    print(f"\n[A] Designed self-complementarity detection (k=8):")
    print(f"    random scaffolds      mean j2 = {mr:.0f}")
    print(f"    A + spacer + RC(A)    mean j2 = {ms:.0f}   ({ms/max(mr,1):.0f}x higher)")

    # B. ViennaRNA cross-tool -------------------------------------------
    print(f"\n[B] Cross-tool agreement vs ViennaRNA MFE:")
    try:
        import RNA
        seqs = [R(rng.randint(80, 120)) for _ in range(150)]
        js = [j2(s, 5) for s in seqs]
        mfes = [-RNA.fold(s.replace("T", "U"))[1] for s in seqs]
        print(f"    Spearman(j2, -MFE) over 150 random seqs = {spearman(js, mfes):+.3f}")
        print(f"    (positive => our cheap k-mer proxy tracks real thermodynamics)")
    except ImportError:
        print("    ViennaRNA not installed (pip install ViennaRNA) — skipped")

    # C. Krasnogor: scaffold choice on a fixed real routing -------------
    print(f"\n[C] Scaffold choice on the real Douglas 2009 monolith routing:")
    from src import contactmap as cmap
    mono = os.path.join(DATA, "Nature09_monolith.json")
    m13 = "".join(c for c in open(os.path.join(DATA, "m13mp18.txt")).read()
                  if c in "ACGT")
    N = 7560
    m13_tiled = (m13 * 2)[:N]
    shuffled = "".join(rng.sample(m13_tiled, N))
    repetitive = ("ACGT" * (N // 4 + 1))[:N]            # low-complexity = pathological
    candidates = []
    for label, seq in [("M13 (natural)", m13_tiled),
                       ("M13 shuffled", shuffled),
                       ("ACGT repeat", repetitive)]:
        cm = cmap._load_cadnano(mono, scaffold_sequence=seq)
        cm.name = label
        tot = sc.score(cm, k=10).objectives["total"]
        candidates.append(cm)
        print(f"    {label:<16} total off-target = {tot:,.0f}")
    res = opt.optimize(candidates, k=10)
    best = res["nodes"][res["best_compromise"]]["name"]
    print(f"    -> optimize() best compromise: {best!r}")
    print(f"    (same routing, different sequence => different off-target: "
          f"the paper's claim)")


if __name__ == "__main__":
    main()
