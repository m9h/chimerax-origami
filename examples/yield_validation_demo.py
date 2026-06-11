#!/usr/bin/env python3
"""Test 6 (capstone): predicted off-target vs. MEASURED wet-lab data.

Real scaffold variants from the Krasnogor paper (Zenodo 14748478): three
triangle DNA-origami designs, one routing, three scaffold sequences. We score
each with our INDEPENDENT off-target scorer and compare to the paper's single-
molecule optical-tweezers measurement of structural non-uniformity (Fig. 7D),
keyed by the supplementary mapping T1=DEER, T2=LION, T3=BEAR.

    python examples/yield_validation_demo.py
"""

import csv
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if "chimerax" not in sys.modules:
    for n in ["chimerax", "chimerax.core", "chimerax.core.toolshed",
              "chimerax.core.commands", "chimerax.core.errors"]:
        sys.modules[n] = types.ModuleType(n)
    sys.modules["chimerax.core.toolshed"].BundleAPI = object
    sys.modules["chimerax.core.errors"].UserError = type("UserError", (Exception,), {})

from src import contactmap as cmap, score as sc

DATA = os.path.join(os.path.dirname(__file__), "data")


def spearman(a, b):
    def rank(x):
        o = sorted(range(len(x)), key=lambda i: x[i])
        r = [0] * len(x)
        for pos, i in enumerate(o):
            r[i] = pos
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    return 1 - 6 * sum((ra[i] - rb[i]) ** 2 for i in range(n)) / (n * (n * n - 1))


def main():
    meas = {}
    with open(os.path.join(DATA, "krasnogor_measured.csv")) as f:
        for r in csv.DictReader(f):
            meas[r["codename"]] = r

    print("Test 6: our off-target score vs. measured structural non-uniformity")
    print("=" * 70)
    print(f"{'variant':>16} {'our off-target':>15} {'meas non-unif':>14} {'unfold force':>13}")
    triangle = ["DEER", "LION", "BEAR"]   # T1, T2, T3
    ours, nonunif = [], []
    for cn in triangle:
        cm = cmap._load_scadnano(os.path.join(DATA, "krasnogor_variants", cn + ".sc"))
        o = sc.score(cm, k=8).objectives["total"]
        m = meas[cn]
        ours.append(o)
        nonunif.append(float(m["non_uniformity"]))
        print(f"{cn+' ('+m['paper_label']+')':>16} {o:>15.0f} "
              f"{m['non_uniformity']:>14} {m['unfolding_force']:>13}")

    print(f"\n  lower off-target = fewer kinetic traps (our prediction)")
    print(f"  lower non-uniformity = more uniform, better-folded (measured, "
          f"single-molecule optical tweezers)")
    rho = spearman(ours, nonunif)
    print(f"\n  Spearman(our off-target, measured non-uniformity) = {rho:+.2f}")
    if rho > 0.99:
        print("  => our independent static score reproduces the measured folding-quality")
        print("     ranking on real DNA origami. Static frustration predicts assembly.")
    print("\n  (n=3 triangle variants; rectangle R1/R2/R3 = GOAT/LAMB/MOLE have AFM/gel")
    print("   data in the paper's supplementary figures, not extracted here.)")


if __name__ == "__main__":
    main()
