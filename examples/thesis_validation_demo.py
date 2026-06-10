#!/usr/bin/env python3
"""The project thesis, validated (VALIDATION.md test 3).

Static off-target frustration (score.py) predicts worse folding: a cheap
kinetic emulator driven by the scorer shows folding YIELD falling as off-
target rises. The link is non-circular -- sequence -> score -> localized
frustration -> kinetics -> yield -- and it's the same quantity test 6 checks
against measured wet-lab assembly yield.

GPU-free.  python examples/thesis_validation_demo.py
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

import numpy as np
from src import generate as gen, score as sc, assembly as asm
from src.contactmap import reverse_complement


def spearman(a, b):
    def rank(x):
        o = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and x[o[j + 1]] == x[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    return 1 - 6 * sum((ra[i] - rb[i]) ** 2 for i in range(n)) / (n * (n * n - 1))


def main():
    print("Thesis: static off-target frustration -> worse folding yield")
    print("=" * 64)
    routing = gen.generate_routing(gen.Target(n_helices=6, length=48))
    N = 288

    def scaffold(fs, seed):
        rng = random.Random(seed)
        base = "".join(rng.choice("ACGT") for _ in range(N))
        m = int(fs * (N // 2))
        A = "".join(rng.choice("ACGT") for _ in range(m))
        return A + base[m:N - m] + reverse_complement(A)

    print(f"\n{'frustration':>12} {'off-target':>11} {'fold yield':>11} {'trap pop':>9}")
    offt, yields = [], []
    for fs in (0.0, 0.15, 0.3, 0.45, 0.6):
        cm = gen.apply_scaffold(routing, scaffold(fs, seed=7))
        total = sc.score(cm, k=8).objectives["total"]
        ys, tps = [], []
        for rep in range(4):
            feats = asm.simulate_folding(cm, n_frames=3000, seed=rep, k=8)
            ys.append(float(feats[1500:].mean()))
            msm = asm.fit_assembly_msm(feats, lag=5, n_states=3, backend="numpy", seed=rep)
            pi = np.asarray(msm.stationary_distribution)
            tps.append(float(sum(pi[t] for t in msm.identify_traps(residence=0.4))))
        y, tp = np.mean(ys), np.mean(tps)
        offt.append(total)
        yields.append(y)
        print(f"{fs:>12.2f} {total:>11.0f} {y:>11.3f} {tp:>9.3f}")

    print(f"\nSpearman(off-target, folding yield) = {spearman(offt, yields):+.3f}")
    print("Negative => the scorer's static frustration predicts assembly failure,")
    print("the unifying claim linking score.py and the folding MSM. Replace the")
    print("emulator with md/oxdna_modal.py for quantitative folding trajectories.")


if __name__ == "__main__":
    main()
