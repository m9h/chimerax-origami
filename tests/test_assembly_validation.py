"""Assembly-MSM validation (VALIDATION.md test 3).

Two validations:
  - MSM validity: the slowest implied timescale is ~constant across lag
    times (Markovianity / Chapman-Kolmogorov-style check), the standard test
    that the recovered MSM is a faithful kinetic model.
  - The project thesis: static off-target frustration (score.py) lowers the
    folding YIELD of the cheap kinetic emulator (simulate_folding) — i.e.
    frustration predicts worse assembly, the claim test 6 checks against the
    wet lab. The link is non-circular: it flows sequence -> score ->
    localized per-domain frustration -> kinetics -> yield.
"""

import random

import numpy as np

from src import assembly as asm
from src import generate as gen, score as sc
from src.contactmap import reverse_complement


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


def test_implied_timescale_convergence():
    # The slowest implied timescale must plateau across lag -> the MSM is
    # Markovian on this trajectory.
    feats, _ = asm.synthetic_assembly_trajectory(n_frames=6000, n_pairs=30, seed=0)
    slowest = []
    for lag in (4, 6, 8, 10):
        ts = asm.fit_assembly_msm(feats, lag=lag, n_states=3,
                                  backend="numpy", seed=0).implied_timescales()
        slowest.append(max(ts) if ts else 0.0)
    assert min(slowest) > 0
    assert max(slowest) / min(slowest) < 1.3      # flat to within 30%


def _scaffold(frac_struct, length, seed):
    rng = random.Random(seed)
    base = "".join(rng.choice("ACGT") for _ in range(length))
    m = int(frac_struct * (length // 2))
    A = "".join(rng.choice("ACGT") for _ in range(m))
    return A + base[m:length - m] + reverse_complement(A)


def _yield_of(cm, reps=3):
    return float(np.mean([asm.simulate_folding(cm, n_frames=1500, seed=r, k=6)[750:].mean()
                          for r in range(reps)]))


def test_frustration_lowers_folding_yield():
    # THE THESIS: more static off-target frustration -> lower folding yield.
    routing = gen.generate_routing(gen.Target(n_helices=4, length=24))
    N = 96
    offt, yields = [], []
    for fs in (0.0, 0.2, 0.4, 0.6):
        cm = gen.apply_scaffold(routing, _scaffold(fs, N, seed=42))
        offt.append(sc.score(cm, k=6).objectives["total"])
        yields.append(_yield_of(cm))
    # off-target rises with fs; yield must fall -> strong negative correlation.
    assert _spearman(offt, yields) < -0.5
    assert yields[0] > yields[-1]      # least-frustrated folds best


def test_simulate_folding_output_is_sane():
    cm = gen.apply_scaffold(gen.generate_routing(gen.Target(n_helices=3, length=20)),
                            "ACGT" * 15)
    feats = asm.simulate_folding(cm, n_frames=500, seed=0, k=6)
    assert feats.shape == (500, len(cm.intended_pairs))
    assert 0.0 <= feats.mean() <= 1.0
