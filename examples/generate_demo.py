#!/usr/bin/env python3
"""Standalone demo of generative routing -> the full design loop.

GPU-free (deterministic parametric-router fallback for the diffusion model).
Shows the inverse-design composition the whole stack was built for:

    target shape --generate--> routing
                 --apply M13--> sequenced design
                 --score/shape--> evaluate
                 --evolve--> optimize the scaffold on that routing

Swap in md/diffusion_modal.py::DiffusionBackend for the real Nat Commun 2026
diffusion generator with no other changes.

    python examples/generate_demo.py
"""

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

from src import generate as gen
from src import score as sc
from src import shape as sh
from src import evolve as ev

DATA = os.path.join(os.path.dirname(__file__), "data")


def main():
    print("Generative routing -> full design loop")
    print("=" * 60)

    # 1. Generate a routing for a target shape (6-helix bundle, 64 bp/helix).
    target = gen.Target(type="bundle", n_helices=6, length=64)
    cm = gen.generate_routing(target, backend=None)
    print(f"\n[1] generate(target={target.type}, {target.n_helices}x{target.length}) ->")
    print(f"    scaffold {len(cm.scaffold)} nt, {len(cm.staples)} staples, "
          f"{len(cm.intended_pairs)} base pairs, {len(set(cm.helices))} helices")

    # 2. Thread the real M13 scaffold through the routing; staples become its
    #    Watson-Crick complement automatically.
    m13 = "".join(c for c in open(os.path.join(DATA, "m13mp18.txt")).read()
                  if c in "ACGT")
    cm = gen.apply_scaffold(cm, m13)
    print(f"\n[2] apply M13 -> staple[0] = {cm.staples[0][:24]}...")

    # 3. Evaluate with the two forward models.
    sd = sc.score(cm, k=8)
    res = sh.predict_shape(cm, n_ensemble=8)
    print(f"\n[3] evaluate: off-target total = {sd.objectives['total']:.0f}  |  "
          f"Rg = {res.radius_of_gyration:.2f} nm  |  mean RMSF = {res.mean_rmsf:.3f} nm")

    # 4. Optimize the scaffold sequence on this generated routing.
    result = ev.evolve(cm, generations=150, seed=0, point_rate=0.03)
    print(f"\n[4] evolve on the generated routing: off-target "
          f"{result['seed_total']:.0f} -> {result['best_total']:.0f} "
          f"({result['archive_size']} stepping stones)")

    print("\nDone. target -> routing -> sequence -> score/shape -> optimize,")
    print("all on one ContactMap. Wire DiffusionBackend for the real generator.")


if __name__ == "__main__":
    main()
