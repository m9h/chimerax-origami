#!/usr/bin/env python3
"""Standalone demo of the DGNN geometric forward model (origami shape).

Runs on a bare DGX Spark venv (no ChimeraX, no GPU) by stubbing the
chimerax runtime, exactly like recursive_improvement_demo.py. Exercises the
deterministic lattice fallback of src/shape.py — swap in a DGNNBackend from
md/gnn_shape_modal.py once a checkpoint exists to get the real model.

    python examples/shape_predict_demo.py
"""

import os
import sys
import types

# --- make `src` importable without a ChimeraX session -------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if "chimerax" not in sys.modules:
    for name in ["chimerax", "chimerax.core", "chimerax.core.toolshed",
                 "chimerax.core.commands", "chimerax.core.errors"]:
        sys.modules[name] = types.ModuleType(name)
    sys.modules["chimerax.core.toolshed"].BundleAPI = object
    sys.modules["chimerax.core.errors"].UserError = type("UserError", (Exception,), {})

import numpy as np

from src.contactmap import ContactMap, reverse_complement
from src import shape as sh
from src import evolve as ev


def demo_design():
    seg = "GGGGCCCCAAAATTTTGGGGCCCC"
    scaffold = (seg + "ACAC" + reverse_complement(seg)) * 4
    return ContactMap(scaffold=scaffold, staples=["TTTTGGGGCCCCAAAA"], name="m13_demo")


def show(label, cm):
    res = sh.predict_shape(cm, backend=None, n_ensemble=12, seed=0)
    s = res.summary()
    print(f"  {label:<22} Rg={s['radius_of_gyration_nm']:6.2f} nm   "
          f"extent={s['max_extent_nm']:6.2f} nm   "
          f"mean RMSF={s['mean_rmsf_nm']:.3f} nm   "
          f"backend={s['backend']}")
    return res


def main():
    print("DGNN geometric forward model — origami shape (lattice fallback)")
    print("=" * 70)

    seed = demo_design()

    # 1. Predict the shape of the starting design.
    print("\n[1] Shape of the seed design:")
    res_seed = show("seed", seed)
    top = res_seed.summary()["flexible_hotspots"][:3]
    print(f"      most flexible base pairs: "
          f"{', '.join(f'#{h['node']}({h['rmsf_nm']:.2f}nm)' for h in top)}")

    # 2. Sequence-dependence: the same length, soft (TA) vs stiff (GC).
    print("\n[2] Flexibility is sequence-dependent (same length):")
    soft = ContactMap(scaffold="TA" * 60, name="soft (TA x60)")
    stiff = ContactMap(scaffold="GC" * 60, name="stiff (GC x60)")
    r_soft = show("soft  (TA-rich)", soft)
    r_stiff = show("stiff (GC-rich)", stiff)
    ratio = r_soft.mean_rmsf / max(r_stiff.mean_rmsf, 1e-9)
    print(f"      -> soft scaffold is {ratio:.2f}x floppier than the stiff one")

    # 3. Kabsch RMSD to a target shape (here: the seed's own coords -> ~0).
    print("\n[3] Shape RMSD to a target (sanity: target = self -> ~0):")
    rmsd = sh.predict_shape(seed, n_ensemble=12, seed=0,
                            target_coords=res_seed.coords).shape_rmsd_to_target
    print(f"      RMSD to self = {rmsd:.4f} nm")

    # 4. Tie-in to the loop: evolve a low-frustration scaffold, then check that
    #    the geometric forward model still returns a well-formed shape for it.
    print("\n[4] Forward-model check on an evolved design:")
    evolved = ev.evolve(seed, generations=200, seed=0, point_rate=0.03)
    evolved_cm = ContactMap(scaffold=evolved["best"]["scaffold"],
                            staples=seed.staples, name="evolved")
    show("evolved (best)", evolved_cm)
    print(f"      off-target frustration {evolved['seed_total']:.0f} -> "
          f"{evolved['best_total']:.0f}; geometry still well-formed above")

    print("\nDone. Wire md/gnn_shape_modal.py::DGNNBackend for the real DGNN.")


if __name__ == "__main__":
    main()
