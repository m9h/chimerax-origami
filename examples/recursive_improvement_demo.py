#!/usr/bin/env python3
"""Standalone demo of the Sakana-style recursive-improvement loop.

Runs on a bare DGX Spark venv (no ChimeraX needed) by stubbing the
chimerax runtime, exactly as tests/conftest.py does. Mirrors the spirit of
chimerax-vampnet's examples/live_adaptive_sampling.py — a self-improving
loop you can watch tick — but the "experiment" each step is a cheap
off-target score instead of a 100 ns MD launch.

    python examples/recursive_improvement_demo.py
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

from src.contactmap import ContactMap, reverse_complement
from src import evolve as ev


def make_frustrated_scaffold():
    """A deliberately trap-rich scaffold: repeated self-complementary blocks
    (the kind of secondary structure the Krasnogor paper selects against).
    """
    block = "GGGGCCCCAAAATTTTGGGGCCCC"
    return (block + "ACAC" + reverse_complement(block)) * 4


def main():
    seed = ContactMap(scaffold=make_frustrated_scaffold(), name="m13_demo")

    print("Recursive-improvement loop over a DNA-origami scaffold")
    print("=" * 60)

    def tick(step):
        if step["generation"] % 50 == 0:
            print(f"  gen {step['generation']:>4}  "
                  f"archive={step['archive_size']:>3}  "
                  f"best_total={step['best_total']:8.1f}  "
                  f"niche={step['best_descriptor']}")

    result = ev.evolve(seed, generations=400, seed=0, point_rate=0.03, on_step=tick)

    print("-" * 60)
    print(f"seed total off-target frustration : {result['seed_total']:.1f}")
    print(f"best total off-target frustration : {result['best_total']:.1f}")
    print(f"improvement                       : {result['improvement']:.1f} "
          f"({100*result['improvement']/max(result['seed_total'],1):.1f}%)")
    print(f"archive niches explored           : {result['archive_size']} "
          f"/ {result['n_niches_possible']} possible")
    print(f"stepping stones in best lineage   : {len(result['best_lineage'])}")
    print("\nbest scaffold (first 60 nt):")
    print("  " + result["best"]["scaffold"][:60])


if __name__ == "__main__":
    main()
