#!/usr/bin/env python3
"""Standalone demo of the VAMPnet-on-oxDNA method bridge (origami assembly_msm).

Runs GPU-free on a bare venv (numpy fallback MSM) by stubbing the chimerax
runtime, like the other example demos. Generates a synthetic oxDNA-like
folding trajectory with a known kinetic trap, then runs the SAME
featurize -> cluster -> MSM -> transition-graph pipeline chimerax-vampnet
uses for protein conformational dynamics — and recovers the origami's folding
intermediates, flagging the off-target trap as a metastable state.

    python examples/assembly_msm_demo.py
"""

import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if "chimerax" not in sys.modules:
    for name in ["chimerax", "chimerax.core", "chimerax.core.toolshed",
                 "chimerax.core.commands", "chimerax.core.errors"]:
        sys.modules[name] = types.ModuleType(name)
    sys.modules["chimerax.core.toolshed"].BundleAPI = object
    sys.modules["chimerax.core.errors"].UserError = type("UserError", (Exception,), {})

from src import assembly as asm


def main():
    print("VAMPnet-on-oxDNA bridge — Markov state model of origami folding")
    print("=" * 70)

    # 1. A synthetic oxDNA-like folding trajectory: unfolded -> trap -> folded,
    #    where the trap forms ~half its contacts in the WRONG register.
    feats, true_states = asm.synthetic_assembly_trajectory(
        n_frames=4000, n_pairs=32, seed=0)
    print(f"\ntrajectory: {feats.shape[0]} frames x {feats.shape[1]} intended "
          f"base pairs (contact-occupancy features)")
    print(f"ground-truth occupancy: unfolded={feats[true_states==0].mean():.0%}  "
          f"trap={feats[true_states==1].mean():.0%}  "
          f"folded={feats[true_states==2].mean():.0%}")

    # 2. The shared pipeline: cluster frames -> MSM. Same machinery vampnet
    #    applies to protein MD (deeptime if installed, numpy fallback here).
    msm = asm.fit_assembly_msm(feats, lag=5, n_states=3, backend="auto", seed=0)
    s = msm.summary()

    print(f"\nfitted MSM (backend={s['backend']}):")
    labels = s["state_labels"]
    for i in range(s["n_states"]):
        flag = "  <-- KINETIC TRAP (off-target basin)" if i in s["trap_states"] else ""
        print(f"  state {i}: {labels[i]:<13} "
              f"frac_folded={s['frac_folded'][i]:.2f}  "
              f"population={s['stationary_distribution'][i]:.2f}{flag}")

    print(f"\nimplied timescales (frames): "
          f"{[round(t, 1) for t in s['implied_timescales']]}")
    print(f"traps found: {s['n_traps']}  (states {s['trap_states']})")

    # 3. The payoff: an off-target interaction is, kinetically, a metastable
    #    trap state — the exact object a protein-folding MSM surfaces. The
    #    transition graph below is byte-compatible with vampnet's msm output.
    g = msm.transition_graph()
    print("\ntransition graph (vampnet-compatible JSON shape):")
    print(f"  nodes={len(g['nodes'])}  edges={len(g['edges'])}  "
          f"edge_density={g['edge_density']:.2f}")
    folded = [n['id'] for n in g['nodes'] if n['label'] == 'folded'][0]
    for n in g['nodes']:
        if n['is_trap']:
            to_folded = next((e['rate'] for e in g['edges']
                              if e['src'] == n['id'] and e['dst'] == folded), 0.0)
            print(f"  trap state {n['id']}: self-residence "
                  f"{g['transition_matrix'][n['id']][n['id']]:.2f}, "
                  f"escape-to-folded rate {to_folded:.3f}/lag")

    print("\nSame featurize->VAMPnet/MSM->graph pipeline as chimerax-vampnet,")
    print("applied to assembly instead of conformational dynamics. Wire")
    print("md/oxdna_modal.py to produce real trajectories.")


if __name__ == "__main__":
    main()
