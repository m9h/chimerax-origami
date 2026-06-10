"""Generative routing for chimerax-origami — target shape -> DNA-origami
design (the inverse-design front end).

Mirrors how chimerax-vampnet treats generative protein models (AlphaFlow /
BioEmu / Boltz): a learned model proposes a structure, the rest of the stack
evaluates it. Here the model is the **diffusion DNA-origami generator** of
*De novo design of DNA origami with a generative diffusion model* (Nat Commun
2026, s41467-026-73578-z): trained on simulated equilibrium conformations, it
turns a target shape into a physically plausible routing (scaffold path +
staple set) via guided diffusion + strand routing, with integrated structure
prediction for evaluation.

This module is the bundle-side interface. A backend (md/diffusion_modal.py)
runs the real model; with no backend it falls back to a deterministic
*parametric router* — a minimal procedural origami compiler that lays a
scaffold snaking through an H-helix bundle with one staple per helix and
exact base pairing. The fallback is a real, valid, scorable design (not a
stub): it lets `generate -> score -> shape -> assembly_msm -> evolve` run end
to end today, and swaps out for the diffusion model with no API change.

OUTPUT is a ContactMap (the shared abstraction): scaffold (poly-N until a
sequence is applied), staples, intended_pairs, helices. `apply_scaffold`
threads a real sequence through the routing and re-derives staple bases as
the Watson-Crick complement — so a generated routing is immediately scorable
and sequence-optimizable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .contactmap import ContactMap, reverse_complement, _COMPLEMENT


@dataclass
class Target:
    """A target shape spec. The diffusion backend consumes the full spec
    (incl. an optional voxel/mesh); the fallback uses n_helices x length.

      type:      'bundle' | 'sheet' | 'rod' (a hint; fallback treats all as a
                 lattice of n_helices helices each `length` bp)
      n_helices: number of parallel helices
      length:    bp per helix
    """
    type: str = "bundle"
    n_helices: int = 6
    length: int = 64
    voxels: Optional[object] = None   # (X,Y,Z) occupancy for the diffusion model


def build_bundle_routing(n_helices: int, length: int, name: str = "generated") -> ContactMap:
    """Deterministic parametric router: a scaffold that snakes through an
    n_helices x length bundle (helix 0 left->right, helix 1 right->left, ...),
    one staple per helix, every base paired exactly once.

    scaffold global index s runs 0..n_helices*length-1 in snake order;
    helices[s] = s // length. Staple h (1-based strand h+1) has `length`
    bases; staple base t pairs scaffold base h*length + (length-1-t)
    (antiparallel), so staple h == reverse_complement(scaffold[h]) once a
    sequence is applied.
    """
    n = n_helices * length
    scaffold = "N" * n
    helices = [s // length for s in range(n)]
    staples = ["N" * length for _ in range(n_helices)]
    intended_pairs = []
    for h in range(n_helices):
        for t in range(length):
            sc_idx = h * length + (length - 1 - t)
            intended_pairs.append(("sc-st", 0, sc_idx, h + 1, t))
    return ContactMap(scaffold=scaffold, staples=staples,
                      intended_pairs=intended_pairs, helices=helices, name=name)


def apply_scaffold(cm: ContactMap, scaffold_sequence: str) -> ContactMap:
    """Thread a scaffold sequence through a routing and re-derive staple bases
    as the complement of their paired scaffold base. Returns a new ContactMap
    with real sequences, ready for `score` / `shape` / `evolve`.
    """
    seq = scaffold_sequence.upper().replace("U", "T")
    n = len(cm.scaffold)
    if len(seq) < n:
        seq = seq + "N" * (n - len(seq))
    scaffold = seq[:n]
    staple_chars = [list(s) for s in cm.staples]
    for (_, _, sc_idx, strand, st_idx) in cm.intended_pairs:
        base = scaffold[sc_idx] if sc_idx < len(scaffold) else "N"
        if 0 <= strand - 1 < len(staple_chars) and st_idx < len(staple_chars[strand - 1]):
            staple_chars[strand - 1][st_idx] = _COMPLEMENT.get(base, "N")
    staples = ["".join(c) for c in staple_chars]
    return ContactMap(scaffold=scaffold, staples=staples,
                      intended_pairs=list(cm.intended_pairs),
                      helices=cm.helices, name=cm.name)


class RoutingGenerator:
    """Generate a routing for a target shape via a diffusion backend, with a
    deterministic parametric fallback. Mirrors Evo2Mutator / ShapePredictor:
    any backend error degrades to the fallback rather than raising.

    backend: object exposing .sample(target: Target, seed: int) -> dict with
        keys scaffold_len / staples (list of lengths) / intended_pairs /
        helices, OR a ready ContactMap. Pass the client from
        md/diffusion_modal.py, or None for the fallback.
    """

    def __init__(self, backend=None):
        self.backend = backend

    def generate(self, target: Target, seed: int = 0) -> ContactMap:
        if self.backend is not None:
            try:
                out = self.backend.sample(target, seed)
                if isinstance(out, ContactMap):
                    return out
                return ContactMap(
                    scaffold="N" * out["scaffold_len"],
                    staples=["N" * L for L in out["staples"]],
                    intended_pairs=[tuple(p) for p in out["intended_pairs"]],
                    helices=out.get("helices"),
                    name=out.get("name", "generated"),
                )
            except Exception:
                pass
        return build_bundle_routing(target.n_helices, target.length)


def generate_routing(target: Target, backend=None, seed: int = 0) -> ContactMap:
    """Convenience wrapper. Returns a ContactMap routing for the target."""
    return RoutingGenerator(backend=backend).generate(target, seed=seed)
