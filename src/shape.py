"""Geometric forward model for chimerax-origami — predicts the 3D shape
(and per-node flexibility) of a design from its contact map.

This is the structural twin of how chimerax-vampnet turns a sequence into a
predicted structure (its AlphaFold/Boltz/ESMFold adapters). Here the
predictor is a **DGNN** — the DNA-origami graph neural network of
Truong-Quoc, Lee, Kim & Kim, *Nat Mater* 2024 (Do-Nyun Kim lab, SNU) — which
infers the relaxed 3D conformation of an origami almost in real time, the
fast surrogate for an oxDNA/CanDo relaxation (the MarS-FM analog on the DNA
side).

THE CONTACT MAP IS, AGAIN, THE NATIVE INPUT
The DGNN graph is exactly our `ContactMap`, re-typed:
  - **node**  = a base pair (position + orientation)        <- intended_pairs / scaffold bases
  - **structural edge** = backbone bond / crossover, carrying *sequence-
        dependent* mechanical stiffness (base-pair-step mechanics)  <- routing + sequence
  - **electrostatic edge** = distance-dependent repulsion (emerges during
        relaxation)
The published DGNN runs 5 mechanical-relaxation blocks + 1 electrostatic
refinement block, trained with a hybrid data-driven + physics-informed loss,
and uses an ensemble strategy for near-real-time monomer inference. We build
the graph here and hand it to a backend (md/gnn_shape_modal.py) or, with no
backend, fall back to a deterministic lattice placement so the wiring/tests
run GPU-free.

WHY IT MATTERS FOR THE LOOP
Shape is set mostly by *routing* (held fixed in the sequence-evolve loop),
but the DGNN's structural edges are sequence-dependent, so the predictor
also yields a per-base-pair **flexibility (RMSF)** that the scaffold sequence
*does* modulate. So this module is (a) a validation/forward-model step for a
finished design (`origami shape`) and (b) the geometric fitness for a future
routing-space optimizer — the counterpart to score.py's sequence fitness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .contactmap import ContactMap


# Relative base-pair-step bending stiffness (illustrative, normalized ~1.0).
# Flexible pyrimidine-purine steps (e.g. TA) are softer; GC-rich steps
# stiffer. Real values would come from the DGNN's trained edge features /
# published cgDNA parameter sets; these are placeholders so the fallback
# flexibility profile is sequence-dependent rather than uniform.
_STEP_STIFFNESS = {
    "AA": 1.0, "AT": 1.1, "AG": 1.0, "AC": 1.0,
    "TA": 0.7, "TT": 1.0, "TG": 0.9, "TC": 0.9,
    "GA": 1.0, "GT": 0.9, "GG": 1.2, "GC": 1.3,
    "CA": 0.9, "CT": 1.0, "CG": 1.2, "CC": 1.2,
}
_BP_RISE_NM = 0.34
_HELIX_PITCH_NM = 2.6   # inter-helix spacing on a honeycomb/square lattice


@dataclass
class ShapeResult:
    coords: object              # (N, 3) numpy array — mean predicted positions
    rmsf: object                # (N,) numpy array — per-node flexibility
    n_nodes: int
    radius_of_gyration: float
    max_extent: float
    mean_rmsf: float
    max_rmsf: float
    shape_rmsd_to_target: Optional[float] = None
    backend: str = "fallback"

    def summary(self) -> dict:
        top = sorted(range(self.n_nodes), key=lambda i: -float(self.rmsf[i]))[:15]
        return {
            "backend": self.backend,
            "n_nodes": self.n_nodes,
            "radius_of_gyration_nm": round(self.radius_of_gyration, 2),
            "max_extent_nm": round(self.max_extent, 2),
            "mean_rmsf_nm": round(self.mean_rmsf, 3),
            "max_rmsf_nm": round(self.max_rmsf, 3),
            "shape_rmsd_to_target_nm": (
                None if self.shape_rmsd_to_target is None
                else round(self.shape_rmsd_to_target, 2)
            ),
            "flexible_hotspots": [{"node": i, "rmsf_nm": round(float(self.rmsf[i]), 3)}
                                  for i in top],
        }


def build_graph(cm: ContactMap) -> dict:
    """Re-type a ContactMap as a DGNN-style graph.

    nodes are base pairs (or scaffold bases when the design is routing-only);
    structural edges are backbone steps carrying sequence-dependent stiffness;
    helix ids (when present) place nodes on the lattice. Returns a dict of
    plain lists/arrays so it serializes cleanly to a remote backend.
    """
    import numpy as np

    seq = cm.scaffold.upper()
    n = len(seq)
    helices = cm.helices if cm.helices else [0] * n
    # If helices is per-vstrand (cadnano import), broadcast a single helix id
    # across the whole scaffold so the fallback still lays out a bundle.
    if len(helices) != n:
        helices = [0] * n

    node_base = list(seq)
    # Structural (backbone) edges + per-edge stiffness from the dinucleotide step.
    edge_src, edge_dst, edge_stiff = [], [], []
    for i in range(n - 1):
        step = seq[i:i + 2]
        edge_src.append(i)
        edge_dst.append(i + 1)
        edge_stiff.append(_STEP_STIFFNESS.get(step, 1.0))

    return {
        "n_nodes": n,
        "node_base": node_base,
        "helix": list(helices),
        "edge_index": np.array([edge_src, edge_dst], dtype=np.int64),
        "edge_stiffness": np.array(edge_stiff, dtype=np.float32),
    }


def _fallback_predict(graph: dict, n_ensemble: int, seed: int = 0):
    """Deterministic, GPU-free stand-in for the DGNN.

    Places each base pair on a lattice (helix index -> transverse position,
    base index -> axial rise) and adds seeded thermal jitter inversely scaled
    by the local structural stiffness, so the resulting per-node RMSF is
    sequence-dependent (soft steps wobble more) — a crude but honest proxy
    for the DGNN's mechanical relaxation. Replace with the trained model via
    md/gnn_shape_modal.py.
    """
    import numpy as np

    n = graph["n_nodes"]
    helix = np.asarray(graph["helix"], dtype=np.float32)
    stiff = np.ones(n, dtype=np.float32)
    ei = graph["edge_index"]
    es = graph["edge_stiffness"]
    # Spread each edge's stiffness onto its two endpoint nodes.
    for e in range(ei.shape[1]):
        stiff[ei[0, e]] += es[e]
        stiff[ei[1, e]] += es[e]
    stiff = stiff / 2.0

    rng = np.random.default_rng(seed)
    base = np.zeros((n, 3), dtype=np.float32)
    base[:, 0] = np.arange(n) * _BP_RISE_NM            # axial
    base[:, 1] = helix * _HELIX_PITCH_NM               # transverse (lattice row)
    base[:, 2] = (helix % 2) * (_HELIX_PITCH_NM / 2)   # honeycomb stagger

    ens = np.empty((n_ensemble, n, 3), dtype=np.float32)
    amp = 0.5 / np.sqrt(stiff)[:, None]                # softer node -> larger jitter
    for m in range(n_ensemble):
        ens[m] = base + amp * rng.standard_normal((n, 3)).astype(np.float32)
    return ens


def _metrics(coords, rmsf):
    import numpy as np
    c = np.asarray(coords)
    centered = c - c.mean(axis=0, keepdims=True)
    rg = float(np.sqrt((centered ** 2).sum(axis=1).mean()))
    extent = float(np.linalg.norm(c.max(0) - c.min(0)))
    return rg, extent


def _kabsch_rmsd(a, b):
    """RMSD between two (N,3) point sets after optimal rigid alignment."""
    import numpy as np
    a = np.asarray(a); b = np.asarray(b)
    if a.shape != b.shape:
        return None
    ac = a - a.mean(0); bc = b - b.mean(0)
    h = ac.T @ bc
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1, 1, d]) @ u.T
    a_rot = ac @ r.T
    return float(np.sqrt(((a_rot - bc) ** 2).sum(axis=1).mean()))


class ShapePredictor:
    """Predicts a ShapeResult from a ContactMap via a DGNN backend.

    backend: object exposing .predict(graph: dict, n_ensemble: int) -> ndarray
        of shape (n_ensemble, N, 3). Pass the client from
        md/gnn_shape_modal.py, or None to use the deterministic fallback.
    Mirrors Evo2Mutator's contract: any backend error degrades to the
    fallback rather than raising mid-analysis.
    """

    def __init__(self, backend=None, n_ensemble: int = 8):
        self.backend = backend
        self.n_ensemble = n_ensemble

    def predict(self, cm: ContactMap, target_coords=None, seed: int = 0) -> ShapeResult:
        import numpy as np
        graph = build_graph(cm)
        used = "fallback"
        ens = None
        if self.backend is not None:
            try:
                ens = np.asarray(self.backend.predict(graph, self.n_ensemble))
                used = type(self.backend).__name__
            except Exception:
                ens = None
        if ens is None:
            ens = _fallback_predict(graph, self.n_ensemble, seed=seed)

        coords = ens.mean(axis=0)
        rmsf = np.sqrt(((ens - coords[None]) ** 2).sum(axis=2).mean(axis=0))
        rg, extent = _metrics(coords, rmsf)
        rmsd = _kabsch_rmsd(coords, target_coords) if target_coords is not None else None
        return ShapeResult(
            coords=coords, rmsf=rmsf, n_nodes=graph["n_nodes"],
            radius_of_gyration=rg, max_extent=extent,
            mean_rmsf=float(rmsf.mean()), max_rmsf=float(rmsf.max()),
            shape_rmsd_to_target=rmsd, backend=used,
        )


def predict_shape(cm: ContactMap, backend=None, n_ensemble: int = 8,
                  target_coords=None, seed: int = 0) -> ShapeResult:
    """Convenience wrapper. Returns a ShapeResult (see .summary())."""
    return ShapePredictor(backend=backend, n_ensemble=n_ensemble).predict(
        cm, target_coords=target_coords, seed=seed)
