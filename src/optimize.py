"""Multi-objective Pareto selection over candidate scaffold sequences.

Structural twin of chimerax-vampnet's msm.py: msm.py turns a fitted model
into a structured graph (nodes + edges) for the MCP bridge; this module
turns a set of scored candidate scaffolds into a structured Pareto front
(non-dominated set + the trade-off graph) for the same downstream consumer.

The Krasnogor / scaffoldselector strategy: rather than collapse the four
off-target objectives into one weighted sum (which hides trade-offs), keep
the *Pareto-optimal* candidates — those not strictly beaten on every
objective by another candidate — and let the designer pick the trade-off.
This is the assembly-design analog of vampnet's implied-timescales test:
both expose a frontier the user reads to make a downstream choice.
"""

from __future__ import annotations

from typing import List

from .contactmap import ContactMap
from .score import score, ScoredDesign


def dominates(a: List[float], b: List[float]) -> bool:
    """True if objective-vector a Pareto-dominates b (minimization): a is
    no worse on every objective and strictly better on at least one.
    """
    no_worse = all(x <= y for x, y in zip(a, b))
    strictly_better = any(x < y for x, y in zip(a, b))
    return no_worse and strictly_better


def pareto_front(scored: List[ScoredDesign]) -> List[int]:
    """Return indices of the non-dominated candidates."""
    front = []
    vecs = [s.vector() for s in scored]
    for i, vi in enumerate(vecs):
        if not any(j != i and dominates(vecs[j], vi) for j in range(len(vecs))):
            front.append(i)
    return front


def optimize(candidates: List[ContactMap], k: int = 8) -> dict:
    """Score every candidate scaffold and return the Pareto front.

    candidates: a list of ContactMaps differing in scaffold sequence (the
    same origami routing, different sequence realizations — e.g. circular
    permutations of M13, alternative natural scaffolds, or synthetic
    sequences). Returns a JSON-shaped dict mirroring msm.transition_graph:
    a node per candidate + edges marking domination relationships, plus the
    extracted Pareto front and the single best-compromise candidate (min L2
    norm of the normalized objective vector).
    """
    scored = [score(c, k=k) for c in candidates]
    vecs = [s.vector() for s in scored]
    front = set(pareto_front(scored))

    # Normalize objectives column-wise for the compromise pick + node coords.
    n_obj = len(vecs[0]) if vecs else 0
    maxes = [max((v[o] for v in vecs), default=1.0) or 1.0 for o in range(n_obj)]
    norm = [[v[o] / maxes[o] for o in range(n_obj)] for v in vecs]

    def l2(v):
        return sum(x * x for x in v) ** 0.5

    best_i = min(range(len(scored)), key=lambda i: l2(norm[i])) if scored else None

    nodes = [
        {
            "id": i,
            "name": scored[i].cm.name or f"candidate_{i}",
            "objectives": scored[i].vector(),
            "total": scored[i].objectives["total"],
            "on_pareto_front": i in front,
            "is_best_compromise": (i == best_i),
        }
        for i in range(len(scored))
    ]
    edges = [
        {"src": j, "dst": i, "relation": "dominates"}
        for i in range(len(scored))
        for j in range(len(scored))
        if i != j and dominates(vecs[j], vecs[i])
    ]

    return {
        "n_candidates": len(scored),
        "objective_names": [
            "j1_staple_wrong_scaffold",
            "j2_scaffold_scaffold",
            "j3_staple_staple",
            "j4_staple_hairpin",
        ],
        "pareto_front": sorted(front),
        "best_compromise": best_i,
        "nodes": nodes,
        "edges": edges,
    }
