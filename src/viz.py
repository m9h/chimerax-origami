"""ChimeraX visualization for chimerax-origami.

Twin of chimerax-vampnet's viz.py. Where vampnet's `color_by_state`
recolors trajectory frames by their VAMPnet-assigned metastable state,
here `color_by_frustration` recolors the bases/helices of a loaded
nanostructure by their accumulated off-target (frustration) density —
the same idea (map a learned per-element scalar onto the 3D model) applied
to assembly frustration instead of conformational state.
"""

from __future__ import annotations

from typing import Dict, List


# Blue (cold / no frustration) -> red (hot / many off-target traps).
def _heat_rgba(t: float):
    t = max(0.0, min(1.0, t))
    return (int(255 * t), 0, int(255 * (1 - t)), 255)


def _per_base_frustration(scored) -> Dict[int, float]:
    """Accumulate, per scaffold base index, the weighted off-target events
    touching it. Uses the scaffold-centric channels (j1 staple<->scaffold,
    j2 scaffold<->scaffold) since those map cleanly onto scaffold positions.
    """
    acc: Dict[int, float] = {}
    k = scored.k
    for entry in scored.hotspots.get("j2", []):
        i, j, kk = entry
        for b in range(i, i + kk):
            acc[b] = acc.get(b, 0.0) + 1.0
    for entry in scored.hotspots.get("j1", []):
        # entry == ("staple", staple_seq, pos_staple, pos_scaffold_rc, kk)
        if len(entry) >= 5:
            _, _, _ps, pj, kk = entry[:5]
            for b in range(pj, pj + kk):
                acc[b] = acc.get(b, 0.0) + 1.0
    return acc


def color_by_flexibility(session, shape_result, structure=None) -> dict:
    """Color a nanostructure by per-base-pair flexibility (DGNN RMSF).

    Twin of color_by_frustration, but the per-node scalar is the predicted
    RMSF from src/shape.py rather than the off-target density. Cold = rigid,
    hot = floppy. Returns a JSON-serializable summary.
    """
    rmsf = shape_result.rmsf
    n = shape_result.n_nodes
    hi = float(max(rmsf)) if n else 0.0
    if hi <= 0.0:
        return {"colored": False, "reason": "uniform/zero RMSF", "max_rmsf_nm": 0.0}

    n_painted = 0
    if structure is not None and hasattr(structure, "residues"):
        residues = structure.residues
        for i in range(min(n, len(residues))):
            rgba = _heat_rgba(float(rmsf[i]) / hi)
            for a in residues[i].atoms:
                a.color = rgba
            n_painted += 1
    return {
        "colored": structure is not None,
        "n_nodes": n,
        "n_painted": n_painted,
        "max_rmsf_nm": round(hi, 3),
    }


def color_by_frustration(session, scored, structure=None) -> dict:
    """Color a nanostructure model by off-target frustration density.

    If no ChimeraX structure is bound to the design (routing-only import),
    we still return the per-base frustration profile so an MCP client / the
    HTML report can render it. Returns a JSON-serializable summary.
    """
    profile = _per_base_frustration(scored)
    if not profile:
        return {"colored": False, "reason": "no scaffold-mapped hotspots",
                "max_frustration": 0.0, "n_hot_bases": 0}

    hi = max(profile.values())
    # If a structure is present, paint its residues (one nucleotide ==
    # one residue in ChimeraX nucleic models).
    n_painted = 0
    if structure is not None and hasattr(structure, "residues"):
        residues = structure.residues
        for base_idx, val in profile.items():
            if 0 <= base_idx < len(residues):
                r = residues[base_idx]
                rgba = _heat_rgba(val / hi)
                for a in r.atoms:
                    a.color = rgba
                n_painted += 1

    hot_bases = sorted(profile.items(), key=lambda kv: -kv[1])[:25]
    return {
        "colored": structure is not None,
        "n_hot_bases": len(profile),
        "n_painted": n_painted,
        "max_frustration": hi,
        "top_hotspots": [{"base": b, "frustration": v} for b, v in hot_bases],
    }
