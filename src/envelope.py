"""Lipid-bilayer envelope modeling for chimerax-origami — the
"enveloped delivery vehicle" module.

Structural twin of chimerax-vampnet's animate.py. animate.py builds a
*derived* model (an interpolation along a slow mode) and adds it to the
session; this module builds a *derived* model too — the lipid-conjugate
handle placement + a wrapping bilayer shell that turns a bare DNA-origami
nanostructure into a virus-like enveloped particle.

Scientific basis — Perrault & Shih, "Virus-Inspired Membrane Encapsulation
of DNA Nanostructures To Achieve In Vivo Stability" (ACS Nano 2014):
lipid-DNA conjugates annealed to *outer* staple handles direct the assembly
of a single bilayer around the structure. Tight unilamellar wrapping needs
a controlled handle density of ~1 conjugate per ~180 nm^2. Envelopment in a
PEGylated bilayer conferred nuclease protection, dropped immune activation
~100x, and improved pharmacokinetic bioavailability ~17x.

WHY THIS LIVES NEXT TO chimerax-vampnet
The envelope is a *perturbation that reshapes the fate landscape* of the
particle — exactly as the anti-NRR antibody is a perturbation that reshapes
Notch1's conformational landscape in the vampnet demo. And the target the
vampnet bundle picked, Notch1 NRR, is a *membrane* receptor whose v0.3 MD
needed a COM-distance restraint to a membrane anchor. Both projects meet at
the same lipid-bilayer interface; see CONNECTIONS.md.
"""

from __future__ import annotations

import math
from typing import List

from .contactmap import ContactMap


# Perrault & Shih target handle density for tight unilamellar wrapping.
_DEFAULT_HANDLE_NM2 = 180.0

# Geometry constants for the back-of-envelope surface estimate.
_BP_RISE_NM = 0.34          # nm per base pair along a helix
_HELIX_DIAM_NM = 2.0        # B-form duplex diameter
_BILAYER_THICKNESS_NM = 4.0  # ~ a PEGylated DOPC bilayer


def _estimate_surface_area_nm2(cm: ContactMap) -> float:
    """Crude solvent-accessible outer-surface estimate from the contact map.

    With no 3D model we approximate the folded brick as a sphere of equal
    duplex volume, then return its surface area. This is intentionally rough
    — when a real oxDNA/cadnano 3D model is bound we should integrate the
    actual molecular surface instead (TODO v0.2).
    """
    n_bp = max(len(cm.scaffold), 1)
    helix_volume_nm3 = n_bp * _BP_RISE_NM * math.pi * (_HELIX_DIAM_NM / 2) ** 2
    r = (3 * helix_volume_nm3 / (4 * math.pi)) ** (1 / 3)
    return 4 * math.pi * r * r, r


def design_envelope(session, cm: ContactMap, handle_density_nm2: float = _DEFAULT_HANDLE_NM2,
                    structure=None) -> dict:
    """Plan a lipid-bilayer envelope for the active design.

    Computes the outer surface area, the number of lipid-conjugate handles
    needed at the target density, picks candidate outer staples to extend
    with handles, and (when a 3D model is present) adds a translucent shell
    surface representing the wrapping bilayer.

    Returns a JSON-serializable plan.
    """
    area_nm2, radius_nm = _estimate_surface_area_nm2(cm)
    n_handles = max(1, round(area_nm2 / handle_density_nm2))

    # Choose handle-carrier staples: in a real importer these are the staples
    # whose 5'/3' ends lie on the *outer* lattice face. Without lattice
    # geometry we deterministically sample evenly across the staple list so
    # the count and spacing are right even if the exact staples are
    # placeholders. TODO(v0.2): select by outer-face geometry.
    n_staples = len(cm.staples)
    if n_staples:
        step = max(1, n_staples // n_handles)
        carriers = list(range(0, n_staples, step))[:n_handles]
    else:
        carriers = []

    shell_added = False
    if structure is not None:
        shell_added = _add_bilayer_shell(session, structure, radius_nm)

    return {
        "surface_area_nm2": round(area_nm2, 1),
        "approx_radius_nm": round(radius_nm, 2),
        "handle_density_nm2": handle_density_nm2,
        "n_lipid_handles": n_handles,
        "n_carrier_staples": len(carriers),
        "carrier_staple_indices": carriers,
        "bilayer_thickness_nm": _BILAYER_THICKNESS_NM,
        "shell_model_added": shell_added,
        "predicted_effects": {
            # Order-of-magnitude figures from Perrault & Shih 2014, surfaced
            # so an MCP agent can reason about the delivery trade-off.
            "nuclease_protection": "conferred (PEGylated bilayer)",
            "immune_activation_fold_change": 0.01,
            "pharmacokinetic_bioavailability_fold_change": 17.0,
        },
        "reference": "Perrault & Shih, ACS Nano 2014 (10.1021/nn5011914)",
    }


def _add_bilayer_shell(session, structure, radius_nm: float) -> bool:
    """Add a translucent spherical shell ~radius+bilayer around the model
    centroid as a stand-in for the wrapping bilayer. Replace with a true
    offset molecular surface once a 3D model loader lands.
    """
    try:
        import numpy as np
        from chimerax.surface import sphere_geometry2
        from chimerax.core.models import Surface

        atoms = structure.atoms
        center = atoms.coords.mean(axis=0)
        # nm -> Angstrom for ChimeraX coordinate space.
        r_ang = (radius_nm + _BILAYER_THICKNESS_NM) * 10.0
        va, na, ta = sphere_geometry2(2000)
        va = va * r_ang + center
        shell = Surface("lipid envelope", session)
        shell.set_geometry(va, na, ta)
        shell.color = (120, 170, 255, 90)  # translucent blue
        session.models.add([shell])
        return True
    except Exception:
        # ChimeraX surface API not available (headless test context).
        return False
