"""Envelope validation vs. Perrault & Shih, ACS Nano 2014 (VALIDATION.md
test 5).

Validates the lipid-bilayer envelope planner against the paper's reported
geometry (1 conjugate / ~180 nm^2) and in-vivo effect figures (nuclease
protection, ~100x lower immune activation, ~17x bioavailability), plus the
core handle-count arithmetic and the documented solid-sphere undercount.
"""

from src import envelope as env
from src.contactmap import ContactMap


def _plan(cm, **kw):
    # session is only used for the 3D shell (structure=None here), so None is fine.
    return env.design_envelope(None, cm, **kw)


def test_default_density_matches_paper():
    assert env._DEFAULT_HANDLE_NM2 == 180.0      # Perrault & Shih target density


def test_handle_count_follows_density_relation():
    cm = ContactMap(scaffold="N" * 7308, staples=["N" * 32] * 200)
    # override with the octahedron's ~50 nm enclosing surface (4*pi*25^2).
    area = 7854.0
    plan = _plan(cm, surface_area_nm2=area)
    assert plan["surface_area_nm2"] == round(area, 1)
    assert plan["n_lipid_handles"] == round(area / 180.0)     # exact relation
    # a ~50 nm particle at 1/180 nm^2 needs a few dozen handles.
    assert 30 <= plan["n_lipid_handles"] <= 60


def test_doubling_density_halves_handles():
    cm = ContactMap(scaffold="N" * 5000, staples=["N" * 32] * 150)
    n1 = _plan(cm, handle_density_nm2=180.0, surface_area_nm2=7200.0)["n_lipid_handles"]
    n2 = _plan(cm, handle_density_nm2=360.0, surface_area_nm2=7200.0)["n_lipid_handles"]
    assert n1 == 2 * n2


def test_reported_invivo_effects_match_paper():
    plan = _plan(ContactMap(scaffold="N" * 7308, staples=["N" * 32] * 200))
    eff = plan["predicted_effects"]
    assert eff["pharmacokinetic_bioavailability_fold_change"] == 17.0   # ~17x
    assert eff["immune_activation_fold_change"] == 0.01                 # ~100x lower
    assert "conferred" in eff["nuclease_protection"].lower()   # nuclease protection
    assert "Perrault" in plan["reference"]


def test_solid_sphere_estimate_undercounts_hollow_wireframe():
    # The solid-sphere area for the octahedron's DNA volume is several-fold
    # smaller than its true ~50 nm enclosing surface -> the estimate is a
    # documented lower bound; pass a real area for quantitative counts.
    cm = ContactMap(scaffold="N" * 7308, staples=["N" * 32] * 200)
    estimate = _plan(cm)["surface_area_nm2"]
    enclosing = 7854.0
    assert estimate < enclosing                # undercount, as documented
    assert _plan(cm, surface_area_nm2=enclosing)["n_lipid_handles"] > \
        _plan(cm)["n_lipid_handles"]           # real area -> more handles


def test_carrier_staples_are_chosen():
    cm = ContactMap(scaffold="N" * 7308, staples=["N" * 32] * 200)
    plan = _plan(cm, surface_area_nm2=7854.0)
    assert plan["n_carrier_staples"] == plan["n_lipid_handles"]
    assert len(plan["carrier_staple_indices"]) == plan["n_lipid_handles"]
