"""End-to-end validation vs. measured wet-lab data (VALIDATION.md test 6).

The capstone. Uses the REAL scaffold variants from the Krasnogor paper
(Zenodo 14748478): three triangle DNA-origami designs sharing one routing
with different scaffold sequences. Our INDEPENDENT off-target scorer is
correlated against the paper's single-molecule optical-tweezers measurement
of structural non-uniformity (Fig. 7D; 379-742 molecules per variant),
keyed by the supplementary's blinded-name mapping:
    T1 = DEER, T2 = LION, T3 = BEAR.

Claim under test: lower predicted off-target -> lower measured non-uniformity
(more uniform, better-folded structures).
"""

import csv
import os

from src import contactmap as cmap
from src import score as sc

DATA = os.path.join(os.path.dirname(__file__), "..", "examples", "data")
VARIANTS = os.path.join(DATA, "krasnogor_variants")


def _measured():
    rows = {}
    with open(os.path.join(DATA, "krasnogor_measured.csv")) as f:
        for r in csv.DictReader(f):
            rows[r["codename"]] = r
    return rows


def _our_offtarget(codename):
    cm = cmap._load_scadnano(os.path.join(VARIANTS, codename + ".sc"))
    return sc.score(cm, k=8).objectives["total"]


def test_designs_load_and_share_routing():
    # the three triangle variants share scaffold length (same routing).
    for cn in ("DEER", "LION", "BEAR"):
        cm = cmap._load_scadnano(os.path.join(VARIANTS, cn + ".sc"))
        assert len(cm.scaffold) == 2410
        assert len(cm.staples) > 0


def test_offtarget_predicts_measured_non_uniformity():
    meas = _measured()
    triangle = ["DEER", "LION", "BEAR"]   # T1, T2, T3
    our = {cn: _our_offtarget(cn) for cn in triangle}
    nonunif = {cn: float(meas[cn]["non_uniformity"]) for cn in triangle}

    # rank order by our prediction must equal rank order by measured non-uniformity.
    our_order = sorted(triangle, key=lambda c: our[c])
    meas_order = sorted(triangle, key=lambda c: nonunif[c])
    assert our_order == meas_order == ["DEER", "LION", "BEAR"]   # T1 < T2 < T3

    # and our scores are strictly monotone with the measurement.
    vals = [(our[c], nonunif[c]) for c in our_order]
    assert vals[0][0] < vals[1][0] < vals[2][0]
    assert vals[0][1] < vals[1][1] < vals[2][1]


def test_scores_reproduce_documented_values():
    # pin the headline numbers (k=8) so the result is reproducible.
    our = {cn: round(_our_offtarget(cn)) for cn in ("DEER", "LION", "BEAR")}
    assert our["DEER"] < our["LION"] < our["BEAR"]
    # documented in examples/yield_validation_demo.py output.
    assert 1000 < our["DEER"] < 1250
    assert 1400 < our["BEAR"] < 1700
