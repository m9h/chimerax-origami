"""Geometric forward-model (DGNN shape predictor) tests.

GPU-free: the fallback path and a deterministic FakeDGNN backend exercise
the whole build_graph -> predict -> metrics wiring, mirroring how the Evo2
tests use a FakeEvo2. No torch / torch_geometric / Modal involved.
"""

import numpy as np

from src.contactmap import ContactMap, reverse_complement
from src import shape as sh


def _design():
    seg = "GGGGCCCCAAAATTTT"
    scaffold = (seg + "ACAC" + reverse_complement(seg)) * 3
    return ContactMap(scaffold=scaffold, staples=["TTTTGGGGCCCCAAAA"], name="shape_seed")


def test_build_graph_matches_contact_map():
    cm = _design()
    g = sh.build_graph(cm)
    assert g["n_nodes"] == len(cm.scaffold)
    # backbone edges = n-1, each carrying a stiffness feature.
    assert g["edge_index"].shape == (2, len(cm.scaffold) - 1)
    assert g["edge_stiffness"].shape[0] == len(cm.scaffold) - 1


def test_fallback_predict_is_deterministic_and_shaped():
    cm = _design()
    r1 = sh.predict_shape(cm, backend=None, n_ensemble=8, seed=3)
    r2 = sh.predict_shape(cm, backend=None, n_ensemble=8, seed=3)
    assert r1.backend == "fallback"
    assert r1.n_nodes == len(cm.scaffold)
    assert r1.coords.shape == (len(cm.scaffold), 3)
    assert r1.radius_of_gyration > 0
    # deterministic under fixed seed
    assert np.allclose(r1.coords, r2.coords)
    assert r1.max_rmsf >= r1.mean_rmsf > 0


def test_flexibility_is_sequence_dependent():
    # A soft (TA-rich) scaffold should be floppier than a stiff (GC) one.
    soft = ContactMap(scaffold="TATATATATATATATATATATATA", name="soft")
    stiff = ContactMap(scaffold="GCGCGCGCGCGCGCGCGCGCGCGC", name="stiff")
    rs = sh.predict_shape(soft, n_ensemble=16, seed=0)
    rg = sh.predict_shape(stiff, n_ensemble=16, seed=0)
    assert rs.mean_rmsf > rg.mean_rmsf


def test_kabsch_rmsd_zero_against_self():
    cm = _design()
    r = sh.predict_shape(cm, n_ensemble=4, seed=1)
    rmsd = sh.predict_shape(cm, n_ensemble=4, seed=1,
                            target_coords=r.coords).shape_rmsd_to_target
    assert rmsd is not None and rmsd < 1e-3


class FakeDGNN:
    """Deterministic stand-in: returns a straight bundle plus seeded noise."""

    def predict(self, graph, n_ensemble):
        n = graph["n_nodes"]
        rng = np.random.default_rng(0)
        base = np.zeros((n, 3), dtype=np.float32)
        base[:, 0] = np.arange(n) * 0.34
        return base[None] + 0.1 * rng.standard_normal((n_ensemble, n, 3)).astype(np.float32)


def test_backend_is_used_when_provided():
    cm = _design()
    r = sh.predict_shape(cm, backend=FakeDGNN(), n_ensemble=6)
    assert r.backend == "FakeDGNN"
    assert r.n_nodes == len(cm.scaffold)


def test_broken_backend_falls_back():
    class Broken:
        def predict(self, graph, n_ensemble):
            raise RuntimeError("simulated CUDA OOM")

    cm = _design()
    r = sh.predict_shape(cm, backend=Broken(), n_ensemble=4)
    assert r.backend == "fallback"          # degraded, did not raise
    assert r.radius_of_gyration > 0


def test_summary_is_json_shaped():
    cm = _design()
    s = sh.predict_shape(cm, n_ensemble=4).summary()
    assert set(s) >= {"backend", "n_nodes", "radius_of_gyration_nm",
                      "mean_rmsf_nm", "flexible_hotspots"}
    assert isinstance(s["flexible_hotspots"], list)
