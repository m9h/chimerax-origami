"""Assembly-MSM bridge tests.

The synthetic folding trajectory has a KNOWN 3-state structure (unfolded ->
trap -> folded), so the whole featurize -> cluster -> MSM -> transition-graph
pipeline has a ground truth to recover — the assembly analog of vampnet's
synthetic Markov-chain unit test. Uses the numpy backend so CI needs no
deeptime/torch.
"""

import numpy as np

from src import assembly as asm
from src.contactmap import ContactMap


def test_synthetic_trajectory_has_three_regimes():
    feats, states = asm.synthetic_assembly_trajectory(n_frames=1500, n_pairs=24, seed=0)
    assert feats.shape == (1500, 24)
    assert set(np.unique(states)) <= {0, 1, 2}
    # folded frames carry far more formed contacts than unfolded ones.
    assert feats[states == 2].mean() > feats[states == 0].mean() + 0.4


def test_msm_recovers_states_and_is_stochastic():
    feats, _ = asm.synthetic_assembly_trajectory(n_frames=3000, n_pairs=30, seed=1)
    msm = asm.fit_assembly_msm(feats, lag=5, n_states=3, backend="numpy", seed=1)
    assert msm.n_states == 3
    T = np.asarray(msm.transition_matrix)
    assert np.allclose(T.sum(axis=1), 1.0, atol=1e-6)      # row-stochastic
    assert np.all(np.asarray(msm.stationary_distribution) >= -1e-9)
    assert abs(sum(msm.stationary_distribution) - 1.0) < 1e-6


def test_labels_span_unfolded_to_folded():
    feats, _ = asm.synthetic_assembly_trajectory(n_frames=3000, n_pairs=30, seed=2)
    msm = asm.fit_assembly_msm(feats, lag=5, n_states=3, backend="numpy", seed=2)
    labels = msm.state_labels()
    assert "unfolded" in labels and "folded" in labels


def test_trap_state_is_identified():
    feats, _ = asm.synthetic_assembly_trajectory(n_frames=4000, n_pairs=30, seed=3)
    msm = asm.fit_assembly_msm(feats, lag=5, n_states=3, backend="numpy", seed=3)
    traps = msm.identify_traps(residence=0.4)
    # the middle (trap) state should be metastable and flagged.
    assert len(traps) >= 1
    for t in traps:
        assert msm.state_labels()[t] == "intermediate"


def test_transition_graph_matches_vampnet_contract():
    feats, _ = asm.synthetic_assembly_trajectory(n_frames=2000, n_pairs=24, seed=4)
    msm = asm.fit_assembly_msm(feats, lag=5, n_states=3, backend="numpy", seed=4)
    g = msm.transition_graph()
    # Same top-level keys as chimerax-vampnet/src/msm.py::transition_graph.
    assert set(g) >= {"states", "transition_matrix", "stationary_distribution",
                      "lag", "nodes", "edges", "edge_density"}
    assert len(g["nodes"]) == 3
    assert all({"id", "stationary", "label", "is_trap"} <= set(n) for n in g["nodes"])


def test_implied_timescales_present_and_positive():
    feats, _ = asm.synthetic_assembly_trajectory(n_frames=3000, n_pairs=30, seed=5)
    msm = asm.fit_assembly_msm(feats, lag=5, n_states=3, backend="numpy", seed=5)
    ts = msm.implied_timescales()
    assert all(t > 0 for t in ts)


def test_featurize_assembly_from_coords():
    # Two frames, 4 nucleotides; pair (0,1) close in frame 0, far in frame 1.
    coords = np.array([
        [[0, 0, 0], [0.5, 0, 0], [10, 0, 0], [10.5, 0, 0]],
        [[0, 0, 0], [9.0, 0, 0], [10, 0, 0], [10.5, 0, 0]],
    ], dtype=np.float32)
    occ = asm.featurize_assembly(coords, pairs=[(0, 1), (2, 3)], cutoff=2.0)
    assert occ.shape == (2, 2)
    assert occ[0, 0] == 1.0 and occ[1, 0] == 0.0   # pair (0,1) breaks
    assert occ[0, 1] == 1.0 and occ[1, 1] == 1.0   # pair (2,3) stays formed


def test_pairs_from_design():
    cm = ContactMap(scaffold="ACGT", intended_pairs=[("sc-st", 0, 5, 1, 12)])
    assert asm.pairs_from_design(cm) == [(5, 12)]
