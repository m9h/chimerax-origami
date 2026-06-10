"""VAMPnet / Markov-state modeling of DNA-origami ASSEMBLY trajectories —
the method-level bridge to chimerax-vampnet.

This is the module where the two bundles stop being analogies and start
sharing a pipeline. chimerax-vampnet learns metastable *protein conformational*
states from MD by: featurize (CA contact map) -> VAMPnet/MSM -> transition
graph. Here we run the identical pipeline on an **oxDNA folding trajectory**:

    featurize (base-pair contact OCCUPANCY) -> VAMPnet/MSM -> transition graph

and recover the metastable *folding intermediates* of the origami. Off-target
interactions (score.py's four classes) then appear here as what they really
are kinetically: **metastable trap states** on the assembly landscape — the
nucleic-acid analog of a misfolded protein basin in the vampnet MSM.

WHAT IS SHARED, CONCRETELY
  - The feature is a contact map, exactly as in vampnet.featurize: vampnet
    uses CA-CA distances; we use whether each *intended* base pair is formed
    (distance below a cutoff) per frame -> an (n_frames, n_pairs) occupancy
    matrix. Same shape, same role.
  - The estimator is the same library vampnet wraps (deeptime): cluster the
    featurized frames into microstates, estimate a Markov state model, read
    off metastable states + implied timescales. When deeptime/torch are
    present we use them (the literal vampnet machinery); otherwise a compact
    numpy fallback (k-means + count-matrix MSM) runs the same logic dep-free
    so the pipeline is always exercisable (mirrors vampnet's synthetic
    Markov-chain unit test).
  - The OUTPUT contract is byte-for-byte vampnet's msm.transition_graph:
    {states, transition_matrix, stationary_distribution, lag, nodes, edges,
     edge_density}. An MCP agent or downstream tool consumes an assembly MSM
    and a protein MSM through the same code path.

The trajectory itself is produced by an oxDNA folding simulation
(md/oxdna_modal.py), the data-generation analog of vampnet's md/*_modal.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


# ----------------------------------------------------------------------
# Featurization — the contact map becomes the VAMPnet feature.
# ----------------------------------------------------------------------
def featurize_assembly(coords, pairs, cutoff: float = 2.0):
    """oxDNA trajectory coords (F, N, 3) + intended base pairs -> occupancy.

    pairs: list of (i, j) nucleotide indices that are *designed* to hybridize
        (from the contact map's intended_pairs). cutoff is in the coords' own
        length units (oxDNA simulation units ~0.85 nm; paired nucleotides sit
        well within ~1 unit). Returns an (F, P) float matrix where entry (t,p)
        is 1.0 if pair p is formed in frame t, else 0.0 — the assembly analog
        of vampnet's standardized CA-distance feature.
    """
    import numpy as np
    c = np.asarray(coords)
    ii = np.array([p[0] for p in pairs], dtype=np.int64)
    jj = np.array([p[1] for p in pairs], dtype=np.int64)
    d = np.sqrt(((c[:, ii, :] - c[:, jj, :]) ** 2).sum(-1))
    return (d < cutoff).astype(np.float32)


def pairs_from_design(cm) -> List[Tuple[int, int]]:
    """Extract (i, j) nucleotide-index pairs from a ContactMap's
    intended_pairs. Each intended pair tuple is (kind, a_strand, a_idx,
    b_strand, b_idx); for a single-scaffold occupancy feature we use the
    flat base indices the caller has already resolved. Routing-only designs
    (no intended_pairs) return [] — supply occupancy directly instead.
    """
    pairs = []
    for p in (cm.intended_pairs or []):
        if len(p) >= 5:
            pairs.append((int(p[2]), int(p[4])))
        elif len(p) == 2:
            pairs.append((int(p[0]), int(p[1])))
    return pairs


# ----------------------------------------------------------------------
# The MSM object — mirrors the role of vampnet's fitted model.
# ----------------------------------------------------------------------
@dataclass
class AssemblyMSM:
    transition_matrix: object        # (K, K) row-stochastic
    stationary_distribution: object  # (K,)
    dtraj: object                    # (F,) microstate-per-frame
    frac_folded: List[float]         # mean occupancy per state (folding progress)
    lag: int
    backend: str

    @property
    def n_states(self) -> int:
        return len(self.stationary_distribution)

    def implied_timescales(self) -> List[float]:
        import numpy as np
        ev = np.linalg.eigvals(np.asarray(self.transition_matrix))
        ev = np.sort(np.real(ev))[::-1]
        ts = []
        for lam in ev[1:]:  # skip the stationary eigenvalue (~1)
            if 0 < lam < 1:
                ts.append(float(-self.lag / np.log(lam)))
        return ts

    def state_labels(self) -> List[str]:
        """Name states by folding progress: lowest occupancy = 'unfolded',
        highest = 'folded', the rest 'intermediate'."""
        order = sorted(range(self.n_states), key=lambda s: self.frac_folded[s])
        labels = ["intermediate"] * self.n_states
        labels[order[0]] = "unfolded"
        labels[order[-1]] = "folded"
        return labels

    def identify_traps(self, residence: float = 0.5) -> List[int]:
        """Kinetic traps = metastable states (high self-residence T[i,i])
        that are NOT the productive folded state and NOT the fully-unfolded
        entry state. These are the off-target basins — a partially/mis-folded
        configuration the system dwells in, the assembly-landscape analog of a
        misfolded protein metastable state.
        """
        labels = self.state_labels()
        T = self.transition_matrix
        traps = []
        for i in range(self.n_states):
            if labels[i] in ("folded", "unfolded"):
                continue
            if float(T[i][i]) >= residence:
                traps.append(i)
        return traps

    def transition_graph(self) -> dict:
        """Identical shape to chimerax-vampnet/src/msm.py::transition_graph,
        extended with per-node folding labels + trap flags so an MCP client
        consumes assembly and protein MSMs through one code path.
        """
        K = self.n_states
        T = [[float(x) for x in row] for row in self.transition_matrix]
        pi = [float(x) for x in self.stationary_distribution]
        labels = self.state_labels()
        traps = set(self.identify_traps())
        nodes = [
            {"id": s, "stationary": pi[s], "frac_folded": float(self.frac_folded[s]),
             "label": labels[s], "is_trap": s in traps}
            for s in range(K)
        ]
        edges = []
        for i in range(K):
            for j in range(K):
                if i != j and T[i][j] > 0.0:
                    edges.append({"src": i, "dst": j, "rate": T[i][j]})
        max_edges = K * (K - 1)
        return {
            "states": list(range(K)),
            "transition_matrix": T,
            "stationary_distribution": pi,
            "lag": int(self.lag),
            "nodes": nodes,
            "edges": edges,
            "edge_density": (len(edges) / max_edges) if max_edges else 0.0,
        }

    def summary(self) -> dict:
        labels = self.state_labels()
        traps = self.identify_traps()
        return {
            "backend": self.backend,
            "n_states": self.n_states,
            "lag": self.lag,
            "implied_timescales": self.implied_timescales(),
            "state_labels": labels,
            "frac_folded": [round(float(x), 3) for x in self.frac_folded],
            "stationary_distribution": [round(float(x), 3) for x in self.stationary_distribution],
            "n_traps": len(traps),
            "trap_states": traps,
        }


# ----------------------------------------------------------------------
# Estimation — deeptime (the vampnet machinery) if present, else numpy.
# ----------------------------------------------------------------------
def _kmeans(X, k, seed=0, iters=100):
    """Tiny seeded Lloyd's k-means with k-means++ seeding (numpy fallback for
    deeptime.clustering). X should be standardized by the caller so no single
    high-population, low-variance basin captures multiple centroids."""
    import numpy as np
    rng = np.random.default_rng(seed)
    # k-means++ seeding: first centroid random, each next farther from chosen.
    C = [X[rng.integers(0, len(X))]]
    for _ in range(1, k):
        d2 = np.min(((X[:, None, :] - np.asarray(C)[None, :, :]) ** 2).sum(-1), axis=1)
        probs = d2 / (d2.sum() + 1e-12)
        C.append(X[rng.choice(len(X), p=probs)])
    C = np.asarray(C, dtype=np.float64)
    labels = np.full(len(X), -1, dtype=np.int64)
    for _ in range(iters):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        new = d.argmin(1)
        if np.array_equal(new, labels):
            break
        labels = new
        for c in range(k):
            m = labels == c
            if m.any():
                C[c] = X[m].mean(0)
    return labels, C


def _count_msm(dtraj, k, lag):
    """Count-matrix MSM at the given lag (numpy fallback for deeptime.markov)."""
    import numpy as np
    C = np.zeros((k, k), dtype=np.float64)
    for t in range(len(dtraj) - lag):
        C[dtraj[t], dtraj[t + lag]] += 1.0
    C += 1e-9  # regularize unseen transitions
    T = C / C.sum(1, keepdims=True)
    # Stationary distribution = normalized left eigenvector for eigenvalue 1.
    ev, evec = np.linalg.eig(T.T)
    pi = np.real(evec[:, np.argmin(np.abs(ev - 1.0))])
    pi = np.abs(pi)
    pi = pi / pi.sum()
    return T, pi


def fit_assembly_msm(features, lag: int = 5, n_states: int = 3,
                     backend: str = "auto", seed: int = 0) -> AssemblyMSM:
    """Cluster featurized assembly frames and estimate a Markov state model.

    backend:
        "auto"     use deeptime (the chimerax-vampnet machinery) if importable,
                   else the numpy fallback.
        "deeptime" force deeptime (raises if unavailable).
        "numpy"    force the dependency-free fallback.

    Returns an AssemblyMSM. features is (F, P) occupancy from
    featurize_assembly (or a synthetic generator).
    """
    import numpy as np
    X = np.asarray(features, dtype=np.float64)

    use_deeptime = backend == "deeptime"
    if backend == "auto":
        try:
            import deeptime  # noqa: F401
            use_deeptime = True
        except Exception:
            use_deeptime = False

    if use_deeptime:
        # The literal vampnet path: deeptime clustering + MSM. Kept import-
        # local so the bundle/tests never require deeptime.
        from deeptime.clustering import KMeans
        from deeptime.markov.msm import MaximumLikelihoodMSM
        est = KMeans(n_clusters=n_states, fixed_seed=seed).fit_fetch(X)
        dtraj = est.transform(X)
        msm = MaximumLikelihoodMSM(lagtime=lag).fit_fetch(dtraj)
        T = np.asarray(msm.transition_matrix)
        # Re-map to the active state set (deeptime may drop unconnected states).
        K = T.shape[0]
        pi = np.asarray(msm.stationary_distribution)
        used = "deeptime"
        # Recompute per-state folding fraction over the dtraj.
        frac = [float(X[dtraj == s].mean()) if (dtraj == s).any() else 0.0
                for s in range(K)]
        return AssemblyMSM(T, pi, dtraj, frac, lag, used)

    # numpy fallback. Standardize per contact (z-score) before clustering so
    # the discriminating wrong-register contacts get weight and the dominant
    # folded basin doesn't absorb extra centroids — vampnet standardizes its
    # CA-distance features for the same reason.
    Xz = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-6)
    dtraj, _ = _kmeans(Xz, n_states, seed=seed)
    T, pi = _count_msm(dtraj, n_states, lag)
    frac = [float(X[dtraj == s].mean()) if (dtraj == s).any() else 0.0
            for s in range(n_states)]
    return AssemblyMSM(T, pi, dtraj, frac, lag, "numpy")


# ----------------------------------------------------------------------
# Trajectory loading + a synthetic generator for tests/demo.
# ----------------------------------------------------------------------
def load_trajectory(path: str):
    """Load an assembly trajectory. Supports:
      - .npz with 'occupancy' (F, P)            -> returns ("occupancy", arr)
      - .npz with 'coords' (F, N, 3) [+ 'pairs']-> returns ("coords", arr, pairs)
    A real oxDNA .dat/.conf parser is a v0.2 TODO (read per-configuration
    nucleotide positions); md/oxdna_modal.py emits the .npz directly.
    """
    import numpy as np
    if not path.lower().endswith(".npz"):
        raise ValueError("v0.1 expects an .npz trajectory; oxDNA .dat parser is TODO")
    d = np.load(path, allow_pickle=True)
    if "occupancy" in d.files:
        return ("occupancy", d["occupancy"])
    if "coords" in d.files:
        pairs = d["pairs"].tolist() if "pairs" in d.files else None
        return ("coords", d["coords"], pairs)
    raise ValueError("npz must contain 'occupancy' or 'coords'")


def synthetic_assembly_trajectory(n_frames: int = 2000, n_pairs: int = 30,
                                  seed: int = 0, noise: float = 0.08):
    """Generate a toy oxDNA-like folding trajectory with KNOWN metastable
    states, the assembly analog of vampnet's synthetic 4-state Markov chain
    test. A hidden 3-state chain (unfolded -> trap -> folded, with the trap as
    a kinetic dead-end the system must back out of) emits a noisy contact-
    occupancy vector per frame:

        unfolded : ~5%  of intended pairs formed
        trap     : ~55% formed, but partly on the WRONG register
        folded   : ~95% of intended pairs formed

    Returns (features (F, P), true_states (F,)). A correct MSM recovers the
    three states and flags the trap.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    # Hidden transition matrix: trap is metastable (high self-residence) and
    # mostly exits back toward unfolded, only slowly to folded.
    P = np.array([
        [0.90, 0.08, 0.02],   # unfolded
        [0.10, 0.85, 0.05],   # trap (sticky)
        [0.01, 0.02, 0.97],   # folded (absorbing-ish)
    ])
    means = np.array([0.05, 0.55, 0.95])
    # The trap forms a different *register*: shuffle which pairs it satisfies.
    trap_mask = rng.random(n_pairs) < 0.55
    states = np.zeros(n_frames, dtype=np.int64)
    s = 0
    feats = np.zeros((n_frames, n_pairs), dtype=np.float32)
    for t in range(n_frames):
        states[t] = s
        if s == 1:
            base = np.where(trap_mask, 0.9, 0.1)        # wrong-register contacts
        else:
            base = np.full(n_pairs, means[s])
        feats[t] = (rng.random(n_pairs) < base).astype(np.float32)
        feats[t] += noise * rng.standard_normal(n_pairs)
        s = rng.choice(3, p=P[s])
    return feats, states
