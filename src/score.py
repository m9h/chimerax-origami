"""Off-target interaction scoring for chimerax-origami.

This is the structural twin of chimerax-vampnet's vampnet_core.py: where
vampnet fits a model that turns features into a low-dimensional landscape
(states + scores), this module turns a contact map into a four-objective
*frustration vector* — the off-target interactions that compete with the
intended fold.

Method (Shirt-Ediss, Torelli, Navarro & Krasnogor, Nat Commun 2026):
a candidate sequence is scored against four classes of unintended
hybridization. We count them with a k-mer / reverse-complement index — for
each length-k window of every strand, an off-target event is any *other*
window (on any strand, or the same strand at a non-intended offset) whose
sequence is the reverse complement. Longer complementary runs are weighted
super-linearly because hybridization free energy grows with duplex length.

The four objective channels returned by `score()`:
    j1  staple <-> wrong-scaffold
    j2  scaffold <-> scaffold (intra-scaffold secondary structure)
    j3  staple <-> staple
    j4  intra-staple hairpins

Each is a kinetic trap on the assembly landscape — the nucleic-acid analog
of a misfolded metastable protein state in the vampnet MSM.
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Tuple, Dict

from .contactmap import ContactMap, reverse_complement


# k-mer length for the off-target index. ~7-8 is the practical floor where
# a DNA duplex becomes kinetically meaningful at room temperature; the
# Krasnogor scorer sweeps a range, we default to one value and weight by run.
_DEFAULT_K = 8


def _kmer_positions(seq: str, k: int) -> Dict[str, List[int]]:
    idx: Dict[str, List[int]] = defaultdict(list)
    for i in range(len(seq) - k + 1):
        idx[seq[i:i + k]].append(i)
    return idx


def _run_weight(k: int) -> float:
    """Super-linear weight for a length-k complementary run (proxy for the
    Boltzmann weight of the unintended duplex). Grows ~ k^2 so a 16-mer
    trap counts far more than two 8-mer traps.
    """
    return (k / _DEFAULT_K) ** 2


def _count_offtarget(a: str, b: str, k: int, same: bool, intended_offset=None) -> Tuple[float, list]:
    """Count reverse-complement k-mer matches between strand a and strand b.

    If same is True, a is b (intra-strand); we skip the trivial self-match
    and the intended register. Returns (weighted_score, hotspots) where
    hotspots is a list of (pos_a, pos_b, k) for surfacing in viz.
    """
    b_rc_index = _kmer_positions(reverse_complement(b), k)
    score = 0.0
    hotspots = []
    for i in range(len(a) - k + 1):
        win = a[i:i + k]
        # A window of a hybridizes to b wherever b's reverse complement
        # contains it; equivalently where win appears in revcomp(b).
        for j in b_rc_index.get(win, ()):
            if same and abs(i - j) < k:
                continue  # trivial self overlap
            score += _run_weight(k)
            hotspots.append((i, j, k))
    return score, hotspots


class ScoredDesign:
    """Holds a contact map + its four-objective off-target score. Mirrors
    the role of vampnet's fitted model object (carries summary() + the
    payload that downstream commands consume).
    """

    def __init__(self, cm: ContactMap, k: int = _DEFAULT_K):
        self.cm = cm
        self.k = k
        self.objectives, self.hotspots = self._compute()

    def _compute(self):
        cm = self.cm
        k = self.k
        scaf = cm.scaffold.upper()
        staples = [s.upper() for s in cm.staples]

        # j2: scaffold self-complementarity.
        j2, hot2 = _count_offtarget(scaf, scaf, k, same=True)

        # j1: staple <-> wrong-scaffold. The *intended* staple-scaffold
        # duplexes are by construction reverse-complementary; in the absence
        # of an intended_pairs register (v0.1 routing-only imports) we count
        # all staple/scaffold complementarity and treat it as an upper bound.
        # TODO(v0.2): subtract intended_pairs so only *off*-target remains.
        j1 = 0.0
        hot1 = []
        for s in staples:
            sc, h = _count_offtarget(s, scaf, k, same=False)
            j1 += sc
            hot1 += [("staple", s, *t) for t in h]

        # j3: staple <-> staple (distinct staples).
        j3 = 0.0
        hot3 = []
        for ia in range(len(staples)):
            for ib in range(ia + 1, len(staples)):
                sc, h = _count_offtarget(staples[ia], staples[ib], k, same=False)
                j3 += sc
                hot3 += h

        # j4: intra-staple hairpins.
        j4 = 0.0
        hot4 = []
        for s in staples:
            sc, h = _count_offtarget(s, s, k, same=True)
            j4 += sc
            hot4 += h

        objectives = {
            "j1_staple_wrong_scaffold": j1,
            "j2_scaffold_scaffold": j2,
            "j3_staple_staple": j3,
            "j4_staple_hairpin": j4,
            "total": j1 + j2 + j3 + j4,
        }
        hotspots = {"j1": hot1, "j2": hot2, "j3": hot3, "j4": hot4}
        return objectives, hotspots

    def vector(self) -> List[float]:
        return [
            self.objectives["j1_staple_wrong_scaffold"],
            self.objectives["j2_scaffold_scaffold"],
            self.objectives["j3_staple_staple"],
            self.objectives["j4_staple_hairpin"],
        ]

    def summary(self) -> dict:
        return {
            "name": self.cm.name,
            "k": self.k,
            "objectives": self.objectives,
            "n_hotspots": {key: len(v) for key, v in self.hotspots.items()},
        }


def score(cm: ContactMap, k: int = _DEFAULT_K) -> ScoredDesign:
    """Score a contact map. Returns a ScoredDesign (see summary())."""
    return ScoredDesign(cm, k=k)
