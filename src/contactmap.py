"""Design loading + contact-map construction for chimerax-origami.

This is the structural twin of chimerax-vampnet's featurize.py. Where
vampnet turns a conformational ensemble into a CA-CA *distance/contact*
matrix, this module turns a DNA-origami design into a base-pairing
*contact map*: the graph of which scaffold/staple bases are intended to
hybridize. Both bundles then reason on the contact map.

Loads designs from four sources on (near-)equal footing:

  - cadnano  (.json; honeycomb or square lattice; Douglas lab)
  - scadnano (.sc / .json)
  - oxDNA    (non-relaxed .top + .conf / .dat)
  - contactmap (.npz / .json; the canonical interchange format that
               scaffoldselector consumes — sequences + intended pairs)

The Nat Commun 2026 paper / scaffoldselector operate on the contact map
directly, noting that "caDNAno, scadnano and non-relaxed oxDNA CAD formats
can all be easily converted to the contact map format." This module is
where that conversion lives.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# Session-scoped registry holding the active design. We keep a single
# active design (unlike vampnet's multi-ensemble union) because a Pareto
# scaffold selection is per-design.
_DESIGN_KEY = "_origami_design"

_COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}


@dataclass
class ContactMap:
    """The shared abstraction. Mirrors a featurized ensemble in vampnet.

    scaffold:        the (single) scaffold sequence, 5'->3'.
    staples:         list of staple sequences, 5'->3'.
    intended_pairs:  list of (kind, a_strand, a_idx, b_strand, b_idx) tuples
                     describing the *designed* Watson-Crick contacts. kind is
                     'sc-st' (scaffold-staple). a_strand/b_strand index into
                     [scaffold] + staples (0 == scaffold).
    helices:         optional per-base helix id (for frustration coloring /
                     mapping back onto a 3D model).
    """

    scaffold: str
    staples: List[str] = field(default_factory=list)
    intended_pairs: List[tuple] = field(default_factory=list)
    helices: Optional[list] = None
    source_format: str = "contactmap"
    name: str = "design"

    @property
    def n_bases(self) -> int:
        return len(self.scaffold) + sum(len(s) for s in self.staples)

    def summary(self) -> dict:
        return {
            "name": self.name,
            "format": self.source_format,
            "scaffold_length": len(self.scaffold),
            "n_staples": len(self.staples),
            "n_bases": self.n_bases,
            "n_intended_pairs": len(self.intended_pairs),
        }


def reverse_complement(seq: str) -> str:
    return "".join(_COMPLEMENT.get(b, "N") for b in reversed(seq.upper()))


# ----------------------------------------------------------------------
# Loaders.
# ----------------------------------------------------------------------
def _design_set(session, cm: ContactMap):
    setattr(session, _DESIGN_KEY, cm)


def design_get(session) -> ContactMap:
    cm = getattr(session, _DESIGN_KEY, None)
    if cm is None:
        raise RuntimeError("no design loaded — run `origami load_design` first")
    return cm


def _detect_format(path: str) -> str:
    p = path.lower()
    if p.endswith(".sc"):
        return "scadnano"
    if p.endswith((".top", ".conf", ".dat", ".oxview")):
        return "oxdna"
    if p.endswith(".npz"):
        return "contactmap"
    if p.endswith(".json"):
        # cadnano and scadnano both use .json; disambiguate by content.
        try:
            with open(path) as f:
                head = json.load(f)
            if "vstrands" in head:
                return "cadnano"
            if "helices" in head and "strands" in head:
                return "scadnano"
            if "scaffold" in head and "intended_pairs" in head:
                return "contactmap"
        except Exception:
            pass
        return "cadnano"
    return "contactmap"


def load_design(session, path: str, format: str = "auto",
                scaffold_sequence: Optional[str] = None) -> dict:
    """Load a design from disk and build its contact map.

    scaffold_sequence: optional 5'->3' scaffold sequence to apply along the
        recovered scaffold routing (e.g. M13mp18 / p7249 / p8064). When given,
        staple bases are set to the Watson-Crick complement of the scaffold
        base they pair, so the off-target scorer sees real sequences. When
        omitted, the scaffold is poly-N (routing is still exact; scoring is a
        placeholder until a sequence is applied).

    Returns the ContactMap.summary() dict.
    """
    fmt = format
    if fmt == "auto":
        fmt = _detect_format(path)

    if fmt == "cadnano":
        cm = _load_cadnano(path, scaffold_sequence=scaffold_sequence)
    elif fmt == "scadnano":
        cm = _load_scadnano(path)
    elif fmt == "oxdna":
        cm = _load_oxdna(path)
    elif fmt == "contactmap":
        cm = _load_contactmap(path)
    else:
        raise ValueError(f"unknown design format: {fmt}")

    cm.source_format = fmt
    cm.name = os.path.splitext(os.path.basename(path))[0]
    _design_set(session, cm)
    return cm.summary()


def _load_contactmap(path: str) -> ContactMap:
    """The canonical interchange format. Either an .npz or a .json with
    keys: scaffold (str), staples (list[str]), intended_pairs (list),
    helices (optional list).
    """
    if path.lower().endswith(".npz"):
        import numpy as np
        d = np.load(path, allow_pickle=True)
        scaffold = str(d["scaffold"])
        staples = [str(s) for s in d["staples"]] if "staples" in d.files else []
        pairs = d["intended_pairs"].tolist() if "intended_pairs" in d.files else []
        helices = d["helices"].tolist() if "helices" in d.files else None
        return ContactMap(scaffold, staples, pairs, helices)
    with open(path) as f:
        d = json.load(f)
    return ContactMap(
        scaffold=d["scaffold"],
        staples=d.get("staples", []),
        intended_pairs=[tuple(p) for p in d.get("intended_pairs", [])],
        helices=d.get("helices"),
    )


_EMPTY = [-1, -1, -1, -1]


def _mult(vs: dict, b: int) -> int:
    """Number of nucleotides a lattice position contributes: 0 if deleted
    (skip == -1), else 1 + insertion length (loop)."""
    skip = vs.get("skip") or []
    loop = vs.get("loop") or []
    if b < len(skip) and skip[b] == -1:
        return 0
    return 1 + (loop[b] if b < len(loop) else 0)


def _walk_strands(vmap: dict, which: str):
    """Recover strand paths from a cadnano scaf/stap routing.

    Each cell is [prev_helix, prev_base, next_helix, next_base] — a doubly
    linked list over (helix, base) positions, where prev is the 5' neighbour
    and next the 3' neighbour. Returns a list of strands, each an ordered
    list of (helix, base) positions from 5' -> 3'. Handles linear strands
    (start at a 5' end: prev == [-1, -1]) and circular strands (no 5' end:
    walk the cycle once).
    """
    arrays = {num: vs.get(which, []) for num, vs in vmap.items()}

    def cell(h, b):
        arr = arrays.get(h)
        if arr is None or b < 0 or b >= len(arr):
            return _EMPTY
        return arr[b]

    occupied = {(h, b) for h, arr in arrays.items()
                for b, c in enumerate(arr) if c != _EMPTY}
    visited = set()
    strands = []

    def trace(h, b):
        path = []
        while (h, b) != (-1, -1) and (h, b) not in visited:
            visited.add((h, b))
            path.append((h, b))
            c = cell(h, b)
            h, b = (c[2], c[3]) if c[2] != -1 else (-1, -1)
        return path

    # Linear strands first (those with a real 5' terminus).
    for (h, b) in sorted(occupied):
        c = cell(h, b)
        if c[0] == -1 and c[1] == -1:  # no 5' neighbour -> a 5' end
            s = trace(h, b)
            if s:
                strands.append(s)
    # Any remaining occupied positions belong to circular strands.
    for (h, b) in sorted(occupied):
        if (h, b) not in visited:
            s = trace(h, b)
            if s:
                strands.append(s)
    return strands


def _load_cadnano(path: str, scaffold_sequence: Optional[str] = None) -> ContactMap:
    """Parse a cadnano2 .json into an exact base-pair contact map.

    Walks the scaf[] / stap[] linked-list routing (see douglaslab/cadnano2's
    decode) to recover the scaffold path and every staple path, assigns a flat
    5'->3' nucleotide index to each strand (honouring skips/insertions), and
    derives intended_pairs from co-located scaffold+staple nucleotides (which
    are Watson-Crick paired, antiparallel). With scaffold_sequence, applies it
    along the scaffold and sets each paired staple base to the complement;
    otherwise the scaffold is poly-N (routing exact, sequence a placeholder).

    intended_pairs entries are ("sc-st", 0, scaffold_idx, staple_strand,
    staple_idx) — strand 0 is the scaffold, staple_strand is 1-based among
    staples. cm.helices[i] is the helix number of scaffold nucleotide i.
    """
    with open(path) as f:
        doc = json.load(f)
    vstrands = doc.get("vstrands", [])
    vmap = {vs["num"]: vs for vs in vstrands}

    scaf_paths = _walk_strands(vmap, "scaf")
    stap_paths = _walk_strands(vmap, "stap")

    # --- assign flat nucleotide indices, recording lattice position ---------
    # scaf_at[(h,b)] = list of scaffold flat indices visiting that position
    # (length = multiplicity), in scaffold 5'->3' order.
    scaf_at: dict = {}
    scaf_helix: list = []     # helix id per scaffold nucleotide
    n_scaf = 0
    for path in scaf_paths:
        for (h, b) in path:
            for _ in range(_mult(vmap[h], b)):
                scaf_at.setdefault((h, b), []).append(n_scaf)
                scaf_helix.append(h)
                n_scaf += 1

    # stap_at[(h,b)] = list of (staple_strand_1based, local_idx), 5'->3'.
    stap_at: dict = {}
    staple_lengths: list = []
    for s_i, path in enumerate(stap_paths):
        local = 0
        for (h, b) in path:
            for _ in range(_mult(vmap[h], b)):
                stap_at.setdefault((h, b), []).append((s_i + 1, local))
                local += 1
        staple_lengths.append(local)

    # --- derive intended base pairs at co-occupied positions ----------------
    # Scaffold and staple run antiparallel through a position, so pair the
    # k-th scaffold nucleotide with the (m-1-k)-th staple nucleotide there.
    intended_pairs = []
    for pos, s_list in scaf_at.items():
        t_list = stap_at.get(pos)
        if not t_list:
            continue
        m = min(len(s_list), len(t_list))
        for k in range(m):
            sc_idx = s_list[k]
            strand, st_idx = t_list[m - 1 - k]
            intended_pairs.append(("sc-st", 0, sc_idx, strand, st_idx))

    # --- apply sequences ----------------------------------------------------
    if scaffold_sequence:
        seq = scaffold_sequence.upper().replace("U", "T")
        if len(seq) < n_scaf:
            seq = seq + "N" * (n_scaf - len(seq))   # pad short scaffolds
        scaffold = seq[:n_scaf]
    else:
        scaffold = "N" * max(n_scaf, 1)

    staples = ["N" * L for L in staple_lengths]
    staple_chars = [list(s) for s in staples]
    for (_, _, sc_idx, strand, st_idx) in intended_pairs:
        base = scaffold[sc_idx] if sc_idx < len(scaffold) else "N"
        staple_chars[strand - 1][st_idx] = _COMPLEMENT.get(base, "N")
    staples = ["".join(c) for c in staple_chars]

    cm = ContactMap(scaffold=scaffold, staples=staples,
                    intended_pairs=intended_pairs, helices=scaf_helix)
    return cm


def _load_scadnano(path: str) -> ContactMap:
    """Parse a scadnano design (helices + strands with domains).

    scadnano stores explicit strand objects with sequences and domains, so
    recovery is more direct than cadnano. TODO(v0.2): map each domain's
    (helix, start, end) onto intended_pairs.
    """
    with open(path) as f:
        doc = json.load(f)
    strands = doc.get("strands", [])
    seqs = [s.get("sequence", "") for s in strands if s.get("sequence")]
    scaffold = ""
    staples = []
    for s in strands:
        seq = s.get("sequence", "")
        if s.get("is_scaffold") or s.get("name", "").lower().startswith("scaf"):
            scaffold = seq
        elif seq:
            staples.append(seq)
    if not scaffold and seqs:
        scaffold = max(seqs, key=len)  # longest strand == scaffold heuristic
        staples = [s for s in seqs if s != scaffold]
    return ContactMap(scaffold=scaffold, staples=staples, intended_pairs=[])


def _load_oxdna(path: str) -> ContactMap:
    """Parse an oxDNA topology (.top: per-nucleotide base + 3'/5' neighbors).

    The .top file lists, per nucleotide, (strand_id, base, neighbor_3,
    neighbor_5). We group by strand_id to recover sequences. TODO(v0.2):
    use the paired .conf/.dat to recover intended_pairs from spatial
    proximity of complementary bases.
    """
    strands: dict = {}
    with open(path) as f:
        lines = f.read().splitlines()
    for line in lines[1:]:  # first line is header: <n_nuc> <n_strands>
        parts = line.split()
        if len(parts) < 2:
            continue
        sid, base = parts[0], parts[1]
        strands.setdefault(sid, []).append(base)
    seqs = ["".join(v) for v in strands.values()]
    if not seqs:
        return ContactMap(scaffold="N", staples=[])
    scaffold = max(seqs, key=len)
    staples = [s for s in seqs if s != scaffold]
    return ContactMap(scaffold=scaffold, staples=staples, intended_pairs=[])
