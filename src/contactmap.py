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


def load_design(session, path: str, format: str = "auto") -> dict:
    """Load a design from disk and build its contact map.

    Returns the ContactMap.summary() dict.
    """
    fmt = format
    if fmt == "auto":
        fmt = _detect_format(path)

    if fmt == "cadnano":
        cm = _load_cadnano(path)
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


def _load_cadnano(path: str) -> ContactMap:
    """Parse a cadnano2 .json (vstrands with scaf/stap routing arrays).

    A full router walks the scaf[] / stap[] [prev_h, prev_b, next_h,
    next_b] linked-list arrays to recover strand paths, then reads the
    'scaf'/'stap' sequence assignment if present. That walk is the bulk of
    a real cadnano importer (see douglaslab/cadnano2's `decode`); for the
    v0.1 scaffold we extract sequences when the design carries an applied
    sequence and otherwise emit a routing-only contact map with a
    placeholder poly-N scaffold so the downstream scorer/optimizer wiring
    can be exercised.

    TODO(v0.2): port cadnano2's strand-graph walk so intended_pairs is
    recovered exactly (scaffold base i <-> staple base j at each crossover).
    """
    with open(path) as f:
        doc = json.load(f)
    vstrands = doc.get("vstrands", [])
    # Length proxy: total scaffold lattice positions that are occupied.
    scaf_len = 0
    for vs in vstrands:
        scaf = vs.get("scaf", [])
        scaf_len += sum(1 for cell in scaf if cell != [-1, -1, -1, -1])
    scaffold = doc.get("scaffold_sequence", "N" * max(scaf_len, 1))
    cm = ContactMap(scaffold=scaffold, staples=[], intended_pairs=[])
    cm.helices = [vs.get("num") for vs in vstrands]
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
