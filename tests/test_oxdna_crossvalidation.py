"""Importer cross-validation against tacoxDNA / oxDNA (VALIDATION.md test 2).

CI-safe: compares our cadnano importer's parse of caca.json against the
topology that tacoxDNA produced for the SAME file (caca.json.top, vendored
from the oxDNA NEW_RELAX_PROCEDURE example). Two fully independent cadnano
parsers must agree on nucleotide count, strand count, and per-strand lengths.

The live oxDNA run (relaxation + base-pair formation + geometry vs `shape`)
is documented and reproduced by examples/oxdna_validation.py; it needs a
local oxDNA build so it is not part of CI.
"""

import os

from src import contactmap as cmap

DATA = os.path.join(os.path.dirname(__file__), "..", "examples", "data")


def _tacoxdna_strand_lengths(top_path):
    """Parse an oxDNA topology (tacoxDNA output): header '<N> <n_strands>',
    then one line per nucleotide '<strand_id> <base> <3'> <5'>'."""
    with open(top_path) as f:
        n_nuc, n_strands = (int(x) for x in f.readline().split()[:2])
        sizes = {}
        for line in f:
            p = line.split()
            if p:
                sizes[p[0]] = sizes.get(p[0], 0) + 1
    return n_nuc, n_strands, sorted(sizes.values())


def test_importer_matches_tacoxdna_topology():
    cm = cmap._load_cadnano(os.path.join(DATA, "caca.json"))
    n_nuc, n_strands, ox_lengths = _tacoxdna_strand_lengths(
        os.path.join(DATA, "caca.json.top"))

    my_lengths = sorted([len(cm.scaffold)] + [len(s) for s in cm.staples])

    # totals agree
    assert cm.n_bases == n_nuc == 608
    assert 1 + len(cm.staples) == n_strands == 6
    # per-strand lengths agree exactly between the two independent parsers
    assert my_lengths == ox_lengths == [42, 42, 42, 42, 150, 290]


def test_importer_pairs_match_scaffold_length():
    # caca is fully duplexed: all 290 scaffold bases are intended-paired, which
    # is exactly the base-pair count oxDNA forms on relaxation (see
    # examples/oxdna_validation.py: 290/290 at the calibrated threshold).
    cm = cmap._load_cadnano(os.path.join(DATA, "caca.json"))
    assert len(cm.intended_pairs) == 290
    assert len({p[2] for p in cm.intended_pairs}) == 290
