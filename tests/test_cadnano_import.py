"""Real cadnano2 .json importer tests.

Builds a minimal but VALID 2-helix design by hand (a scaffold that runs
along helix 0, crosses to helix 1, and returns; one staple per helix) so the
recovered scaffold path, staple sequences, and base pairing all have an exact
known answer. This is the importer's ground-truth unit test, in the spirit of
vampnet's synthetic-trajectory tests.

Geometry (8 columns per helix):
  scaffold 5'->3': (0,0)..(0,7) -> (1,7)..(1,0)            = 16 nt
  staple A 5'->3': (0,7)..(0,0)  pairs scaffold (0,*)      =  8 nt
  staple B 5'->3': (1,0)..(1,7)  pairs scaffold (1,*)      =  8 nt
"""

import json
import os
import tempfile

from src import contactmap as cmap
from src.contactmap import reverse_complement


def _empty(n):
    return [[-1, -1, -1, -1] for _ in range(n)]


def _mini_design():
    """Return a cadnano2-format doc for the 2-helix design above."""
    N = 8
    h0_scaf = _empty(N)
    h1_scaf = _empty(N)
    h0_stap = _empty(N)
    h1_stap = _empty(N)

    # --- scaffold: helix 0 left->right, crossover at b=7, helix 1 right->left
    for b in range(N):
        prev = [-1, -1] if b == 0 else [0, b - 1]
        nxt = [0, b + 1] if b < N - 1 else [1, N - 1]   # last crosses to (1,7)
        h0_scaf[b] = prev + nxt
    for b in range(N):
        # helix 1 scaffold runs decreasing base; (1,7) entered from (0,7).
        prev = [0, 7] if b == N - 1 else [1, b + 1]
        nxt = [1, b - 1] if b > 0 else [-1, -1]          # (1,0) is the 3' end
        h1_scaf[b] = prev + nxt

    # --- staple A on helix 0: 5' at (0,7) -> 3' at (0,0) (decreasing base)
    for b in range(N):
        prev = [-1, -1] if b == N - 1 else [0, b + 1]
        nxt = [0, b - 1] if b > 0 else [-1, -1]
        h0_stap[b] = prev + nxt

    # --- staple B on helix 1: 5' at (1,0) -> 3' at (1,7) (increasing base)
    for b in range(N):
        prev = [-1, -1] if b == 0 else [1, b - 1]
        nxt = [1, b + 1] if b < N - 1 else [-1, -1]
        h1_stap[b] = prev + nxt

    return {
        "name": "mini2helix",
        "vstrands": [
            {"num": 0, "row": 0, "col": 0, "scaf": h0_scaf, "stap": h0_stap,
             "loop": [0] * N, "skip": [0] * N},
            {"num": 1, "row": 0, "col": 1, "scaf": h1_scaf, "stap": h1_stap,
             "loop": [0] * N, "skip": [0] * N},
        ],
    }


def _write(doc):
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(doc, f)
    return path


def test_scaffold_length_and_pairs():
    path = _write(_mini_design())
    cm = cmap._load_cadnano(path)
    assert len(cm.scaffold) == 16
    assert len(cm.staples) == 2
    assert all(len(s) == 8 for s in cm.staples)
    # every scaffold base is paired exactly once (fully duplexed design).
    assert len(cm.intended_pairs) == 16
    scaf_idxs = sorted(p[2] for p in cm.intended_pairs)
    assert scaf_idxs == list(range(16))


def test_helix_assignment():
    path = _write(_mini_design())
    cm = cmap._load_cadnano(path)
    # first 8 scaffold nts on helix 0, next 8 on helix 1.
    assert cm.helices[:8] == [0] * 8
    assert cm.helices[8:] == [1] * 8


def test_sequence_application_and_staple_complement():
    S = "ACGTACGTAACCGGTT"   # 16 nt, known scaffold
    path = _write(_mini_design())
    cm = cmap._load_cadnano(path, scaffold_sequence=S)
    assert cm.scaffold == S
    # staple A pairs scaffold[0:8] antiparallel -> reverse complement.
    # staple B pairs scaffold[8:16] antiparallel -> reverse complement.
    expected = {reverse_complement(S[0:8]), reverse_complement(S[8:16])}
    assert set(cm.staples) == expected


def test_poly_n_when_no_sequence():
    path = _write(_mini_design())
    cm = cmap._load_cadnano(path)
    assert set(cm.scaffold) == {"N"}
    assert all(set(s) == {"N"} for s in cm.staples)


def test_skip_shortens_scaffold():
    doc = _mini_design()
    doc["vstrands"][0]["skip"][3] = -1     # delete one scaffold base
    path = _write(doc)
    cm = cmap._load_cadnano(path)
    assert len(cm.scaffold) == 15          # 16 - 1 deleted
    # skip removes the whole lattice position (scaffold + staple base there),
    # so exactly that one base pair disappears: 16 -> 15.
    assert len(cm.intended_pairs) == 15
    assert len(cm.staples[0]) == 7         # staple A also lost its base at (0,3)


def test_insertion_lengthens_scaffold():
    doc = _mini_design()
    doc["vstrands"][0]["loop"][2] = 1      # insert one base
    path = _write(doc)
    cm = cmap._load_cadnano(path)
    assert len(cm.scaffold) == 17          # 16 + 1 inserted


def test_detect_format_picks_cadnano():
    path = _write(_mini_design())
    assert cmap._detect_format(path) == "cadnano"


def test_circular_scaffold_walks_full_cycle():
    # Make the scaffold circular: link the 3' end (1,0) back to the 5' end (0,0).
    doc = _mini_design()
    h0, h1 = doc["vstrands"][0], doc["vstrands"][1]
    h0["scaf"][0][0:2] = [1, 0]    # (0,0) prev now (1,0)
    h1["scaf"][0][2:4] = [0, 0]    # (1,0) next now (0,0)
    path = _write(doc)
    cm = cmap._load_cadnano(path)
    assert len(cm.scaffold) == 16  # all 16 still recovered via cycle walk
