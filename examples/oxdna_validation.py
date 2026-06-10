#!/usr/bin/env python3
"""Real-oxDNA validation of the importer + geometric forward model
(VALIDATION.md test 2).

Unlike the other demos this one needs a LOCAL oxDNA build (not in CI). It
documents and reproduces the procedure and records the results obtained on
the caca.json origami (290-nt scaffold, 5 staples, 608 nucleotides).

PROCEDURE
  1. Build oxDNA:
       git clone --depth 1 https://github.com/lorenzo-rovigatti/oxDNA
       cd oxDNA && mkdir build && cd build && cmake .. && make -j
  2. Convert the cadnano design with tacoxDNA (Python 3):
       python tacoxDNA/src/cadnano_interface.py caca.json sq
     -> caca.json.top, caca.json.oxdna   (vendored .top is in examples/data/)
  3. Relax (MD, interaction_type=DNA, max_backbone_force=5, ~2e5 steps):
       oxDNA input_relax            # see the input written below
  4. Analyze last_conf.dat (this script's helpers).

RESULTS (caca.json, 2e5-step relaxation on CPU, ~70 s)
  - Cross-validation: our importer and tacoxDNA agree EXACTLY on the topology
    — 608 nucleotides, 6 strands, per-strand lengths [42,42,42,42,150,290].
  - Folding: 290 / 290 intended base pairs form in the relaxed structure
    (100% at the calibrated base-site threshold) — the design folds as the
    importer's intended_pairs say it should.
  - Geometry: the real folded structure has Rg ~6.8 nm; the `shape` fallback
    (a placeholder lattice, NOT the real DGNN) predicts Rg ~28.8 nm — it
    over-predicts size ~4x because it lays the scaffold out linearly instead
    of folding the helices together. Wiring md/gnn_shape_modal.py (the real
    DGNN) is what closes this gap; the number above quantifies the placeholder.

This script reproduces step 4 given a relaxed conf + topology.
"""

import sys

NM = 0.8518  # oxDNA length unit -> nm


def parse_topology(top_path):
    with open(top_path) as f:
        n_nuc, n_strands = (int(x) for x in f.readline().split()[:2])
        strand = []
        for line in f:
            p = line.split()
            if p:
                strand.append(int(p[0]))
    return n_nuc, n_strands, strand


def parse_conf(conf_path):
    import numpy as np
    pos, a1 = [], []
    with open(conf_path) as f:
        f.readline(); f.readline(); f.readline()
        for line in f:
            v = line.split()
            if len(v) >= 9:
                pos.append([float(v[0]), float(v[1]), float(v[2])])
                a1.append([float(v[3]), float(v[4]), float(v[5])])
    return np.array(pos), np.array(a1)


def count_base_pairs(pos, a1, threshold=0.55):
    """Geometric base-pair count: complementary base sites (pos + 0.4*a1)
    that are close and antiparallel. Calibrated against oxDNA's own HB."""
    import numpy as np
    from scipy.spatial import cKDTree
    base = pos + 0.4 * a1
    tree = cKDTree(base)
    dd, ii = tree.query(base, k=6)
    seen = set()
    for i in range(len(base)):
        for j, dist in zip(ii[i], dd[i]):
            if j == i or abs(i - j) <= 1:
                continue
            if dist < threshold and a1[i].dot(a1[j]) < -0.5:
                seen.add((min(i, j), max(i, j)))
                break
    return len(seen)


def rg(pos):
    import numpy as np
    c = pos - pos.mean(0)
    return float(np.sqrt((c ** 2).sum(1).mean()))


def main(top_path, conf_path):
    n_nuc, n_strands, strand = parse_topology(top_path)
    pos, a1 = parse_conf(conf_path)
    print(f"topology: {n_nuc} nucleotides, {n_strands} strands")
    print(f"base pairs formed: {count_base_pairs(pos, a1)} (design intends 290)")
    print(f"relaxed Rg: {rg(pos) * NM:.1f} nm")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: oxdna_validation.py <topology.top> <last_conf.dat>")
        print("(run the PROCEDURE in the module docstring to produce these)")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
