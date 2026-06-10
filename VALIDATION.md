# Real-world validation plan

The unit tests prove the wiring on synthetic/hand-built inputs with known
answers. This document lists the **real** experiments that would validate
each component against published designs, datasets, and tools — i.e. turn
the scaffold into something trustworthy. Ordered roughly easiest → hardest.

## 0. Importer round-trips (do first — cheap, high-signal)

- **Rothemund canonical 2D set.** Import the classic cadnano files (rectangle,
  three-trapezoid, smiley, star, triangle; Rothemund 2006) with the M13mp18
  scaffold (7,249 nt) applied. Assertions that must hold:
  - recovered scaffold length == applied sequence length (7,249);
  - staple count matches the published design (the rectangle is 226 staples);
  - **every staple base in a duplex region is paired exactly once**, and every
    paired staple base is the Watson–Crick complement of its scaffold partner;
  - re-export the assigned staple sequences and diff against cadnano's own
    "Export Staples" CSV — they must be identical.
- **3D lattice designs.** A honeycomb-lattice bundle (e.g. Douglas et al. 2009
  *Nature* monolith / six-helix bundle) and a square-lattice block, to exercise
  crossovers between many helices and confirm helix assignment + pairing.
- **Insertions/deletions + circular scaffold.** A design using loops/skips and
  a circular M13 — confirm length accounting and the cycle walk on real files.
- **Scaffold variants.** p7249 (M13mp18), p7560, p8064, phiX174 — same routing,
  different sequence; confirms `sequence` application is routing-faithful.

## 1. Off-target scorer (`score`) vs. the Krasnogor paper / scaffoldselector

- **Reproduce the paper's ranking.** Shirt-Ediss, Torelli, Navarro & Krasnogor
  (*Nat Commun* 2026) identify *favourable* and *unfavourable* scaffold regions
  and test them on 2D and 3D origami. Run `origami score` / `origami optimize`
  over their scaffold variants and check the ranking agrees (favourable
  scaffolds score lower total off-target).
- **Cross-check against scaffoldselector** on the same contact maps: our Pareto
  front should overlap theirs (same four objectives). Differences localize bugs.
- **Thermodynamic cross-check.** For a handful of staples, compare our k-mer
  off-target hits against **NUPACK** / **ViennaRNA** predicted secondary
  structure and cross-hybridization (MFE, equilibrium pair probabilities).
  Hotspots should coincide.

## 2. Geometric forward model (`shape`, DGNN) vs. oxDNA / CanDo

- **RMSD to oxDNA.** Relax each reference design in **oxDNA** (oxDNA.org or
  local CUDA) and compare the predicted shape (Kabsch RMSD, radius of gyration)
  against the DGNN/`shape` output. The Nat Mater 2024 paper reports near-oxDNA
  accuracy — reproduce that on a held-out design once a DGNN checkpoint is wired.
- **CanDo flexibility.** Compare per-base-pair RMSF against **CanDo**'s thermal
  fluctuation / B-factor-like output on the same design; the soft (nicked,
  single-stranded, low-stiffness) regions should rank highest in both.
- **Known floppy vs. rigid designs.** A design with a deliberate flexible hinge
  (e.g. a DNA-origami hinge/box lid) should show a localized RMSF hotspot.

## 3. Assembly MSM bridge (`assembly_msm`) vs. published folding studies

- **oxDNA folding/annealing trajectories.** Generate real folding trajectories
  (the `md/oxdna_modal.py` annealing protocol, or published sets such as the
  Snodin/Doye oxDNA origami folding simulations) and run `assembly_msm`.
  Validate that:
  - implied timescales are converged (flat vs. lag — the standard MSM test);
  - the recovered metastable states match known intermediates (seam closure,
    partial-bundle states);
  - **states flagged as traps correspond to mis-registered / kinetically stuck
    configurations** in the trajectory, not productive on-pathway intermediates.
- **Correlate with the static scorer.** Designs the `score` module flags as
  high off-target should show *more / deeper trap states* in the assembly MSM —
  this is the central claim (static frustration ⇒ kinetic traps) and is directly
  falsifiable.

## 4. Sequence FM (`Evo2Mutator`) vs. random + scaffoldselector

- **Ablation.** On M13 circular-permutation candidates, compare three loops:
  random mutation, Evo-2-guided (`mode=score`), and Evo-2-generate. Metric:
  off-target total reached vs. compute budget, and the **GC/length/motif
  realism** of the winning scaffolds (Evo 2 should avoid pathological motifs).
- **Held-out plausibility.** Score natural vs. shuffled scaffolds with Evo 2;
  natural should score higher log-likelihood (a sanity check the backend works).

## 5. Encapsulation (`envelope`) vs. Perrault & Shih

- Reproduce the handle-density arithmetic for the Perrault & Shih octahedron:
  at 1 conjugate / ~180 nm² the predicted handle count should match the paper's
  geometry, and the carrier-staple set should land on outer-face staples once a
  real 3D model (from `shape`/oxDNA) feeds `envelope`.

## 6. End-to-end: predict assembly quality, then check against the wet lab

The capstone experiment. Take a panel of published origami with **measured
assembly yield** (gel/AFM/TEM) — ideally spanning good and bad folders — and
run the full stack: `score` (frustration) + `assembly_msm` (trap depth) +
`shape` (geometric strain). Test whether **low predicted frustration / shallow
traps correlate with high measured yield**. A positive correlation across an
independent design panel is the result that would justify a methods paper;
a null result tells you which forward model is wrong.

## Cross-cutting test-suite additions (CI-friendly)

- **Conservation invariants** on any imported design: Σ nucleotides ==
  scaffold + Σ staples; every duplex scaffold base paired ≤ 1×; complementarity
  of every recovered pair.
- **Golden-file tests**: check a small real cadnano file into `examples/data/`
  and pin its summary (lengths, staple count, a few pair indices) so importer
  regressions are caught.
- **Determinism**: fixed-seed `evolve` / `assembly_msm` reproducible across runs
  (already covered for synthetic inputs; extend to a real imported design).
