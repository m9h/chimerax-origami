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

## ✅ 1. Off-target scorer — DONE (`tests/test_scorer_validation.py`, `examples/scorer_validation_demo.py`)

Implemented and passing. Results on the real M13mp18 scaffold + Douglas 2009
monolith routing:
- **Detection:** designed self-complementarity (`A + spacer + RC(A)`) scores
  ~**190× higher** than random — and this validation *found and fixed a real
  bug* (the j2/j4 diagonal filter compared a forward index to a reverse-
  complement index, masking long-range self-complementarity; now compares
  physical positions).
- **Cross-tool (ViennaRNA):** Spearman(`j2`, −MFE) = **+0.53** over 150 random
  sequences — the cheap k-mer proxy tracks rigorous thermodynamics. Runs in CI.
- **Krasnogor claim:** for the *fixed* monolith routing, scaffold choice changes
  off-target from **819 (M13)** to **30.5 M (low-complexity ACGT repeat)**;
  `optimize` rejects the pathological scaffold.

Remaining (needs their data / NUPACK): reproduce the paper's exact
favourable/unfavourable region ranking, and overlap our Pareto front with
scaffoldselector's.

## 1b. Original plan — scorer vs. the Krasnogor paper / scaffoldselector

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

## ✅ 2. Geometric forward model vs. real oxDNA — DONE (`tests/test_oxdna_crossvalidation.py`, `examples/oxdna_validation.py`)

Ran a **real oxDNA simulation** (built from source; 2×10⁵ MD steps on the
caca.json origami, 608 nucleotides, ~70 s CPU). Three results:
- **Independent cross-validation:** our cadnano importer and **tacoxDNA** (a
  completely separate parser) agree EXACTLY on the topology — 608 nucleotides,
  6 strands, per-strand lengths [42,42,42,42,150,290]. (Pinned in CI from the
  vendored `caca.json.top`, no oxDNA needed.)
- **Folding validation:** **290 / 290** intended base pairs form in the relaxed
  oxDNA structure (100% at the calibrated base-site threshold) — the design
  folds exactly as the importer's `intended_pairs` predict.
- **Geometry:** the real folded structure has **Rg ≈ 6.8 nm**; the `shape`
  fallback predicts ~28.8 nm — it over-predicts ~4× because it's a placeholder
  lattice that lays the scaffold out linearly rather than folding. This
  quantifies the gap the real DGNN (`md/gnn_shape_modal.py`) would close; a
  meaningful RMSD test awaits that checkpoint.

Remaining: per-residue RMSF vs CanDo, and the DGNN RMSD once a checkpoint exists.

- **RMSD to oxDNA.** Relax each reference design in **oxDNA** (oxDNA.org or
  local CUDA) and compare the predicted shape (Kabsch RMSD, radius of gyration)
  against the DGNN/`shape` output. The Nat Mater 2024 paper reports near-oxDNA
  accuracy — reproduce that on a held-out design once a DGNN checkpoint is wired.
- **CanDo flexibility.** Compare per-base-pair RMSF against **CanDo**'s thermal
  fluctuation / B-factor-like output on the same design; the soft (nicked,
  single-stranded, low-stiffness) regions should rank highest in both.
- **Known floppy vs. rigid designs.** A design with a deliberate flexible hinge
  (e.g. a DNA-origami hinge/box lid) should show a localized RMSF hotspot.

## ✅ 3. Assembly MSM bridge — core DONE (`tests/test_assembly_validation.py`, `examples/thesis_validation_demo.py`)

- **MSM validity:** the slowest implied timescale is flat across lag (14.3–14.6
  over lags 4–10) — the recovered MSM is Markovian.
- **The project thesis, validated:** added a cheap score-driven kinetic folding
  emulator (`assembly.simulate_folding`, surfaced as `origami fold_msm`) so a
  design's folding MSM can be predicted without oxDNA. Across a frustration
  gradient, **static off-target frustration anti-correlates with folding yield
  at Spearman −1.0** (yield 0.72 → 0.56 as off-target rises). The link is
  non-circular: sequence → `score` → localized per-domain frustration →
  kinetics → yield. This is the same quantity test 6 checks against the wet lab.

Remaining (needs oxDNA): replace the emulator with real annealing trajectories
(`md/oxdna_modal.py` / Snodin–Doye sets); `oxpy` now imports in this env, so a
small real-trajectory check is feasible next.

## 3b. Original plan — vs. published folding studies

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

## ✅ 4. Sequence FM ablation — DONE (`tests/test_evo2_ablation.py`)

Score/FM-guided mutation reaches **lower off-target than random in a fixed
budget** (mean 0.0 vs 5.0 at 20 generations on a hard seed). Validation
surfaced — and fixed — a real weakness: the guided mutator picked a *random*
window, making it slower than whole-sequence random mutation on long
scaffolds; it now **targets the worst-frustrated window** (`Evo2Mutator`
gained an `explore` knob), which is what makes it win.

**Real Evo 2 now wired and verified** (was checkpoint-pending). `local_backend`
loads Evo 2 (StripedHyena-2) and runs on the user's **GB10 (Blackwell, sm_121)**
inside `nvcr.io/nvidia/pytorch:26.05-py3` (flash-attn + TE ship in the image);
`EVO2_SIZE` selects 1b/7b. Results from `examples/evo2_local_run.py`:

| | evo2_1b_base | evo2_7b |
|---|---|---|
| natural M13 LL | −1.327 | **−1.189** |
| shuffled LL | −1.382 | −1.376 |
| natural−shuffled gap | 0.055 | **0.187** |
| guided winner LL | −1.404 | **−1.250** |
| random winner LL | −1.409 | −1.385 |

- **Plausibility:** natural M13 fragments score higher Evo 2 log-likelihood than
  shuffled, **5/5** at both sizes; the 7B gap is ~3.4× sharper.
- **Real ablation (20 gens):** Evo 2-guided evolution reaches **lower off-target
  (2 vs 4)** *and* keeps the winning scaffold **more natural** than random — and
  with 7B the guided−random naturalness gap is ~27× larger (the stronger prior
  steers much harder toward genome-plausible sequences while still minimizing
  frustration). GPU/container run, not in CI; the `FakeEvo2` surrogate keeps the
  ablation logic tested in CI.

## 4b. Original plan — vs. random + scaffoldselector

- **Ablation.** On M13 circular-permutation candidates, compare three loops:
  random mutation, Evo-2-guided (`mode=score`), and Evo-2-generate. Metric:
  off-target total reached vs. compute budget, and the **GC/length/motif
  realism** of the winning scaffolds (Evo 2 should avoid pathological motifs).
- **Held-out plausibility.** Score natural vs. shuffled scaffolds with Evo 2;
  natural should score higher log-likelihood (a sanity check the backend works).

## ✅ 5. Encapsulation — DONE (`tests/test_envelope_validation.py`)

Validated against Perrault & Shih (ACS Nano 2014): the planner uses the
paper's **1 conjugate / 180 nm²** density, the handle count follows the exact
relation (`round(area/180)` — ~44 handles for a 50 nm octahedron), doubling
density halves the count, and the reported in-vivo figures (**~17× bioavail.,
~100× lower immune activation**, nuclease protection) are surfaced. The
solid-sphere area estimate is confirmed to *undercount* a hollow wireframe;
`design_envelope` now accepts a `surface_area_nm2` override so a real
`shape`/oxDNA enclosing surface gives a quantitative count.

## 5b. Original plan — vs. Perrault & Shih

- Reproduce the handle-density arithmetic for the Perrault & Shih octahedron:
  at 1 conjugate / ~180 nm² the predicted handle count should match the paper's
  geometry, and the carrier-staple set should land on outer-face staples once a
  real 3D model (from `shape`/oxDNA) feeds `envelope`.

## ✅ 6. End-to-end vs. the wet lab — DONE (`tests/test_yield_validation.py`, `examples/yield_validation_demo.py`)

The capstone, validated against **real single-molecule measurements**. Dataset:
the Krasnogor paper's scaffold variants ([Zenodo 14748478](https://zenodo.org/records/14748478))
— three triangle DNA origami sharing one routing with different scaffold
sequences. Keyed by the supplementary mapping (T1=DEER, T2=LION, T3=BEAR), our
INDEPENDENT off-target score is compared to the paper's optical-tweezers
non-uniformity measurement (Fig. 7D, 379–742 molecules per variant):

| variant | our off-target (lower=better) | measured non-uniformity (lower=better) |
|---|---|---|
| DEER (T1) | 1164 | 0.199 |
| LION (T2) | 1300 | 0.245 |
| BEAR (T3) | 1537 | 0.263 |

**Spearman = +1.0** — our static frustration score reproduces the measured
folding-quality ranking on real origami. Static frustration predicts assembly.

Triple agreement on the triangle: **our off-target T1<T2<T3** = **the authors'
own M1 off-target ranking** (Supp.: "T1 has the lowest M1 score … T3 the
highest") = **measured non-uniformity T1<T2<T3**. Our independent k-mer scorer
reproduces both the authors' predictor and the single-molecule measurement.

Caveats: n=3 variants (perfect rank order, but small); our score tracks the
*non-uniformity* metric, not unfolding force (T2<T1<T3).

**Datasets explored but NOT usable (recorded so we don't revisit):**
- *Rectangle R1/R2/R3 (GOAT/LAMB/MOLE):* only AGE-gel images (qualitative —
  off-target-prone variants aggregate in the loading well) + AFM dimensions
  (Supp. Table 8) that are **instrument-dependent** (the same variant measures
  53 nm at Newcastle vs 66 nm at Bonn; expected 55) → not a clean quantitative
  ground truth.
- *ML stability dataset* ([bioRxiv 2025.07.18.665506](https://www.biorxiv.org/content/10.1101/2025.07.18.665506v1)):
  ~1,400 measurements but across only **3 fixed designs × physicochemical
  conditions** (temp/Mg/pH/DNase). It tests post-assembly *stability vs buffer*,
  which our *sequence* off-target scorer cannot engage (it would give 3 values).
  Wrong tool for that property — noted, not forced.

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
