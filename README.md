# chimerax-origami

[![tests](https://github.com/m9h/chimerax-origami/actions/workflows/tests.yml/badge.svg)](https://github.com/m9h/chimerax-origami/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Off-target-aware DNA-origami sequence design + lipid-envelope modeling,
integrated into UCSF ChimeraX. The DNA-nanostructure **mirror image** of
[`chimerax-vampnet`](../chimerax-vampnet): the two bundles share one
abstraction — the **contact map** — and apply it to the same underlying
problem (folding/assembly reliability as *landscape frustration*) from two
ends:

| | chimerax-vampnet | chimerax-origami |
|---|---|---|
| Object | protein conformational landscape | DNA-origami assembly landscape |
| Input | MD / AlphaFlow / BioEmu ensembles | cadnano / scadnano / oxDNA designs |
| Shared abstraction | CA–CA **contact map** | base-pair **contact map** |
| "Frustration" | metastable traps in free energy | off-target interactions in sequence |
| Forward model | OpenMM / deeptime VAMPnet | k-mer off-target scorer / oxDNA |
| Inverse design | adaptive sampling | multi-objective Pareto scaffold selection |
| Perturbation | anti-NRR antibody shifts state pops | lipid envelope shifts in-vivo fate |
| Output | states + transition rates | Pareto front + frustration hotspots |

It loads a design, converts it to a contact map, scores the **four classes
of off-target interaction** (Shirt-Ediss, Torelli, Navarro & Krasnogor,
*Nat Commun* 2026; `scaffoldselector`), runs a **multi-objective Pareto
selection** over candidate scaffold sequences, and surfaces frustration
hotspots — plus an optional **lipid-bilayer envelope** (Perrault & Shih
virus-inspired encapsulation, *ACS Nano* 2014) — as ChimeraX models.

Every command returns JSON-serializable data so an MCP-capable LLM agent
(Claude Desktop, Cursor, etc.) can drive a design-score-optimize loop via
the included HTTP bridge — exactly as in chimerax-vampnet.

See [`CONNECTIONS.md`](CONNECTIONS.md) for the full analysis tying DNA-origami
planning to the protein-dynamics work, including the **enveloped delivery
vehicle** link.

## The four off-target classes (scored by `origami score`)

1. **staple ↔ wrong-scaffold** — a staple binds a scaffold region other than its intended one
2. **scaffold ↔ scaffold** — the scaffold folds on itself (secondary structure)
3. **staple ↔ staple** — two staples cross-hybridize
4. **intra-staple hairpins** — a staple folds on itself

Each is a kinetic trap on the assembly landscape — the nucleic-acid analog
of a misfolded metastable protein state.

## Commands

```
origami load_design <path> [format auto|cadnano|scadnano|oxdna|contactmap]
origami score                       # 4-vector + per-region hotspots
origami optimize [candidates ...]   # Pareto front over scaffold sequences
origami frustration                 # color helices/bases by off-target density
origami network                     # off-target interaction graph (JSON)
origami envelope [density 180]      # lipid-handle placement + bilayer shell
origami report <path.html>          # scaffoldselector-style HTML report
origami save <path> / load <path>
origami mcp serve [port 7346] / stop
```

## Status — v0.1 (scaffold)

Mirrors the chimerax-vampnet module layout 1:1. The off-target k-mer
scorer and Pareto selector are functional pure-numpy; the cadnano/oxDNA
geometry parsers and the `envelope` bilayer mesh are stubbed with clearly
marked TODOs (same convention vampnet used for the MarS-FM / ESMFold2
adapters pending released checkpoints).

| Module | Role | Mirrors (vampnet) |
|---|---|---|
| `src/cmd.py` | command registration (12 commands) | `cmd.py` |
| `src/contactmap.py` | design loaders → base-pair contact map | `featurize.py` |
| `src/score.py` | 4-class off-target scoring | `vampnet_core.py` |
| `src/optimize.py` | multi-objective Pareto scaffold selection | `msm.py` |
| `src/viz.py` | color by frustration density | `viz.py` |
| `src/envelope.py` | lipid-bilayer delivery envelope | `animate.py` |
| `src/mcp_server.py` | HTTP/JSON bridge for LLM agents | `mcp_server.py` |
| `src/__init__.py` | BundleAPI subclass | `__init__.py` |

## Research roadmap: the foundation-model stack for DNA nanostructures

chimerax-vampnet ingests heterogeneous protein FMs (AlphaFold/AlphaFlow,
BioEmu, Boltz-2, ESMFold2, MarS-FM) on equal footing. The DNA-nanostructure
field acquired its *own* equivalent stack in 2024–26 — the modules below
should ingest these the way vampnet ingests protein FMs.

| Protein FM (vampnet ingests) | role | DNA-nanostructure analog | maturity |
|---|---|---|---|
| AlphaFlow / BioEmu (flow/emulated ensembles) | generative ensemble | **Diffusion DNA-origami generator** — *De novo design of DNA origami with a generative diffusion model*, Nat Commun 2026, [s41467-026-73578-z](https://www.nature.com/articles/s41467-026-73578-z): guided diffusion trained on simulated equilibrium conformations + strand routing | exists (2026) |
| MarS-FM (fast trajectory surrogate, ~600× MD) | fast forward model | **GNN origami shape predictor** — *Prediction of DNA origami shape using graph neural network*, Nat Mater 2024, [s41563-024-01846-8](https://www.nature.com/articles/s41563-024-01846-8): real-time 3D conformation, physics-informed, scales to supramolecular assemblies | exists (2024) |
| ESM / ESM3 (protein language model) | sequence FM | **Evo 2** genomic LLM (Arc Institute, 40B, 1 M-token context, single-nt; controllable DNA generation), [biorxiv 2025.02.18.638918](https://www.biorxiv.org/content/10.1101/2025.02.18.638918v1); Nucleotide Transformer | exists (2025) |
| AlphaFold3 / Boltz-2 (complex prediction) | structure validation | **AlphaFold3 / RoseTTAFoldNA / Chai-1** (nucleic-acid-capable) for junctions, aptamers, protein–DNA interfaces | exists |
| OpenMM (atomistic MD) | gold-standard physics | **oxDNA / oxRNA, mrDNA, SNUPI, CanDo** (coarse-grained); [oxDNA.org](https://academic.oup.com/nar/article/49/W1/W491/6261791) GPU webserver | exists |
| deeptime VAMPnet (learn slow CVs / states) | kinetics/landscape | **— open gap —** no learned assembly-pathway MSM yet | **opportunity** |

**Highest-leverage adds, in order:**
1. **GNN shape predictor → fast forward model.** Lets `evolve.py` / `optimize.py` score *geometric* fidelity (does it fold to the target shape?) in real time, not just sequence off-target. The MarS-FM analog.
2. **Evo 2 → FM-guided mutation operator.** ✅ *scaffolded* — `evolve.Evo2Mutator` + `md/evo2_modal.py` replace random point edits with Evo-2-scored / Evo-2-generated scaffold edits, fusing the FM likelihood with the off-target objective (`cost = off_target − fm_weight·loglik`); degrades to random mutation if no backend. Generates the "favourable scaffold regions" the Krasnogor paper mines biologically. The ESM analog, and the cleanest standalone contribution.
3. **Diffusion generator → inverse-design source.** `evolve` proposes, the diffusion model refines + routes — a generate-and-verify loop mirroring vampnet's adaptive sampling.
4. **AlphaFold3 → motif/interface validator** for aptamers, junctions, and the lipid-handle / antibody-conjugation sites.

**The method-level bridge (a paper, not just an analogy):** apply
chimerax-vampnet's *actual machinery* — a VAMPnet/MSM — to oxDNA **assembly
trajectories**. Just as vampnet learns metastable protein states from MD,
one can learn the metastable *folding intermediates* of an origami and
recover off-target traps as metastable kinetic traps. This makes the two
bundles share code, not just a contact-map abstraction.

## Applications & adjacent research lines (mine these)

The `envelope.py` "enveloped delivery vehicle" thread is one node in a
larger map. Lines worth tracking / building toward:

- **Encapsulation & shielding** (the `envelope.py` family): lipid-bilayer envelope ([Perrault & Shih 2014](https://pubs.acs.org/doi/10.1021/nn5011914)); virus-capsid-protein coating (*Virus-Encapsulated DNA Origami*, [Nano Lett 2014](https://pubs.acs.org/doi/abs/10.1021/nl500677j)); **exosome-mimicking coatings that cross the blood–brain barrier** (2024–25); silicification / oligolysine-PEG coating; *Synthetic Cell Armor* origami nanoshells ([PMC10416349](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10416349/)). All are the same landscape-perturbation pattern → extend `origami envelope`.
- **Therapeutics:** autonomous reconfigurable [nanorobot arrays (2025)](https://phys.org/news/2025-11-nanorobots-based-reconfigurable-dna-origami.html); thrombin tumor-targeting nanorobots (Li et al. 2018); CpG-adjuvant cancer-vaccine scaffolds; DNA-origami T-cell engagers; gene-encoding origami for mammalian expression ([PMC9950468](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9950468/)).
- **The neural-organoid bridge to chimerax-vampnet:** an origami that *displays Notch ligands (DLL4) at nm-precise spacing/valency* to tune the exact receptor clustering vampnet models — origami as a **spatial signaling scaffold**, delivered to brain organoids via exosome-coated chassis. This is the killer app uniting the two repos.
- **Geometry compilers (inverse design of routing):** DAEDALUS / PERDIX / TALOS / ATHENA / vHelix — automated scaffold routing from a target mesh; the geometric counterpart to `evolve.py`'s sequence optimization.
- **DNA data storage & digital-twin lines** (Krasnogor / GitLife): version control for cell engineering, DNA-based data structures — a different application of the same off-target-aware sequence design.

## References

- Shirt-Ediss, Torelli, Navarro & Krasnogor. *Optimising DNA origami assembly by reducing off-target interactions.* Nat Commun (2026). https://www.nature.com/articles/s41467-026-73387-4
- *De novo design of DNA origami with a generative diffusion model.* Nat Commun (2026). https://www.nature.com/articles/s41467-026-73578-z
- *Prediction of DNA origami shape using graph neural network.* Nat Mater (2024). https://www.nature.com/articles/s41563-024-01846-8
- Brixi et al. *Genome modeling and design across all domains of life with Evo 2.* bioRxiv (2025). https://www.biorxiv.org/content/10.1101/2025.02.18.638918v1
- Douglas et al. *Rapid prototyping of 3D DNA-origami shapes with caDNAno.* Nucleic Acids Res (2009).
- Perrault & Shih. *Virus-Inspired Membrane Encapsulation of DNA Nanostructures To Achieve In Vivo Stability.* ACS Nano (2014). https://pubs.acs.org/doi/10.1021/nn5011914
- *Virus-Encapsulated DNA Origami Nanostructures for Cellular Delivery.* Nano Lett (2014). https://pubs.acs.org/doi/abs/10.1021/nl500677j
- `scaffoldselector` — https://scaffoldselector.readthedocs.io/
