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

## References

- Shirt-Ediss, Torelli, Navarro & Krasnogor. *Optimising DNA origami assembly by reducing off-target interactions.* Nat Commun (2026). https://www.nature.com/articles/s41467-026-73387-4
- Douglas et al. *Rapid prototyping of 3D DNA-origami shapes with caDNAno.* Nucleic Acids Res (2009).
- Perrault & Shih. *Virus-Inspired Membrane Encapsulation of DNA Nanostructures To Achieve In Vivo Stability.* ACS Nano (2014). https://pubs.acs.org/doi/10.1021/nn5011914
- `scaffoldselector` — https://scaffoldselector.readthedocs.io/
