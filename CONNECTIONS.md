# Connections: DNA-origami design ↔ protein-dynamics (chimerax-vampnet)

This bundle is the deliberate mirror image of [`chimerax-vampnet`](../chimerax-vampnet).
The two solve the *same* problem — **folding/assembly reliability as the
minimization of landscape frustration** — from opposite ends, and they
share one data structure (the **contact map**) and one control pattern (a
**model/designer-in-the-loop** driven over MCP).

## 1. The structural mirror

| chimerax-vampnet module | chimerax-origami module | shared idea |
|---|---|---|
| `featurize.py` (ensemble → CA contact map) | `contactmap.py` (design → base-pair contact map) | **contact map is the universal substrate** |
| `vampnet_core.py` (features → state landscape) | `score.py` (contact map → 4-objective off-target vector) | reduce structure to a low-D landscape descriptor |
| `msm.py` (transition graph) | `optimize.py` (Pareto-front graph) | expose a frontier the user reads to choose |
| `viz.py` `color_by_state` | `viz.py` `color_by_frustration` | paint a learned per-element scalar onto 3D |
| `animate.py` (derived slow-mode model) | `envelope.py` (derived lipid-shell model) | build a derived model in the session |
| adaptive-sampling loop | `evolve.py` recursive-improvement loop | self-improving loop that grows its own data |
| `mcp_server.py` | `mcp_server.py` | LLM agent drives the loop |

The deep claim worth a methods paper: *whether you are designing a DNA
nanostructure or characterizing a protein conformational switch, you are
minimizing **frustration** on a folding landscape, and the contact map is
the representation on which both problems become the same problem.*
Frustration is not a metaphor here — it is the shared formalism of protein
energy-landscape theory (Wolynes/Onuchic) and DNA-origami design (*Sites of
high local frustration in DNA origami*, Nat Commun 2019). The Krasnogor
off-target score (`score.py`) is literally a de-frustration objective.

## 2. The enveloped-delivery-vehicle connection — yes, and it is direct

**Short answer: yes.** The link runs through three independent threads that
converge on the *same lipid-bilayer interface*, and it is concrete enough
to be a shared module (`envelope.py`), not just an analogy.

### 2a. The DNA side: virus-inspired enveloped origami
Perrault & Shih, *Virus-Inspired Membrane Encapsulation of DNA
Nanostructures To Achieve In Vivo Stability* (ACS Nano 2014), wrap a
DNA-origami octahedron in a single lipid bilayer by hanging lipid–DNA
conjugates off **outer staple handles** at a controlled density (~1 per
~180 nm²). The result mimics an **enveloped virus particle** and yields:
nuclease protection, ~100× lower immune activation, ~17× better
pharmacokinetic bioavailability. This is *the* canonical DNA-origami
**enveloped delivery vehicle**, and it comes straight out of the
Shih/Douglas lineage that also produced cadnano. `origami envelope`
implements its design math (surface area → handle count → carrier staples →
predicted in-vivo effects).

### 2b. The protein side: your target is already a membrane switch
chimerax-vampnet's demo target is the **Notch1 NRR** — a *membrane
receptor* conformational switch. The plan's v0.3 entry adds a
"**COM-distance restraint for NRR membrane anchor**": the protein-dynamics
project already had to model the lipid-bilayer interface to keep the NRR
fragment from dissociating. Both projects independently arrived at the
membrane as the boundary condition that makes the system behave correctly.

### 2c. The control connection: the envelope is a landscape perturbation
In vampnet, the **anti-NRR antibody** is a perturbation that *reshapes the
conformational landscape* — it shifts the stationary population from the
auto-inhibited basin toward the activated basin. In origami, the **lipid
envelope** is a perturbation that *reshapes the fate landscape* of the
particle — it shifts the in-vivo distribution from "degraded / immunogenic"
toward "circulating / deliverable." Same abstraction (a binding event that
re-weights an ensemble's stationary distribution), one on a conformational
landscape, one on a pharmacokinetic one. `origami envelope` is therefore
the structural twin of vampnet's antibody-perturbation analysis.

**Unifying picture for a delivery vehicle that carries a designed protein
to a Notch-expressing neural cell:** the origami provides the enveloped
chassis (2a), the protein cargo's conformational behavior is characterized
by VAMPnet (2b), and in both cases a membrane-binding event is the control
knob (2c). The two bundles are two halves of one synthetic-delivery
design loop.

## 3. The recursive-improvement loop (Sakana-style)

`evolve.py` adds a **Sakana AI Darwin-Gödel-Machine-style** open-ended
self-improvement loop, the origami analog of vampnet's adaptive-sampling
loop. It is **quality-diversity / MAP-Elites**, not plain hill-climbing:

- **genome** = the scaffold sequence (routing/staples held fixed — we
  optimize the *sequence realization*, exactly as `scaffoldselector` does).
- **benchmark** = the four-objective off-target score (`score.py`) — a
  cheap, fast, *objective* fitness, in contrast to Sakana's LLM-judged code
  edits. This is the key adaptation: the recursion is grounded in a
  physics-motivated benchmark, so it cannot reward-hack.
- **archive** = a grid of niches keyed by `(GC-content bin, dominant
  off-target channel)`. Keeping a *grid* of stepping stones — including
  novel-but-worse designs — is what makes it open-ended, the same reason
  Sakana retains under-performing-but-novel agents and vampnet keeps
  sampling under-populated states rather than only the deepest basin.
- **loop** = sample a niche → mutate (point edits + optional splice from a
  natural-sequence library, à la the Krasnogor "favourable scaffold
  regions") → score → admit if non-dominated in its niche.

Run `examples/recursive_improvement_demo.py`: on a deliberately
trap-rich seed it drives total off-target frustration **1210 → 0** across a
16-design stepping-stone lineage. The `on_step` callback is the hook an MCP
agent uses to *watch and steer* the loop — closing the same human/LLM-in-
the-loop circuit vampnet exposes for adaptive MD sampling.

### Why this composes into a "lab"
Stack the three loops and you get a closed design–build–test–learn lab:

1. **evolve** (this bundle) proposes low-frustration scaffold sequences;
2. **score / optimize** filter them (the fast forward model);
3. **oxDNA / CanDo** validate the survivors' 3D folding (the slow forward model);
4. **envelope** plans the delivery chassis;
5. **vampnet** characterizes the protein cargo's dynamics;
6. an **MCP agent** drives all of it and decides the next round.

That is a Sakana-style recursive-improvement *lab* whose every step is
grounded in a physical benchmark rather than self-judged.

## References
- Shirt-Ediss, Torelli, Navarro & Krasnogor. *Optimising DNA origami assembly by reducing off-target interactions.* Nat Commun (2026). https://www.nature.com/articles/s41467-026-73387-4
- Perrault & Shih. *Virus-Inspired Membrane Encapsulation of DNA Nanostructures To Achieve In Vivo Stability.* ACS Nano (2014). https://pubs.acs.org/doi/10.1021/nn5011914
- Douglas et al. *Rapid prototyping of 3D DNA-origami shapes with caDNAno.* NAR (2009).
- *Sites of high local frustration in DNA origami.* Nat Commun (2019). https://www.nature.com/articles/s41467-019-09002-6
- Mardt et al. *VAMPnets for deep learning of molecular kinetics.* Nat Commun (2018).
- Sakana AI. *The Darwin-Gödel Machine* / *The AI Scientist* (open-ended self-improvement).
- `scaffoldselector` — https://scaffoldselector.readthedocs.io/
