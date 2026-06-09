"""Recursive self-improvement loop for DNA-origami scaffold design —
a Sakana-style open-ended evolutionary "lab" over sequences.

This is the origami mirror of chimerax-vampnet's adaptive-sampling loop
(examples/live_adaptive_sampling.py): vampnet's loop repeatedly fits a
VAMPnet, finds the least-sampled state, launches new MD there, and folds
the result back in — a model-in-the-loop that grows its own training data.
Here the loop repeatedly mutates a scaffold, *scores* it against the four
off-target objectives, and folds the survivors back into an archive — a
designer-in-the-loop that grows its own library of stepping-stone designs.

The shape is Sakana AI's Darwin Godel Machine / AI-Scientist recursive
improvement, specialized to a concrete, fast, cheap benchmark (the
Krasnogor off-target score) instead of code edits judged by an LLM:

  archive  := { behavioral_descriptor -> best design in that niche }   (MAP-Elites)
  loop:
    parent  := sample an archive cell (novelty- and quality-weighted)
    child   := mutate(parent)            # point edits / segment splice
    fitness := score(child)              # 4-objective off-target vector
    admit child if it is non-dominated in its niche  (open-ended, keeps
        stepping-stones rather than hill-climbing one scalar)

Keeping a *grid of niches* (quality-diversity), not a single champion, is
what makes it "open-ended": a high-frustration-but-novel scaffold is kept
because it may be the stepping stone to a better region — the same reason
Sakana's archive retains under-performing-but-novel agents, and the same
reason vampnet keeps sampling under-populated states rather than only the
deepest basin.

Pure-Python / numpy-free so it runs inside ChimeraX without extra deps.
Determinism via an explicit seed so a run is reproducible (mirrors how the
vampnet tests pin RNG for the synthetic Markov chain).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

from .contactmap import ContactMap
from .score import score, ScoredDesign
from .optimize import dominates


_BASES = "ACGT"


@dataclass
class Individual:
    scaffold: str
    objectives: List[float]
    total: float
    descriptor: Tuple[int, int]
    parent: Optional[int]
    generation: int
    idx: int = -1


def _gc_content(seq: str) -> float:
    if not seq:
        return 0.0
    gc = sum(1 for b in seq if b in "GC")
    return gc / len(seq)


def _dominant_channel(objectives: List[float]) -> int:
    """Which off-target class dominates — the niche's 'failure mode' axis."""
    return max(range(len(objectives)), key=lambda i: objectives[i]) if objectives else 0


def _descriptor(seq: str, objectives: List[float], gc_bins: int = 8) -> Tuple[int, int]:
    """Behavioral descriptor for the MAP-Elites grid: (GC-content bin,
    dominant off-target channel). Two designs in the same cell are
    'behaviorally similar'; we keep only the non-dominated one per cell.
    """
    gc_bin = min(gc_bins - 1, int(_gc_content(seq) * gc_bins))
    return (gc_bin, _dominant_channel(objectives))


def _mutate(seq: str, rng: random.Random, point_rate: float = 0.02,
            n_splice: int = 0, library: Optional[List[str]] = None) -> str:
    """Mutate a scaffold. Point substitutions at point_rate, plus optional
    segment splices from a natural-sequence library (the Krasnogor paper
    draws 'favourable scaffold regions' from biological sequences — splicing
    is how those get recombined into a candidate).
    """
    chars = list(seq)
    for i in range(len(chars)):
        if rng.random() < point_rate:
            chars[i] = rng.choice(_BASES)
    out = "".join(chars)
    if library and n_splice and len(out) > 20:
        for _ in range(n_splice):
            donor = rng.choice(library)
            if len(donor) < 12:
                continue
            ln = rng.randint(8, min(40, len(donor), len(out)))
            ds = rng.randint(0, len(donor) - ln)
            os_ = rng.randint(0, len(out) - ln)
            out = out[:os_] + donor[ds:ds + ln] + out[os_ + ln:]
    return out


def _make_individual(cm_template: ContactMap, scaffold: str, k: int,
                     parent: Optional[int], generation: int) -> Individual:
    cm = ContactMap(scaffold=scaffold, staples=cm_template.staples,
                    intended_pairs=cm_template.intended_pairs,
                    helices=cm_template.helices, name=cm_template.name)
    sd = score(cm, k=k)
    vec = sd.vector()
    return Individual(
        scaffold=scaffold,
        objectives=vec,
        total=sum(vec),
        descriptor=_descriptor(scaffold, vec),
        parent=parent,
        generation=generation,
    )


class RandomMutator:
    """The baseline operator: uniform-random point edits + optional splice.

    Wraps `_mutate` behind the `(seq, rng) -> seq` mutator interface so the
    loop can treat random and FM-guided proposers identically.
    """

    template = None  # filled by evolve() so a mutator can score full designs

    def __init__(self, point_rate: float = 0.02, library: Optional[List[str]] = None,
                 n_splice: int = 0):
        self.point_rate = point_rate
        self.library = library
        self.n_splice = n_splice

    def __call__(self, seq: str, rng: random.Random) -> str:
        return _mutate(seq, rng, point_rate=self.point_rate,
                       n_splice=self.n_splice, library=self.library)


class Evo2Mutator:
    """FM-guided mutation operator: proposes scaffold edits with the **Evo 2**
    genomic language model (Arc Institute, 2025) instead of uniform-random
    base substitutions, fusing Evo 2's learned sequence prior with the
    off-target *physics* objective.

    This is the DNA-sequence analog of using ESM to guide protein design:
    Evo 2 was trained on genomes across all domains of life at single-
    nucleotide resolution, so its likelihood is a prior over "natural-like"
    DNA — exactly the signal the Krasnogor paper extracts manually when it
    mines "favourable scaffold regions" from biological sequences. Here the
    FM supplies those regions automatically.

    Heavy deps stay OUT of this module (mirroring how vampnet keeps torch /
    modal lazy). `backend` is any object exposing:
        .score(seqs: list[str]) -> list[float]   # mean per-base log-likelihood
        .generate(prefix, n_tokens, n, temperature) -> list[str]   # optional
    Pass the client from `md/evo2_modal.py` (remote Modal), a local Evo 2,
    or a stub. If `backend` is None or a call raises, the operator
    transparently falls back to random mutation so the loop never stalls.

    mode:
        "score"     propose several candidate window edits locally, then keep
                    the one minimizing  off_target_total - fm_weight * loglik
                    (low frustration AND natural-like). Uses only .score().
        "generate"  let Evo 2 autoregressively write the window from its
                    flanking prefix (Evo 2 is a causal LM). Uses .generate(),
                    then ranks the infills by the same combined cost.
    """

    template = None  # an optional ContactMap giving the real staples to score against

    def __init__(self, backend=None, mode: str = "score", k: int = 8,
                 window: int = 24, n_candidates: int = 6, fm_weight: float = 50.0,
                 temperature: float = 0.7, point_rate: float = 0.04,
                 library: Optional[List[str]] = None, n_splice: int = 0):
        self.backend = backend
        self.mode = mode
        self.k = k
        self.window = window
        self.n_candidates = n_candidates
        self.fm_weight = fm_weight
        self.temperature = temperature
        self.point_rate = point_rate
        self.library = library
        self.n_splice = n_splice

    # -- helpers ---------------------------------------------------------
    def _offtarget_total(self, scaffold: str) -> float:
        tmpl = self.template
        cm = ContactMap(
            scaffold=scaffold,
            staples=tmpl.staples if tmpl else [],
            intended_pairs=tmpl.intended_pairs if tmpl else [],
            name="cand",
        )
        return score(cm, k=self.k).objectives["total"]

    def _random_fallback(self, seq: str, rng: random.Random) -> str:
        return _mutate(seq, rng, point_rate=self.point_rate,
                       n_splice=self.n_splice, library=self.library)

    def _combined_cost(self, candidates: List[str]) -> List[float]:
        # Lower is better: low off-target frustration AND high Evo 2 likelihood.
        offt = [self._offtarget_total(c) for c in candidates]
        try:
            lls = self.backend.score(candidates)
        except Exception:
            lls = [0.0] * len(candidates)
        return [offt[i] - self.fm_weight * float(lls[i]) for i in range(len(candidates))]

    def _window_bounds(self, seq: str, rng: random.Random):
        if len(seq) <= self.window:
            return 0, len(seq)
        start = rng.randint(0, len(seq) - self.window)
        return start, start + self.window

    # -- the two proposal strategies ------------------------------------
    def _score_guided(self, seq: str, rng: random.Random) -> str:
        s, e = self._window_bounds(seq, rng)
        # Candidate windows: the original plus several locally-mutated variants.
        cands = [seq]
        for _ in range(self.n_candidates):
            win = list(seq[s:e])
            for i in range(len(win)):
                if rng.random() < self.point_rate * 4:  # denser edits inside the window
                    win[i] = rng.choice(_BASES)
            cands.append(seq[:s] + "".join(win) + seq[e:])
        costs = self._combined_cost(cands)
        return cands[min(range(len(cands)), key=lambda i: costs[i])]

    def _generate_guided(self, seq: str, rng: random.Random) -> str:
        s, e = self._window_bounds(seq, rng)
        prefix = seq[:s]
        infills = self.backend.generate(
            prefix=prefix, n_tokens=(e - s), n=self.n_candidates,
            temperature=self.temperature,
        )
        cands = [seq] + [prefix + fill[: e - s] + seq[e:] for fill in infills]
        costs = self._combined_cost(cands)
        return cands[min(range(len(cands)), key=lambda i: costs[i])]

    def __call__(self, seq: str, rng: random.Random) -> str:
        if self.backend is None:
            return self._random_fallback(seq, rng)
        try:
            if self.mode == "generate" and hasattr(self.backend, "generate"):
                return self._generate_guided(seq, rng)
            return self._score_guided(seq, rng)
        except Exception:
            # Any backend hiccup (network, OOM, schema drift) degrades to
            # random mutation rather than killing a long evolutionary run.
            return self._random_fallback(seq, rng)


def evolve(seed_cm: ContactMap, generations: int = 200, k: int = 8,
           point_rate: float = 0.02, library: Optional[List[str]] = None,
           n_splice: int = 0, seed: int = 0, gc_bins: int = 8,
           mutator=None, on_step=None) -> dict:
    """Run the recursive-improvement loop on the seed design's scaffold.

    Args:
        seed_cm:     starting design (its scaffold is the genome; staples /
                     routing are held fixed — we optimize the sequence
                     realization, exactly as scaffoldselector does).
        generations: number of mutate-score-admit iterations.
        k:           off-target k-mer length passed to score().
        library:     optional natural-sequence pool to splice from.
        mutator:     optional `(seq, rng) -> seq` proposer. Defaults to
                     RandomMutator (uniform point edits). Pass an
                     Evo2Mutator (see md/evo2_modal.py) for FM-guided
                     proposals. Its `.template` is auto-filled with seed_cm
                     so it can score candidates against the real staples.
        on_step:     optional callback(step_dict) — the hook an MCP agent or
                     the live example uses to watch / steer the loop, mirror
                     of vampnet's adaptive-sampling callback.

    Returns a JSON-serializable record: the MAP-Elites archive, the best
    design found, the improvement curve, and the lineage of the best.
    """
    rng = random.Random(seed)
    if mutator is None:
        mutator = RandomMutator(point_rate=point_rate, library=library, n_splice=n_splice)
    # Give the mutator the full design so FM-guided proposers can score
    # candidates against the real staples, not just the bare scaffold.
    if getattr(mutator, "template", "unset") is None:
        mutator.template = seed_cm
    archive: Dict[Tuple[int, int], Individual] = {}
    history: List[Individual] = []

    def admit(ind: Individual):
        ind.idx = len(history)
        history.append(ind)
        cell = ind.descriptor
        cur = archive.get(cell)
        # Open-ended admission: take the child if the cell is empty, or if it
        # Pareto-dominates / is non-dominated-and-lower-total than the
        # incumbent. Non-dominated novelty is retained even if 'worse' on the
        # scalar total — that is the stepping-stone property.
        if cur is None:
            archive[cell] = ind
        elif dominates(ind.objectives, cur.objectives):
            archive[cell] = ind
        elif not dominates(cur.objectives, ind.objectives) and ind.total < cur.total:
            archive[cell] = ind

    seed_ind = _make_individual(seed_cm, seed_cm.scaffold, k, parent=None, generation=0)
    admit(seed_ind)

    best_curve = []
    for g in range(1, generations + 1):
        # Parent selection: sample a filled archive cell, biased toward
        # lower total (quality) but never deterministic (novelty pressure).
        cells = list(archive.values())
        weights = [1.0 / (1.0 + c.total) for c in cells]
        parent = rng.choices(cells, weights=weights, k=1)[0]

        child_seq = mutator(parent.scaffold, rng)
        child = _make_individual(seed_cm, child_seq, k, parent=parent.idx, generation=g)
        admit(child)

        best = min(archive.values(), key=lambda c: c.total)
        best_curve.append(best.total)
        if on_step is not None:
            on_step({
                "generation": g,
                "archive_size": len(archive),
                "child_total": child.total,
                "best_total": best.total,
                "best_descriptor": list(best.descriptor),
            })

    best = min(archive.values(), key=lambda c: c.total)

    # Recover the lineage (stepping stones) of the best design.
    lineage = []
    node = best
    while node is not None:
        lineage.append({"idx": node.idx, "generation": node.generation,
                        "total": node.total, "descriptor": list(node.descriptor)})
        node = history[node.parent] if node.parent is not None else None
    lineage.reverse()

    return {
        "generations": generations,
        "mutator": type(mutator).__name__,
        "fm_guided": isinstance(mutator, Evo2Mutator) and mutator.backend is not None,
        "archive_size": len(archive),
        "n_niches_possible": gc_bins * 4,
        "seed_total": seed_ind.total,
        "best_total": best.total,
        "improvement": seed_ind.total - best.total,
        "best": {
            "scaffold": best.scaffold,
            "objectives": best.objectives,
            "total": best.total,
            "descriptor": list(best.descriptor),
        },
        "archive": [
            {"descriptor": list(c.descriptor), "total": c.total,
             "objectives": c.objectives, "generation": c.generation}
            for c in archive.values()
        ],
        "improvement_curve": best_curve,
        "best_lineage": lineage,
    }
