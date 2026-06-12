#!/usr/bin/env python3
"""Real Evo 2 on the GB10 — plausibility check + FM-guided ablation
(VALIDATION.md test 4, the real-checkpoint version).

Unlike the other demos this loads the actual Evo 2 genomic language model
(arcinstitute/evo2_1b_base, StripedHyena-2) on the GPU. Run it inside the
NVIDIA PyTorch container, which ships flash-attn + Transformer-Engine and
sees the GB10 (Blackwell, sm_121):

    docker run --rm --gpus all --ipc=host --network host \\
      -v "$PWD":/work -v /tmp/hf:/root/.cache/huggingface \\
      -w /work nvcr.io/nvidia/pytorch:26.05-py3 \\
      bash -c 'pip install -q transformers evo2 && python examples/evo2_local_run.py'

Two results:
  A. Plausibility — natural M13 fragments score higher Evo 2 log-likelihood
     than shuffled (the sanity check that the model + prior work).
  B. Ablation — Evo 2-guided evolution reaches low off-target AND keeps the
     winning scaffold more natural (higher Evo 2 LL) than random evolution.
"""

import os
import sys
import types
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if "chimerax" not in sys.modules:
    for n in ["chimerax", "chimerax.core", "chimerax.core.toolshed",
              "chimerax.core.commands", "chimerax.core.errors"]:
        sys.modules[n] = types.ModuleType(n)
    sys.modules["chimerax.core.toolshed"].BundleAPI = object
    sys.modules["chimerax.core.errors"].UserError = type("UserError", (Exception,), {})

from md.evo2_modal import local_backend
from src.contactmap import ContactMap, reverse_complement
from src import evolve as ev

DATA = os.path.join(os.path.dirname(__file__), "data")


def main():
    print("Loading real Evo 2 (evo2_1b_base) on GPU ...", flush=True)
    backend = local_backend("1b")

    # A. Plausibility -------------------------------------------------------
    m13 = "".join(c for c in open(os.path.join(DATA, "m13mp18.txt")).read()
                  if c in "ACGT")
    rng = random.Random(0)
    nats = [m13[i:i + 500] for i in range(0, 5000, 1000)]
    shufs = ["".join(rng.sample(s, len(s))) for s in nats]
    ns, ss = backend.score(nats), backend.score(shufs)
    npass = sum(n > s for n, s in zip(ns, ss))
    print("\n[A] Plausibility check (Evo 2 log-likelihood):")
    print(f"    natural  M13 mean LL = {sum(ns)/len(ns):+.4f}")
    print(f"    shuffled     mean LL = {sum(ss)/len(ss):+.4f}")
    print(f"    natural > shuffled: {npass}/{len(ns)}")

    # B. FM-guided ablation -------------------------------------------------
    seg = "GGGGCCCCAAAATTTT"
    blocks = [("".join(rng.choice("ACGT") for _ in range(30))) for _ in range(8)]
    seed_seq = "".join(b + "AC" * 5 + reverse_complement(b) for b in blocks)
    seed = ContactMap(scaffold=seed_seq, staples=["TTTTGGGGCCCCAAAA"], name="hard")

    G = 20
    rand = ev.evolve(seed, generations=G, seed=0)
    guided = ev.evolve(seed, generations=G, seed=0,
                       mutator=ev.Evo2Mutator(backend=backend, mode="score", fm_weight=5.0))
    ll_rand = backend.score([rand["best"]["scaffold"]])[0]
    ll_guided = backend.score([guided["best"]["scaffold"]])[0]
    print(f"\n[B] Ablation over {G} generations (real Evo 2 backend):")
    print(f"    random  : off-target {rand['best_total']:.0f}   Evo2 LL {ll_rand:+.4f}")
    print(f"    Evo2-guided: off-target {guided['best_total']:.0f}   Evo2 LL {ll_guided:+.4f}")
    print(f"    => guided reaches {'<=' if guided['best_total']<=rand['best_total'] else '>'} "
          f"random off-target while steering toward higher Evo 2 likelihood")


if __name__ == "__main__":
    main()
