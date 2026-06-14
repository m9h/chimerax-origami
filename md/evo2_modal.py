"""Evo 2 genomic-LM backend on Modal — the FM behind evolve.py's
Evo2Mutator.

Self-contained environment recipe — like every Modal adapter in vampnet's
md/, this builds its own image so heavy deps stay isolated to the tool that
needs them. Nothing here imports at module top into the ChimeraX bundle;
src/evolve.py only ever touches the `Evo2Backend` client, and only if the
user opts into FM-guided mutation.

  Base image:   modal.Image.debian_slim(python_version="3.12")
  Pip extras:   torch (cu124); `evo2` + `vortex` (Arc Institute) for the
                StripedHyena-2 architecture; `huggingface_hub`, `einops`,
                `numpy`. Evo 2 also needs `flash-attn` / Transformer-Engine
                for the attention-hybrid layers on Hopper.
  Weights:      arcinstitute/evo2_7b on HuggingFace (Apache-2.0). The 40B
                checkpoint (arcinstitute/evo2_40b) is a drop-in swap via
                --model-size 40b but needs an 80GB card.
  GPU pin:      runs on the user's GB10 (Blackwell, sm_121) — no H100 needed.
  Status:       VERIFIED — `local_backend('1b')` loads arcinstitute/evo2_1b_base
                and runs inference on the GB10 inside nvcr.io/nvidia/pytorch:26.05-py3
                (flash-attn + Transformer-Engine ship in the image). Held-out
                plausibility check passes: natural M13 fragments score higher
                Evo 2 log-likelihood than shuffled (5/5). Run it with
                `examples/evo2_local_run.py` via the documented docker command.

WHY EVO 2 FOR SCAFFOLD DESIGN
Evo 2 (Brixi et al., bioRxiv 2025.02.18.638918) is a biological foundation
model trained on >9 T tokens of genomes across all domains of life at
single-nucleotide resolution, 1 M-token context. Two uses here, both
exposed as remote functions:

  score(seqs)     -> mean per-base log-likelihood under Evo 2. A prior over
                     "natural-like" DNA. evolve.py's Evo2Mutator subtracts
                     fm_weight * loglik from the off-target cost, so a
                     candidate edit must be BOTH low-frustration AND
                     genome-plausible to be accepted — the automatic version
                     of the Krasnogor paper's hand-mined "favourable
                     scaffold regions".
  generate(...)   -> autoregressive infill of a scaffold window from its
                     flanking prefix (Evo 2 is a causal LM). The "generate"
                     mode of Evo2Mutator.

USAGE (cloud — this is how to run 40B, which OOMs on the single GB10)
  pip install modal && modal setup            # one-time auth
  # one command: plausibility check at the chosen size on a cloud GPU.
  EVO2_MODEL_SIZE=40b modal run md/evo2_modal.py     # -> H200, ~$1–1.5
  EVO2_MODEL_SIZE=7b  modal run md/evo2_modal.py     # -> H100

  # tight-loop use from evolve.py: deploy once, then the client binds by name:
  EVO2_MODEL_SIZE=40b modal deploy md/evo2_modal.py
  from md.evo2_modal import Evo2Backend
  from src.evolve import evolve, Evo2Mutator
  backend = Evo2Backend()
  result = evolve(seed_cm, generations=300,
                  mutator=Evo2Mutator(backend=backend, mode="score"))

COST (Modal, per-second billing, no idle charge; ~$30/mo free credits)
  40B on 1×H200 ($4.54/hr): one run (download+load+score) ~$1–1.5; cached re-run <$1.
  7B on 1×H100 ($3.95/hr): ~$0.50/run. The 77 GB checkpoint persists in the
  'evo2-weights' Modal Volume so it downloads only once.
  Locally: 1B/7B run free on the GB10; 40B does NOT fit (see Status).
"""

from __future__ import annotations

import os

try:
    import modal
except Exception:  # modal is only needed to *define*/run the remote app
    modal = None


APP_NAME = "origami-evo2"
MODEL_SIZE = os.environ.get("EVO2_MODEL_SIZE", "7b")  # "1b" | "7b" | "40b"

# Single-GPU config that fits each size. 40B (~80 GB weights + activations) fits
# on ONE H200 (141 GB) — ~$4.54/hr, cheaper than 2×H100 (~$7.90/hr) and no model
# sharding. KEY: unlike the GB10 (unified memory, where the 77 GB checkpoint
# host-copy + the 80 GB on-device model OOM'd a single 119 GB pool), Modal GPUs
# have SEPARATE VRAM and CPU RAM, so the stock loader's peak doesn't collide —
# 40B that OOM'd locally loads fine here.
_GPU = {"1b": "A100-40GB", "7b": "H100", "40b": "H200"}
GPU = _GPU.get(MODEL_SIZE, "H100")
HF_DIR = "/weights/hf"

if modal is not None:
    # Reuse the NGC PyTorch image already verified to run Evo 2 (ships torch +
    # flash-attn + Transformer-Engine) and just add the evo2 package — avoids
    # building flash-attn from source. If a first `modal deploy` hits an image
    # issue, the fallback is debian_slim + pip install torch evo2 vortex flash-attn.
    image = (
        modal.Image.from_registry("nvcr.io/nvidia/pytorch:26.05-py3")
        .entrypoint([])
        .pip_install("transformers", "evo2")
        .env({"HF_HOME": HF_DIR})
    )
    # Persist HF weights across runs so the 77 GB 40B checkpoint downloads once.
    weights_vol = modal.Volume.from_name("evo2-weights", create_if_missing=True)
    app = modal.App(APP_NAME)
else:  # pragma: no cover - lets the file import without modal installed
    image = None
    weights_vol = None
    app = None


# Evo 2 HuggingFace checkpoint names. The 1B model is published as
# 'evo2_1b_base'; 7B/40B keep the plain name. Verified loading + inference on
# an NVIDIA GB10 (Blackwell, sm_121) inside nvcr.io/nvidia/pytorch:26.05-py3
# (flash-attn + Transformer-Engine ship in the image).
_EVO2_NAMES = {"1b": "evo2_1b_base", "7b": "evo2_7b", "40b": "evo2_40b"}


def _load_model(model_size: str):
    """Load Evo 2 once per container. Returns the evo2.Evo2 object exposing
    .score_sequences() and .generate()."""
    from evo2 import Evo2
    return Evo2(_EVO2_NAMES.get(model_size, f"evo2_{model_size}"))


if app is not None:

    @app.cls(image=image, gpu=GPU, timeout=3600, scaledown_window=300,
             volumes={HF_DIR: weights_vol})
    class Evo2Service:
        @modal.enter()
        def boot(self):
            # First call downloads the checkpoint into the mounted Volume;
            # later cold starts reuse it (no 77 GB re-download).
            self.model = _load_model(MODEL_SIZE)
            weights_vol.commit()

        @modal.method()
        def score(self, seqs):
            """Mean per-base log-likelihood for each sequence (higher == more
            natural under Evo 2). Implemented via the model's scoring API;
            the exact call is pinned to the evo2 release in the image."""
            import numpy as np
            out = []
            for s in seqs:
                # evo2.score_sequences returns per-sequence log-likelihoods.
                ll = self.model.score_sequences([s])
                out.append(float(np.mean(ll)))
            return out

        @modal.method()
        def generate(self, prefix, n_tokens, n, temperature=0.7):
            """Autoregressive infill: n continuations of `prefix`, each
            n_tokens long, sampled at the given temperature."""
            outs = []
            for _ in range(int(n)):
                res = self.model.generate(
                    prompt_seqs=[prefix],
                    n_tokens=int(n_tokens),
                    temperature=float(temperature),
                )
                seq = res.sequences[0] if hasattr(res, "sequences") else res[0]
                outs.append(str(seq))
            return outs

    @app.function(image=image)
    def score_cli(seqs: str):
        """`modal run md/evo2_modal.py::score_cli --seqs "AAA,GGG"`"""
        svc = Evo2Service()
        scores = svc.score.remote([s.strip() for s in seqs.split(",") if s.strip()])
        for s, v in zip(seqs.split(","), scores):
            print(f"{v:+.4f}  {s.strip()[:40]}")
        return scores

    @app.local_entrypoint()
    def main():
        """One command:  EVO2_MODEL_SIZE=40b modal run md/evo2_modal.py

        Runs the plausibility check on real M13 (natural vs shuffled) against
        the chosen model size on the cloud GPU. Reads M13 locally and sends the
        fragments to the remote scorer, so nothing but sequences leaves the box.
        """
        import os as _os
        import random
        here = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        m13 = "".join(c for c in open(
            _os.path.join(here, "examples/data/m13mp18.txt")).read() if c in "ACGT")
        rng = random.Random(0)
        nats = [m13[i:i + 500] for i in range(0, 5000, 1000)]
        shufs = ["".join(rng.sample(s, len(s))) for s in nats]
        svc = Evo2Service()
        ns = svc.score.remote(nats)
        ss = svc.score.remote(shufs)
        npass = sum(n > s for n, s in zip(ns, ss))
        print(f"\nEvo 2 ({MODEL_SIZE}, {GPU}) plausibility on real M13:")
        print(f"  natural  mean LL = {sum(ns)/len(ns):+.4f}")
        print(f"  shuffled mean LL = {sum(ss)/len(ss):+.4f}")
        print(f"  natural > shuffled: {npass}/{len(ns)}")


class Evo2Backend:
    """Client used by src.evolve.Evo2Mutator. Presents .score()/.generate()
    and routes to the deployed Modal app. Construct after `modal deploy
    md/evo2_modal.py`. Falls back cleanly: if Modal isn't importable or the
    lookup fails, the methods raise, and Evo2Mutator degrades to random
    mutation (so a run never hard-fails on a missing backend).
    """

    def __init__(self, app_name: str = APP_NAME):
        if modal is None:
            raise RuntimeError("modal not installed; cannot reach the Evo 2 service")
        self._cls = modal.Cls.from_name(app_name, "Evo2Service")
        self._svc = self._cls()

    def score(self, seqs):
        return self._svc.score.remote(list(seqs))

    def generate(self, prefix, n_tokens, n, temperature=0.7):
        return self._svc.generate.remote(prefix, n_tokens, n, temperature)


def local_backend(model_size: str = MODEL_SIZE):
    """No-Modal path: load Evo 2 in-process (e.g. evo2_7b on the GB10's
    120 GB unified memory) and return an object with the same .score() /
    .generate() interface as Evo2Backend. Heavy import stays inside.
    """
    model = _load_model(model_size)

    class _Local:
        def score(self, seqs):
            # evo2.score_sequences returns one mean-log-likelihood per sequence;
            # batch the call (verified: natural M13 > shuffled, 5/5).
            return [float(x) for x in model.score_sequences(list(seqs))]

        def generate(self, prefix, n_tokens, n, temperature=0.7):
            outs = []
            for _ in range(int(n)):
                res = model.generate(prompt_seqs=[prefix], n_tokens=int(n_tokens),
                                     temperature=float(temperature))
                outs.append(str(res.sequences[0] if hasattr(res, "sequences") else res[0]))
            return outs

    return _Local()
