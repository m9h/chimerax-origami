"""Diffusion DNA-origami generator on Modal — the backend behind
src/generate.py.

Self-contained Modal image, same convention as the other md/ adapters
(evo2_modal.py, gnn_shape_modal.py, oxdna_modal.py): heavy deps stay isolated
and never import into the ChimeraX bundle. The bundle only touches the
`DiffusionBackend` client, and only if the user opts into generative routing.

  Base image:   modal.Image.debian_slim(python_version="3.12")
  Pip extras:   torch (cu124), numpy, einops, and the generative model's deps
                (a denoising-diffusion / score model + a strand-routing head).
  Weights:      *** no public checkpoint released with the paper as of
                writing ***. "De novo design of DNA origami with a generative
                diffusion model" (Nat Commun 2026, s41467-026-73578-z)
                describes guided diffusion trained on simulated equilibrium
                conformations + a strand-routing step + integrated structure
                prediction for evaluation, but there is no canonical released
                checkpoint/schema to pin yet. This adapter defines the
                interface and a skeleton; until weights exist, DiffusionBackend
                .sample() raises and src/generate.py falls back to its
                deterministic parametric router.
  GPU pin:      A100.
  Status:       v0.1 scaffold.

PIPELINE (as published)
  target shape (voxels/mesh)
    -> guided reverse diffusion over base-pair positions (sampling toward the
       target while staying on the learned manifold of physical conformations)
    -> strand routing (lay a scaffold path + staples consistent with the
       diffused base-pair graph)
    -> structure prediction for in-loop evaluation (reject designs that don't
       fold to the target)
  => a routing == a ContactMap (scaffold path + staples + intended_pairs),
     which the bundle then scores / shapes / evolves.

USAGE
  modal deploy md/diffusion_modal.py
  from md.diffusion_modal import DiffusionBackend
  from src.generate import RoutingGenerator, Target
  cm = RoutingGenerator(backend=DiffusionBackend()).generate(Target(type="sphere"))
"""

from __future__ import annotations

import os

try:
    import modal
except Exception:
    modal = None


APP_NAME = "origami-diffusion"
GPU = os.environ.get("DIFFUSION_GPU", "A100")

if modal is not None:
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("torch", "numpy", "einops")
    )
    app = modal.App(APP_NAME)
else:  # pragma: no cover
    image = None
    app = None


def _load_model():
    """Instantiate the diffusion generator. No public checkpoint yet (see
    module docstring): raise so src/generate.py uses the parametric fallback.
    Replace with the trained guided-diffusion + routing model when available."""
    raise NotImplementedError(
        "diffusion DNA-origami generator checkpoint not available; "
        "src/generate.py uses the deterministic parametric router meanwhile."
    )


if app is not None:

    @app.cls(image=image, gpu=GPU, timeout=3600, scaledown_window=240)
    class DiffusionService:
        @modal.enter()
        def boot(self):
            self.model = None  # deferred until a checkpoint exists

        @modal.method()
        def sample(self, target, seed):
            """Return a routing dict {scaffold_len, staples, intended_pairs,
            helices} for the target. Skeleton of the published pipeline:
            guided reverse diffusion -> strand routing -> structure-prediction
            filter. Pinned to a checkpoint when available."""
            if self.model is None:
                self.model = _load_model()  # raises until weights exist
            # ... diffuse base-pair positions toward target, route strands ...
            return {}


class DiffusionBackend:
    """Client used by src.generate.RoutingGenerator. Presents .sample(target,
    seed) and routes to the deployed Modal app. If Modal is missing, the app
    isn't deployed, or there's no checkpoint, .sample() raises and the
    generator falls back to the deterministic parametric router.
    """

    def __init__(self, app_name: str = APP_NAME):
        if modal is None:
            raise RuntimeError("modal not installed; cannot reach the diffusion service")
        self._svc = modal.Cls.from_name(app_name, "DiffusionService")()

    def sample(self, target, seed):
        return self._svc.sample.remote(target, seed)
