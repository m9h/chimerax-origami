"""DGNN origami shape predictor on Modal — the geometric forward model
behind src/shape.py.

Self-contained Modal image, same convention as the other md/ adapters
(evo2_modal.py, and vampnet's *_modal.py): heavy deps (torch, PyTorch
Geometric) stay isolated and never import into the ChimeraX bundle. The
bundle only touches the `DGNNBackend` client, and only if the user opts into
geometric prediction.

  Base image:   modal.Image.debian_slim(python_version="3.12")
  Pip extras:   torch (cu124), torch_geometric (+ torch_scatter/sparse),
                numpy, einops
  Weights:      *** no public checkpoint released with the paper as of
                writing ***. Truong-Quoc, Lee, Kim & Kim, Nat Mater 2024
                (DGNN) describe the architecture and a hybrid data-driven +
                physics-informed training scheme but, like vampnet's MarS-FM
                adapter, there is no canonical released checkpoint/schema we
                can pin yet. This adapter therefore defines the *interface*
                and a thin reimplementation skeleton; until weights exist,
                DGNNBackend.predict() raises and src/shape.py degrades to its
                deterministic lattice fallback. Swap _load_model() once a
                checkpoint is available (or train one on oxDNA relaxations —
                see md/README.md).
  GPU pin:      A100 (monomer inference is light; supramolecular assemblies
                of hundreds of blocks want the larger card).
  Status:       v0.1 scaffold.

ARCHITECTURE (as published)
  node              = a base pair (position + orientation)
  structural edge   = backbone bond / crossover; features carry the
                      sequence-dependent base-pair-step mechanics + connection
                      type
  electrostatic edge= distance-dependent repulsion (DNA backbone charge)
  blocks            = 5 mechanical-relaxation blocks (drive toward minimum
                      mechanical energy) + 1 electrostatic refinement block
  inference         = ensemble strategy -> near-real-time monomer shape;
                      unsupervised refinement -> tens-to-hundreds-of-block
                      supramolecular assemblies

USAGE
  modal deploy md/gnn_shape_modal.py
  # then:
  from md.gnn_shape_modal import DGNNBackend
  from src.shape import predict_shape
  res = predict_shape(cm, backend=DGNNBackend(), n_ensemble=8)
  print(res.summary())
"""

from __future__ import annotations

import os

try:
    import modal
except Exception:
    modal = None


APP_NAME = "origami-dgnn"
GPU = os.environ.get("DGNN_GPU", "A100")

if modal is not None:
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "torch",
            "numpy",
            "einops",
            # PyG stack for the message-passing relaxation/refinement blocks.
            "torch_geometric",
        )
    )
    app = modal.App(APP_NAME)
else:  # pragma: no cover
    image = None
    app = None


def _load_model():
    """Instantiate the DGNN. No public checkpoint yet (see module docstring):
    raise a clear error so src/shape.py falls back deterministically. Replace
    this body with `DGNN.from_pretrained(...)` once weights exist."""
    raise NotImplementedError(
        "DGNN checkpoint not available — train on oxDNA relaxations or wait "
        "for the SNU release; src/shape.py uses the lattice fallback meanwhile."
    )


if app is not None:

    @app.cls(image=image, gpu=GPU, timeout=1800, scaledown_window=240)
    class DGNNService:
        @modal.enter()
        def boot(self):
            # Deferred until a checkpoint exists; until then predict() will
            # raise on the client side and shape.py falls back.
            self.model = None

        @modal.method()
        def predict(self, graph, n_ensemble):
            """Return an (n_ensemble, N, 3) array of predicted base-pair
            positions. `graph` is the dict from src.shape.build_graph.

            Skeleton of the published forward pass: build node/edge tensors,
            run the 5 mechanical-relaxation blocks + electrostatic refinement,
            repeat with ensemble noise. Pinned to a checkpoint when available.
            """
            if self.model is None:
                self.model = _load_model()  # raises until weights exist
            import numpy as np  # pragma: no cover - unreachable until weights
            # ... message-passing relaxation over graph["edge_index"] with
            #     graph["edge_stiffness"] as edge features, n_ensemble draws ...
            return np.zeros((n_ensemble, graph["n_nodes"], 3), dtype=np.float32)


class DGNNBackend:
    """Client used by src.shape.ShapePredictor. Presents .predict(graph,
    n_ensemble) and routes to the deployed Modal app. If Modal is missing,
    the app isn't deployed, or the model has no checkpoint, .predict() raises
    and ShapePredictor degrades to the deterministic lattice fallback.
    """

    def __init__(self, app_name: str = APP_NAME):
        if modal is None:
            raise RuntimeError("modal not installed; cannot reach the DGNN service")
        self._svc = modal.Cls.from_name(app_name, "DGNNService")()

    def predict(self, graph, n_ensemble):
        return self._svc.predict.remote(graph, n_ensemble)
