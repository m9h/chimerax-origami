"""oxDNA folding-simulation backend on Modal — produces the ASSEMBLY
trajectories that src/assembly.py turns into a folding MSM.

This is the data-generation analog of chimerax-vampnet's md/*_modal.py MD
producers: vampnet runs OpenMM to generate protein MD; here we run oxDNA
(GPU, oxpy/oxDNA-CUDA) to generate an origami folding trajectory, then emit
an .npz that src/assembly.load_trajectory ingests.

  Base image:   modal.Image.debian_slim(python_version="3.12")
  Build:        oxDNA with CUDA (oxpy python bindings). The community build
                is `git clone https://github.com/lorenzo-rovigatti/oxDNA`,
                cmake -DCUDA=1 -DPython=1, make install. Pinned in the image.
  Inputs:       a cadnano/scadnano design converted to oxDNA topology+conf
                (the same conversion src/contactmap.py documents). For a
                folding (not just relaxation) trajectory, run a melting/
                annealing protocol so staples bind over simulated time.
  GPU pin:      A100 (oxDNA-CUDA scales with strand count).
  Status:       v0.1 scaffold — image recipe + run skeleton; first build/run
                retest pending, same convention as the other md/ adapters.

OUTPUT SCHEMA (consumed by src/assembly.load_trajectory)
  <name>_assembly.npz with either:
    occupancy : (F, P) float  — per-frame intended-pair formation (preferred;
                                computed here from the conf using the design's
                                intended_pairs, so the bundle needs no parser)
    coords    : (F, N, 3)     — raw per-nucleotide positions (+ optional
                                'pairs' (P, 2)) if you'd rather featurize in
                                the bundle.

USAGE
  modal run md/oxdna_modal.py::fold \\
      --top design.top --conf design.conf --pairs design_pairs.npy \\
      --steps 2e8 --anneal "60:20" --out design_assembly.npz
"""

from __future__ import annotations

import os

try:
    import modal
except Exception:
    modal = None


APP_NAME = "origami-oxdna"
GPU = os.environ.get("OXDNA_GPU", "A100")

if modal is not None:
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git", "cmake", "build-essential")
        .pip_install("numpy")
        # TODO(v0.2): build oxDNA-CUDA + oxpy in the image:
        #   .run_commands("git clone https://github.com/lorenzo-rovigatti/oxDNA",
        #                 "cd oxDNA && mkdir build && cd build && "
        #                 "cmake -DCUDA=1 -DPython=1 .. && make -j && make install")
    )
    app = modal.App(APP_NAME)
else:  # pragma: no cover
    image = None
    app = None


if app is not None:

    @app.function(image=image, gpu=GPU, timeout=86400)
    def fold(top: str, conf: str, pairs=None, steps: float = 2e8,
             anneal: str = "60:20", out: str = "assembly.npz"):
        """Run an oxDNA annealing/folding simulation and emit the trajectory.

        anneal "T_hi:T_lo" ramps temperature (deg C) so staples hybridize over
        simulated time, giving a genuine *folding* trajectory (not just a
        relaxation of the finished shape). Skeleton below; wire to oxpy once
        the CUDA build is in the image.
        """
        import numpy as np
        # --- skeleton of the oxpy run -----------------------------------
        # import oxpy
        # with oxpy.Context():
        #     inp = oxpy.InputFile(); inp.init_from_filename("input")
        #     manager = oxpy.OxpyManager(inp)
        #     for window in anneal_schedule(anneal):
        #         manager.update_temperature(window.T)
        #         manager.run(window.steps)
        #         confs.append(manager.config_info().current_configuration())
        # coords = np.stack([conf_to_coords(c) for c in confs])  # (F, N, 3)
        # occ = occupancy_from_pairs(coords, pairs)               # (F, P)
        # np.savez(out, occupancy=occ)
        raise NotImplementedError(
            "oxDNA-CUDA build pending in the Modal image (see TODO above); "
            "until then, generate trajectories with "
            "src.assembly.synthetic_assembly_trajectory for the demo/tests."
        )
