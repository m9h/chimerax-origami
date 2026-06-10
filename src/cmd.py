"""ChimeraX command registration for the chimerax-origami bundle.

Every command returns a JSON-serializable dict (when it returns anything)
so the ChimeraX MCP-server bundle can route results to an external LLM
agent — identical contract to chimerax-vampnet. The numerical work lives in
contactmap.py / score.py / optimize.py / viz.py / envelope.py / evolve.py;
this module is the registration + dispatch layer.
"""

import os

from chimerax.core.commands import (
    CmdDesc,
    register,
    IntArg,
    FloatArg,
    BoolArg,
    StringArg,
    EnumOf,
    OpenFileNameArg,
    SaveFileNameArg,
    RepeatOf,
)
from chimerax.core.errors import UserError


# Session-scoped state: the active scored design + last optimization result.
_SCORED_KEY = "_origami_scored"
_OPT_KEY = "_origami_opt"


def _scored_get(session):
    return getattr(session, _SCORED_KEY, None)


def _scored_set(session, scored):
    setattr(session, _SCORED_KEY, scored)


# ----------------------------------------------------------------------
# Command implementations.
# ----------------------------------------------------------------------
def cmd_generate(session, type="bundle", n_helices=6, length=64, sequence=None, seed=0):
    """Generate a DNA-origami routing for a target shape (diffusion generator,
    Nat Commun 2026; deterministic parametric router fallback) and set it as
    the active design. With `sequence`, threads a scaffold through the routing
    and derives complementary staples. Returns the design summary.
    """
    from . import generate, contactmap
    target = generate.Target(type=type, n_helices=int(n_helices), length=int(length))
    cm = generate.generate_routing(target, backend=None, seed=int(seed))
    if sequence:
        seq = sequence
        if os.path.isfile(sequence):
            with open(sequence) as fh:
                seq = "".join(ln.strip() for ln in fh if not ln.startswith(">"))
        cm = generate.apply_scaffold(cm, seq)
    cm.source_format = "generated"
    contactmap._design_set(session, cm)
    return cm.summary()


def cmd_load_design(session, path, format="auto", sequence=None):
    """Load a cadnano / scadnano / oxDNA / contactmap design and build its
    contact map. `sequence` (optional) applies a scaffold sequence along the
    recovered routing — a path to a .txt/.fasta file, or the raw sequence
    string (e.g. M13mp18). Returns the design summary dict.
    """
    from . import contactmap
    seq = None
    if sequence:
        if os.path.isfile(sequence):
            with open(sequence) as fh:
                lines = [ln.strip() for ln in fh if not ln.startswith(">")]
            seq = "".join(lines)
        else:
            seq = sequence.strip()
    return contactmap.load_design(session, path, format, scaffold_sequence=seq)


def cmd_score(session, k=8):
    """Score the four off-target classes for the active design.

    Returns {"name", "k", "objectives": {j1..j4, total}, "n_hotspots"}.
    """
    from . import contactmap, score
    cm = contactmap.design_get(session)
    scored = score.score(cm, k=int(k))
    _scored_set(session, scored)
    return scored.summary()


def cmd_optimize(session, candidates=None, k=8):
    """Multi-objective Pareto selection over candidate scaffold sequences.

    candidates: list of paths to alternative designs (same routing, different
    scaffold). If omitted, the active design is scored alone (trivial front).
    Returns the Pareto-front graph dict.
    """
    from . import contactmap, optimize
    cms = []
    if candidates:
        from . import contactmap as cmmod
        for p in candidates:
            cmmod.load_design(session, p, "auto")
            cms.append(contactmap.design_get(session))
    else:
        cms = [contactmap.design_get(session)]
    result = optimize.optimize(cms, k=int(k))
    setattr(session, _OPT_KEY, result)
    return result


def cmd_frustration(session):
    """Color the active structure (if any) by off-target frustration density.

    Returns the frustration profile summary.
    """
    scored = _scored_get(session)
    if scored is None:
        raise UserError("no scored design — run `origami score` first")
    from . import viz
    structure = scored.cm.__dict__.get("structure")
    return viz.color_by_frustration(session, scored, structure=structure)


def cmd_network(session):
    """Return the off-target interaction map as a structured graph.

    Mirrors `vampnet network`. Returns nodes (off-target classes with their
    weighted totals) + edges (the worst per-class hotspots).
    """
    scored = _scored_get(session)
    if scored is None:
        raise UserError("no scored design — run `origami score` first")
    obj = scored.objectives
    nodes = [
        {"id": "j1", "label": "staple<->wrong-scaffold", "weight": obj["j1_staple_wrong_scaffold"]},
        {"id": "j2", "label": "scaffold<->scaffold", "weight": obj["j2_scaffold_scaffold"]},
        {"id": "j3", "label": "staple<->staple", "weight": obj["j3_staple_staple"]},
        {"id": "j4", "label": "staple hairpin", "weight": obj["j4_staple_hairpin"]},
    ]
    edges = []
    for key, hs in scored.hotspots.items():
        for h in hs[:10]:
            edges.append({"channel": key, "hotspot": list(h)})
    return {"nodes": nodes, "edges": edges, "total": obj["total"]}


def cmd_envelope(session, density=180.0):
    """Plan a lipid-bilayer delivery envelope for the active design.

    Returns the envelope plan (handle count, carrier staples, predicted
    in-vivo effects). See envelope.py / CONNECTIONS.md.
    """
    from . import contactmap, envelope
    cm = contactmap.design_get(session)
    structure = cm.__dict__.get("structure")
    return envelope.design_envelope(session, cm, handle_density_nm2=float(density),
                                    structure=structure)


def cmd_shape(session, n_ensemble=8, target=None, color=True):
    """Predict the 3D shape + per-base-pair flexibility (RMSF) of the active
    design with the DGNN geometric forward model (Truong-Quoc et al.,
    Nat Mater 2024). Falls back to a deterministic lattice placement when no
    DGNN backend is configured. Returns the shape summary dict.
    """
    from . import contactmap, shape
    cm = contactmap.design_get(session)
    target_coords = None
    if target:
        import numpy as np
        d = np.load(target)
        target_coords = d["coords"] if "coords" in d.files else d[d.files[0]]
    # The bundle uses the deterministic fallback by default; advanced users
    # pass a DGNNBackend (md/gnn_shape_modal.py) from a script.
    result = shape.predict_shape(cm, backend=None, n_ensemble=int(n_ensemble),
                                 target_coords=target_coords)
    structure = cm.__dict__.get("structure")
    if color and structure is not None:
        from . import viz
        viz.color_by_flexibility(session, result, structure=structure)
    return result.summary()


def cmd_assembly_msm(session, path, lag=5, n_states=3, cutoff=2.0, backend="auto"):
    """Build a Markov state model of an oxDNA ASSEMBLY trajectory — the
    method bridge to chimerax-vampnet. Featurizes the trajectory as base-pair
    contact occupancy, fits an MSM (deeptime if available, numpy fallback),
    and returns the folding-intermediate states + transition graph + trap
    states. Returns a JSON dict (see assembly.AssemblyMSM.summary +
    transition_graph).
    """
    from . import assembly, contactmap
    loaded = assembly.load_trajectory(path)
    if loaded[0] == "occupancy":
        features = loaded[1]
    else:
        coords = loaded[1]
        pairs = loaded[2]
        if pairs is None:
            cm = contactmap.design_get(session)
            pairs = assembly.pairs_from_design(cm)
            if not pairs:
                raise UserError("trajectory has coords but no pairs, and the "
                                "active design has no intended_pairs; provide "
                                "'pairs' in the npz or an occupancy array")
        features = assembly.featurize_assembly(coords, pairs, cutoff=float(cutoff))
    msm = assembly.fit_assembly_msm(features, lag=int(lag), n_states=int(n_states),
                                    backend=backend)
    out = msm.summary()
    out["transition_graph"] = msm.transition_graph()
    return out


def cmd_fold_msm(session, n_frames=3000, lag=5, n_states=3, seed=0, k=8):
    """Predict the folding MSM of the active design WITHOUT oxDNA: a cheap
    kinetic emulator (driven by the static off-target score) generates an
    assembly trajectory, then the same MSM pipeline as `assembly_msm` recovers
    folding-intermediate states, traps, and the predicted folding yield.
    Returns the MSM summary + folding_yield + transition graph.
    """
    from . import contactmap, assembly
    cm = contactmap.design_get(session)
    feats = assembly.simulate_folding(cm, n_frames=int(n_frames), seed=int(seed), k=int(k))
    msm = assembly.fit_assembly_msm(feats, lag=int(lag), n_states=int(n_states),
                                    backend="auto")
    out = msm.summary()
    out["folding_yield"] = float(feats[feats.shape[0] // 2:].mean())
    out["transition_graph"] = msm.transition_graph()
    return out


def cmd_evolve(session, generations=200, k=8, point_rate=0.02, seed=0):
    """Run the Sakana-style recursive-improvement loop on the active
    design's scaffold. Returns the archive + best design + improvement curve.
    """
    from . import contactmap, evolve
    cm = contactmap.design_get(session)

    def _log(step):
        if step["generation"] % 25 == 0:
            session.logger.info(
                f"[origami evolve] gen {step['generation']} "
                f"archive={step['archive_size']} best={step['best_total']:.1f}"
            )

    result = evolve.evolve(cm, generations=int(generations), k=int(k),
                          point_rate=float(point_rate), seed=int(seed),
                          on_step=_log)
    return result


def cmd_report(session, path):
    """Emit a scaffoldselector-style HTML report of the optimization.

    Returns {"path": str, "bytes": int}.
    """
    opt = getattr(session, _OPT_KEY, None)
    scored = _scored_get(session)
    from . import report
    return report.write_html(path, scored=scored, optimization=opt)


def cmd_save(session, path):
    """Save the scored design + last Pareto front to a JSON file."""
    import json
    scored = _scored_get(session)
    if scored is None:
        raise UserError("no scored design — run `origami score` first")
    payload = {
        "scored": scored.summary(),
        "scaffold": scored.cm.scaffold,
        "staples": scored.cm.staples,
        "optimization": getattr(session, _OPT_KEY, None),
    }
    with open(path, "w") as f:
        json.dump(payload, f)
    import os
    return {"path": path, "bytes": os.path.getsize(path)}


def cmd_load(session, path):
    """Load a previously saved scored design + Pareto front."""
    import json
    from . import contactmap, score
    with open(path) as f:
        payload = json.load(f)
    cm = contactmap.ContactMap(scaffold=payload["scaffold"],
                               staples=payload.get("staples", []))
    contactmap._design_set(session, cm)
    scored = score.score(cm)
    _scored_set(session, scored)
    if payload.get("optimization") is not None:
        setattr(session, _OPT_KEY, payload["optimization"])
    return scored.summary()


def cmd_mcp_serve(session, port=7346):
    """Start the MCP bridge so MCP-capable LLM agents can drive this bundle."""
    from . import mcp_server
    return mcp_server.start(session, port=int(port))


def cmd_mcp_stop(session):
    """Stop the MCP bridge."""
    from . import mcp_server
    return mcp_server.stop()


# ----------------------------------------------------------------------
# Command descriptors.
# ----------------------------------------------------------------------
_DESC_GENERATE = CmdDesc(
    keyword=[("type", EnumOf(["bundle", "sheet", "rod"])),
             ("n_helices", IntArg), ("length", IntArg),
             ("sequence", StringArg), ("seed", IntArg)],
    synopsis=("Generate a DNA-origami routing for a target shape (diffusion "
              "generator, Nat Commun 2026; parametric-router fallback) and make "
              "it the active design. sequence threads a scaffold + derives "
              "staples. Example: origami generate type bundle n_helices 6 "
              "length 64 sequence m13.txt"),
)
_DESC_LOAD_DESIGN = CmdDesc(
    required=[("path", OpenFileNameArg)],
    keyword=[("format", EnumOf(["auto", "cadnano", "scadnano", "oxdna", "contactmap"])),
             ("sequence", StringArg)],
    synopsis=("Load a DNA-origami design and build its base-pair contact map. "
              "format=auto infers from extension/content. sequence applies a "
              "scaffold sequence (file path or literal, e.g. M13mp18) along the "
              "routing. Example: origami load_design smiley.json sequence m13.txt"),
)
_DESC_SCORE = CmdDesc(
    keyword=[("k", IntArg)],
    synopsis=("Score the four off-target interaction classes (j1 staple<->wrong-"
              "scaffold, j2 scaffold<->scaffold, j3 staple<->staple, j4 hairpin) "
              "via a k-mer reverse-complement index. k~7-9. Example: origami score k 8"),
)
_DESC_OPTIMIZE = CmdDesc(
    keyword=[("candidates", RepeatOf(OpenFileNameArg)), ("k", IntArg)],
    synopsis=("Multi-objective Pareto selection over candidate scaffold "
              "sequences (same routing, different sequence). Returns the "
              "non-dominated front + best compromise. Example: origami "
              "optimize candidates m13.json,phix174.json,synthetic.json"),
)
_DESC_FRUSTRATION = CmdDesc(
    synopsis=("Color the loaded nanostructure by off-target (frustration) "
              "density. Requires `origami score` first."),
)
_DESC_NETWORK = CmdDesc(
    synopsis=("Return the off-target interaction map as a JSON graph "
              "(classes + worst hotspots). Requires `origami score` first."),
)
_DESC_ENVELOPE = CmdDesc(
    keyword=[("density", FloatArg)],
    synopsis=("Plan a virus-inspired lipid-bilayer envelope (Perrault & Shih "
              "2014): handle count at the target density (default 1 per "
              "180 nm^2), carrier staples, predicted in-vivo effects. "
              "Example: origami envelope density 180"),
)
_DESC_SHAPE = CmdDesc(
    keyword=[("n_ensemble", IntArg), ("target", OpenFileNameArg), ("color", BoolArg)],
    synopsis=("Predict the 3D shape + per-base-pair flexibility (RMSF) of the "
              "active design with the DGNN geometric forward model (Truong-Quoc "
              "et al., Nat Mater 2024); deterministic lattice fallback if no "
              "backend. target is an optional .npz of target coords for an "
              "RMSD. Example: origami shape n_ensemble 8 target cube.npz"),
)
_DESC_ASSEMBLY_MSM = CmdDesc(
    required=[("path", OpenFileNameArg)],
    keyword=[("lag", IntArg), ("n_states", IntArg), ("cutoff", FloatArg),
             ("backend", EnumOf(["auto", "deeptime", "numpy"]))],
    synopsis=("Markov state model of an oxDNA assembly trajectory (.npz with "
              "'occupancy' or 'coords'). Recovers folding-intermediate states "
              "+ kinetic traps via the same featurize->VAMPnet/MSM pipeline as "
              "chimerax-vampnet. Example: origami assembly_msm fold.npz lag 5 "
              "n_states 3"),
)
_DESC_FOLD_MSM = CmdDesc(
    keyword=[("n_frames", IntArg), ("lag", IntArg), ("n_states", IntArg),
             ("seed", IntArg), ("k", IntArg)],
    synopsis=("Predict the active design's folding MSM without oxDNA via a "
              "score-driven kinetic emulator. Returns folding-intermediate "
              "states, traps, and predicted folding yield. Example: origami "
              "fold_msm n_frames 3000 lag 5"),
)
_DESC_EVOLVE = CmdDesc(
    keyword=[("generations", IntArg), ("k", IntArg),
             ("point_rate", FloatArg), ("seed", IntArg)],
    synopsis=("Run the Sakana-style recursive-improvement loop (open-ended "
              "MAP-Elites over scaffold sequences, off-target score as "
              "fitness). Returns the stepping-stone archive + best design. "
              "Example: origami evolve generations 300 seed 0"),
)
_DESC_REPORT = CmdDesc(
    required=[("path", SaveFileNameArg)],
    synopsis=("Write a scaffoldselector-style HTML report of the optimization "
              "to path. Example: origami report /tmp/design_report.html"),
)
_DESC_SAVE = CmdDesc(
    required=[("path", SaveFileNameArg)],
    synopsis="Save the scored design + Pareto front to a JSON file.",
)
_DESC_LOAD = CmdDesc(
    required=[("path", OpenFileNameArg)],
    synopsis="Load a previously saved scored design + Pareto front.",
)
_DESC_MCP_SERVE = CmdDesc(
    keyword=[("port", IntArg)],
    synopsis=("Start the MCP bridge so external LLM agents (Claude Desktop, "
              "Cursor) can drive this bundle. Default port 7346."),
)
_DESC_MCP_STOP = CmdDesc(synopsis="Stop the MCP bridge.")


def register_commands(logger):
    register("origami generate", _DESC_GENERATE, cmd_generate, logger=logger)
    register("origami load_design", _DESC_LOAD_DESIGN, cmd_load_design, logger=logger)
    register("origami score", _DESC_SCORE, cmd_score, logger=logger)
    register("origami optimize", _DESC_OPTIMIZE, cmd_optimize, logger=logger)
    register("origami frustration", _DESC_FRUSTRATION, cmd_frustration, logger=logger)
    register("origami network", _DESC_NETWORK, cmd_network, logger=logger)
    register("origami envelope", _DESC_ENVELOPE, cmd_envelope, logger=logger)
    register("origami shape", _DESC_SHAPE, cmd_shape, logger=logger)
    register("origami assembly_msm", _DESC_ASSEMBLY_MSM, cmd_assembly_msm, logger=logger)
    register("origami fold_msm", _DESC_FOLD_MSM, cmd_fold_msm, logger=logger)
    register("origami evolve", _DESC_EVOLVE, cmd_evolve, logger=logger)
    register("origami report", _DESC_REPORT, cmd_report, logger=logger)
    register("origami save", _DESC_SAVE, cmd_save, logger=logger)
    register("origami load", _DESC_LOAD, cmd_load, logger=logger)
    register("origami mcp serve", _DESC_MCP_SERVE, cmd_mcp_serve, logger=logger)
    register("origami mcp stop", _DESC_MCP_STOP, cmd_mcp_stop, logger=logger)
