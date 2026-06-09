"""chimerax-origami — off-target-aware DNA-origami sequence design +
lipid-envelope modeling, integrated into the ChimeraX visualization
environment.

The DNA-nanostructure mirror of chimerax-vampnet: same contact-map
abstraction, applied to assembly-landscape frustration (off-target
interactions) instead of protein conformational dynamics.
"""

__version__ = "0.1.0"

from chimerax.core.toolshed import BundleAPI


class _OrigamiBundleAPI(BundleAPI):
    api_version = 1

    @staticmethod
    def register_command(bi, ci, logger):
        # Lazy import so the bundle can be discovered even when optional
        # dependencies aren't yet importable.
        from . import cmd
        cmd.register_commands(logger)


bundle_api = _OrigamiBundleAPI()
