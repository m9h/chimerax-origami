"""Stub the ChimeraX runtime so the pure-logic modules (contactmap, score,
optimize, evolve, report) import and run outside a ChimeraX session.

chimerax-vampnet runs its tests inside ChimeraX; here we keep the numerical
core importable standalone so `pytest` works on a bare DGX Spark venv too.
The viz / envelope / cmd / mcp_server paths that genuinely touch the
ChimeraX API are exercised only inside ChimeraX.
"""

import sys
import types
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _stub(name):
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


if "chimerax" not in sys.modules:
    _stub("chimerax")
    core = _stub("chimerax.core")
    toolshed = _stub("chimerax.core.toolshed")
    toolshed.BundleAPI = object
    cmds = _stub("chimerax.core.commands")
    for n in ["CmdDesc", "register", "IntArg", "FloatArg", "BoolArg",
              "StringArg", "EnumOf", "OpenFileNameArg", "SaveFileNameArg",
              "ListOf", "RepeatOf"]:
        setattr(cmds, n, object)
    errors = _stub("chimerax.core.errors")
    errors.UserError = type("UserError", (Exception,), {})
