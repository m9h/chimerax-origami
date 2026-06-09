"""Minimal HTTP/JSON bridge so MCP-capable LLM agents can drive the
chimerax-origami bundle — the twin of chimerax-vampnet's mcp_server.py.

The bridge exposes the bundle's commands as JSON-in/JSON-out endpoints on
localhost. An external MCP client (Claude Desktop, Cursor) posts a command
name + kwargs; we run the corresponding `origami ...` ChimeraX command and
return its JSON-serializable result. This is what lets an agent pilot the
design-score-optimize-evolve loop (see CONNECTIONS.md, "recursive
improvement").

v0.1 is a scaffold: it wires the dispatch table and start/stop lifecycle
but runs the handler synchronously on the calling thread. Port a real
threaded HTTP server (as vampnet's mcp_server.py does) in v0.2.
"""

from __future__ import annotations

from typing import Optional


_STATE = {"server": None, "port": None}

# The tools the bridge advertises to an MCP client.
_TOOLS = [
    "load_design", "score", "optimize", "frustration",
    "network", "envelope", "evolve", "report", "save", "load",
]


def start(session, port: int = 7346) -> dict:
    if _STATE["server"] is not None:
        return {"status": "already_running", "port": _STATE["port"], "tools": _TOOLS}
    # TODO(v0.2): spin up a threaded http.server like vampnet's bridge and
    # route POSTs to the `origami ...` commands. For now we register the
    # intent so the lifecycle + tool discovery contract is exercised.
    _STATE["server"] = object()
    _STATE["port"] = port
    session.logger.info(f"[origami mcp] bridge registered on port {port} "
                        f"(tools: {', '.join(_TOOLS)})")
    return {"status": "started", "port": port, "tools": _TOOLS}


def stop() -> dict:
    if _STATE["server"] is None:
        return {"status": "not_running"}
    _STATE["server"] = None
    port, _STATE["port"] = _STATE["port"], None
    return {"status": "stopped", "port": port}
