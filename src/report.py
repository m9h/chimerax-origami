"""Scaffoldselector-style HTML report for chimerax-origami.

The Krasnogor scaffoldselector tool emits "an HTML report containing the
optimal origami sequences". This module produces the equivalent for a
scored design + Pareto optimization, so the bundle's output is drop-in
comparable to the published tool.
"""

from __future__ import annotations

import html
import os
from typing import Optional


def write_html(path: str, scored=None, optimization: Optional[dict] = None) -> dict:
    rows = []
    if scored is not None:
        obj = scored.objectives
        rows.append("<h2>Active design: %s</h2>" % html.escape(scored.cm.name))
        rows.append("<table border=1 cellpadding=4><tr><th>off-target class</th><th>weighted score</th></tr>")
        for key, label in [
            ("j1_staple_wrong_scaffold", "staple &harr; wrong-scaffold"),
            ("j2_scaffold_scaffold", "scaffold &harr; scaffold"),
            ("j3_staple_staple", "staple &harr; staple"),
            ("j4_staple_hairpin", "intra-staple hairpin"),
            ("total", "<b>total</b>"),
        ]:
            rows.append(f"<tr><td>{label}</td><td>{obj[key]:.2f}</td></tr>")
        rows.append("</table>")

    if optimization is not None:
        rows.append("<h2>Pareto front</h2>")
        front = set(optimization.get("pareto_front", []))
        best = optimization.get("best_compromise")
        rows.append("<table border=1 cellpadding=4><tr><th>candidate</th>"
                    "<th>j1</th><th>j2</th><th>j3</th><th>j4</th>"
                    "<th>on front</th><th>best compromise</th></tr>")
        for node in optimization.get("nodes", []):
            o = node["objectives"]
            mark = "&#10003;" if node["id"] in front else ""
            star = "&#9733;" if node["id"] == best else ""
            rows.append(
                f"<tr><td>{html.escape(str(node['name']))}</td>"
                f"<td>{o[0]:.1f}</td><td>{o[1]:.1f}</td><td>{o[2]:.1f}</td><td>{o[3]:.1f}</td>"
                f"<td>{mark}</td><td>{star}</td></tr>"
            )
        rows.append("</table>")

    doc = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>chimerax-origami report</title></head><body>"
        "<h1>chimerax-origami &mdash; off-target design report</h1>"
        + "".join(rows)
        + "<hr><small>Scoring after Shirt-Ediss, Torelli, Navarro &amp; "
          "Krasnogor, Nat Commun 2026.</small></body></html>"
    )
    with open(path, "w") as f:
        f.write(doc)
    return {"path": path, "bytes": os.path.getsize(path)}
