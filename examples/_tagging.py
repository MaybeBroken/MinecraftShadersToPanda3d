"""Bulk-tag a loaded model's sub-parts by name pattern.

A real game scene is never tagged object-by-object by hand — it's tagged by
naming/material convention ("everything named Leaf_* waves", "everything
under Rock_* is static terrain"), the same way a Minecraft resource pack maps
block *ids* to render behaviour. This is that convention applied to any
Panda3D model: match each of its GeomNodes against an ordered list of
(regex, render_type) rules and tag whichever rule matches first.
"""

from __future__ import annotations

import re

__all__ = ["tag_by_pattern"]


def tag_by_pattern(pipe, root, rules, *, default: str | None = None) -> dict[str, int]:
    """Tag every GeomNode under ``root`` using the first matching rule.

    ``rules`` is an ordered list of ``(regex, render_type)`` pairs tested
    against each GeomNode's name. ``default`` (if given) is applied to any
    GeomNode that matched nothing, so a scene can be tagged "mostly terrain,
    except these specific patterns" in one pass.

    Returns a ``{render_type: count}`` summary, mostly for a demo's own
    console/status output — bulk-tagging a real scene should be legible at
    a glance, not a silent side effect.
    """
    compiled = [(re.compile(pattern), render_type) for pattern, render_type in rules]
    counts: dict[str, int] = {}

    for geom_np in root.find_all_matches("**/+GeomNode"):
        name = geom_np.node().get_name()
        render_type = next((rt for regex, rt in compiled if regex.search(name)), default)
        if render_type is None:
            continue
        pipe.set_render_type(geom_np, render_type)
        counts[render_type] = counts.get(render_type, 0) + 1

    return counts
