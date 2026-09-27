"""Saved option files ("prefs"): a pack's option values plus which pack and
profile they were made for.

    # mcshader preferences
    @pack=BSL_v10.0.zip
    @profile=ULTRA

    BLOOM=true
    SHADOW_QUALITY=2
    ...

Pass one to ``mcshader.init(..., prefs="my.prefs")`` and it is applied before
the first build, so the pack compiles once, with the saved values. Save the
current values with ``app.pipe.save_prefs()``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

__all__ = ["PACK_KEY", "PROFILE_KEY", "Prefs", "parse_prefs", "format_prefs"]

PACK_KEY = "@pack"
PROFILE_KEY = "@profile"


@dataclass(frozen=True)
class Prefs:
    pack: str      # "" when the file names no pack
    profile: str   # "" when the file names no profile
    text: str      # the whole file, for ShaderOptions.loads (it skips the header)

    def matches(self, pack_name: str) -> bool:
        """Whether these prefs were written for ``pack_name`` (or any pack)."""
        return not self.pack or self.pack == pack_name


def parse_prefs(text: str) -> Prefs:
    header: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("@") and "=" in line:
            key, _, value = line.partition("=")
            header[key.strip()] = value.strip()
    return Prefs(header.get(PACK_KEY, ""), header.get(PROFILE_KEY, ""), text)


def format_prefs(options_text: str, *, pack: str, profile: str | None,
                 title: str = "mcshader preferences") -> str:
    return "\n".join([
        f"# {title}",
        f"# {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"{PACK_KEY}={pack}",
        f"{PROFILE_KEY}={profile or ''}",
        "",
        options_text.rstrip("\n"),
        "",
    ])
