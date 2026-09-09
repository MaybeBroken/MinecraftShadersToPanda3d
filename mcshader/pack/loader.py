"""Load a Minecraft (OptiFine/Iris) shaderpack from a folder or ``.zip``.

Understands the usual layout: a ``shaders/`` root containing per-dimension
folders (``world0``, ``world1``, ``world-1``), a shared ``program/`` folder
(Iris ``.glsl``), a ``lib/`` folder of includes, and loose ``.vsh``/``.fsh``
pairs. The loader's job is to locate that root, enumerate the programs inside,
and hand raw stage sources (plus an include-resolution file map) to the
decompiler.
"""

from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass

__all__ = ["ShaderPack", "ProgramSource"]

_WORLDS = ("", "world0", "world1", "world-1")
_STAGE_EXTS = (".vsh", ".fsh", ".glsl", ".vert", ".frag")


@dataclass
class ProgramSource:
    """Raw sources for one program, before translation."""
    name: str
    world: str
    vertex: str | None
    fragment: str | None
    combined: str | None  # Iris .glsl with VSH/FSH guards


class ShaderPack:
    def __init__(self, files: dict[str, str], *, name: str = "shaderpack",
                 base_dir: str | None = None):
        #: relative-path -> text, relative to the shaders root.
        self.files = files
        self.name = name
        #: filesystem root when loaded from a directory (enables disk includes).
        self.base_dir = base_dir

    # -- construction ----------------------------------------------------
    @classmethod
    def from_path(cls, path: str) -> "ShaderPack":
        """Load from a directory or a ``.zip`` shaderpack."""
        if os.path.isdir(path):
            return cls._from_dir(path)
        if zipfile.is_zipfile(path):
            return cls._from_zip(path)
        raise ValueError(f"{path!r} is neither a directory nor a zip shaderpack")

    @classmethod
    def _shaders_root(cls, entries: list[str]) -> str:
        """Find the path prefix that ends at the 'shaders' directory."""
        for entry in entries:
            parts = entry.replace("\\", "/").split("/")
            if "shaders" in parts:
                i = parts.index("shaders")
                return "/".join(parts[: i + 1]) + "/"
        return ""  # already at root

    @classmethod
    def _from_dir(cls, path: str) -> "ShaderPack":
        all_files: list[str] = []
        for dirpath, _, filenames in os.walk(path):
            for fn in filenames:
                all_files.append(os.path.relpath(os.path.join(dirpath, fn), path))
        prefix = cls._shaders_root(all_files)
        files: dict[str, str] = {}
        for rel in all_files:
            norm = rel.replace("\\", "/")
            if not norm.startswith(prefix):
                continue
            key = norm[len(prefix):]
            if not key or not key.endswith(
                _STAGE_EXTS + (".inc", ".properties", ".glsl")
            ):
                continue
            try:
                with open(os.path.join(path, rel), "r", encoding="utf-8") as fh:
                    files[key] = fh.read()
            except (OSError, UnicodeDecodeError):
                continue
        base = os.path.join(path, prefix.rstrip("/")) if prefix else path
        return cls(files, name=os.path.basename(os.path.normpath(path)), base_dir=base)

    @classmethod
    def _from_zip(cls, path: str) -> "ShaderPack":
        files: dict[str, str] = {}
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            prefix = cls._shaders_root(names)
            for name in names:
                if name.endswith("/") or not name.startswith(prefix):
                    continue
                key = name[len(prefix):]
                if not key.endswith(_STAGE_EXTS + (".inc", ".properties")):
                    continue
                try:
                    files[key] = zf.read(name).decode("utf-8")
                except (KeyError, UnicodeDecodeError):
                    continue
        return cls(files, name=os.path.splitext(os.path.basename(path))[0])

    # -- enumeration -----------------------------------------------------
    def programs(self) -> list[str]:
        """Distinct program entry names (excludes ``lib/`` includes)."""
        names: set[str] = set()
        for key in self.files:
            head, _, base = key.rpartition("/")
            top = head.split("/", 1)[0] if head else ""
            # Program entry points live at the pack root, in program/, or in a
            # per-dimension world folder — never under lib/ or tex/.
            if top not in ("", "program", "world0", "world1", "world-1"):
                continue
            stem, ext = os.path.splitext(base)
            if ext in _STAGE_EXTS:
                names.add(stem)
        return sorted(names)

    def worlds(self) -> list[str]:
        """Dimension folders that actually contain programs (world0, ...)."""
        found: set[str] = set()
        for key in self.files:
            top = key.split("/", 1)[0] if "/" in key else ""
            if top in ("world0", "world1", "world-1"):
                found.add(top)
        return sorted(found)

    # -- metadata --------------------------------------------------------
    def read_text(self, relpath: str) -> str | None:
        """Return the text of a pack file (relative to the shaders root)."""
        return self.files.get(relpath)

    def properties(self):
        """Parse ``shaders.properties`` (empty :class:`Properties` if absent)."""
        from .properties import Properties, parse_properties

        text = self.files.get("shaders.properties")
        return parse_properties(text) if text is not None else Properties()

    def block_mapping(self):
        """Parse ``block.properties`` into a :class:`BlockMapping`."""
        from .properties import BlockMapping, parse_block_mapping

        text = self.files.get("block.properties")
        return parse_block_mapping(text, "block") if text is not None else BlockMapping("block")

    def item_mapping(self):
        from .properties import BlockMapping, parse_block_mapping

        text = self.files.get("item.properties")
        return parse_block_mapping(text, "item") if text is not None else BlockMapping("item")

    def entity_mapping(self):
        from .properties import BlockMapping, parse_block_mapping

        text = self.files.get("entity.properties")
        return parse_block_mapping(text, "entity") if text is not None else BlockMapping("entity")

    def option_sources(self) -> dict[str, str]:
        """Every ``lib/`` ``.glsl`` file, where option ``#define``s live."""
        return {
            key: text
            for key, text in self.files.items()
            if key.startswith("lib/") and key.endswith(".glsl")
        }

    def _find(self, name: str, ext: str, world: str) -> str | None:
        for candidate in ([f"{world}/{name}{ext}"] if world else []) + [
            f"program/{name}{ext}", f"{name}{ext}"
        ]:
            if candidate in self.files:
                return self.files[candidate]
        return None

    def read_program(self, name: str, world: str = "world0") -> ProgramSource:
        """Collect the vertex/fragment/combined sources for a program."""
        vsh = self._find(name, ".vsh", world) or self._find(name, ".vert", world)
        fsh = self._find(name, ".fsh", world) or self._find(name, ".frag", world)
        combined = self._find(name, ".glsl", world)
        if vsh is None and fsh is None and combined is None:
            raise KeyError(f"program {name!r} not found in pack {self.name!r}")
        return ProgramSource(name, world, vsh, fsh, combined)
