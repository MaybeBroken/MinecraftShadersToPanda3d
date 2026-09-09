"""Shaderpack loading and metadata parsing."""

from .loader import ShaderPack, ProgramSource
from .properties import (
    Properties,
    BlockMapping,
    parse_properties,
    parse_block_mapping,
)

__all__ = [
    "ShaderPack",
    "ProgramSource",
    "Properties",
    "BlockMapping",
    "parse_properties",
    "parse_block_mapping",
]
