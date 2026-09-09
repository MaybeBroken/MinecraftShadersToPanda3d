"""GLSL handling: preprocessing, dialect translation, and the MC uniform catalog."""

from .preprocess import resolve_includes, select_stage
from .dialect import translate_stage, TranslationResult
from .fullscreen import FULLSCREEN_VERTEX
from . import uniforms

__all__ = [
    "resolve_includes",
    "select_stage",
    "translate_stage",
    "TranslationResult",
    "FULLSCREEN_VERTEX",
    "uniforms",
]
