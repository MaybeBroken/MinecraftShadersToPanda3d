import pytest

from mcshader import default_registry, Effect
from mcshader.registry import ShaderRegistry


def test_default_registry_has_builtins():
    r = default_registry()
    for sid in ("glow", "reflection", "waving", "movement"):
        assert sid in r
        assert isinstance(r.get(sid), Effect)


def test_register_custom_and_get():
    r = ShaderRegistry(include_builtins=False)
    e = Effect(id="mine", name="Mine", description="", vertex="v", fragment="f")
    r.register(e)
    assert r.get("mine") is e
    assert r.ids() == ["mine"]


def test_duplicate_id_requires_replace():
    r = default_registry()
    dup = Effect(id="glow", name="x", description="", vertex="", fragment="")
    with pytest.raises(KeyError):
        r.register(dup)
    r.register(dup, replace=True)
    assert r.get("glow") is dup


def test_unknown_id_lists_known():
    r = default_registry()
    with pytest.raises(KeyError) as exc:
        r.get("does_not_exist")
    assert "glow" in str(exc.value)


def test_empty_id_rejected():
    r = ShaderRegistry(include_builtins=False)
    with pytest.raises(ValueError):
        r.register(Effect(id="", name="", description="", vertex="", fragment=""))
