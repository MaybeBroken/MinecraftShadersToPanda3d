from mcshader.effects import builtin_effects, BUILTIN_IDS


def test_builtin_ids():
    assert set(BUILTIN_IDS) == {"glow", "reflection", "waving", "movement"}


def test_effects_have_complete_sources():
    for eid, e in builtin_effects().items():
        assert e.vertex.strip().startswith("#version"), eid
        assert e.fragment.strip().startswith("#version"), eid
        assert "void main" in e.vertex and "void main" in e.fragment


def test_builtin_effects_are_copies():
    a = builtin_effects()["glow"]
    a.params.clear()
    b = builtin_effects()["glow"]
    assert b.params, "mutating one copy must not affect the shared definition"


def test_defaults_skip_resourceless_params():
    refl = builtin_effects()["reflection"]
    # u_env_map has default None and must be omitted from defaults()
    assert "u_env_map" not in refl.defaults()
    assert "u_reflectivity" in refl.defaults()


def test_auto_inputs_declared_in_source():
    for e in builtin_effects().values():
        for name in e.auto_inputs:
            assert name in e.vertex or name in e.fragment, (e.id, name)
