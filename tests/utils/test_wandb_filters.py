import sys
import types
import warnings

import utils.wandb_filters as wf


def test_silence_no_pydantic_installed():
    # Should be a harmless no-op when Pydantic is absent.
    wf.silence_pydantic_field_warnings()


def test_silence_with_stub(monkeypatch):
    class DummyWarning(Warning):
        pass

    warnings_mod = types.ModuleType("pydantic.warnings")
    warnings_mod.UnsupportedFieldAttributeWarning = DummyWarning

    pydantic_mod = types.ModuleType("pydantic")
    pydantic_mod.warnings = warnings_mod

    monkeypatch.setitem(sys.modules, "pydantic", pydantic_mod)
    monkeypatch.setitem(sys.modules, "pydantic.warnings", warnings_mod)

    calls = []

    def fake_filter(action, category, **kwargs):
        calls.append((action, category, kwargs))

    monkeypatch.setattr(warnings, "filterwarnings", fake_filter)

    wf.silence_pydantic_field_warnings()

    assert calls, "filterwarnings should be invoked when the warning class is available"
    assert calls[0][1] is DummyWarning
