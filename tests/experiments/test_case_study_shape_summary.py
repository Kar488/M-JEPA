from __future__ import annotations

import experiments.case_study as cs


def test_shape_tuple_and_fmt_shape_without_numpy():
    class WithShape:
        shape = ("2", 3)

    class WithoutShape:
        pass

    assert cs._shape_tuple(WithShape()) == (2, 3)
    assert cs._shape_tuple(WithoutShape()) is None
    assert cs._fmt_shape((1, 2)) == "(1, 2)"
    assert cs._fmt_shape(None) == "<none>"


def test_key_summaries_include_shapes():
    module_shapes = {"layer.weight": (2, 3), "layer.bias": None}
    incoming_shapes = {"layer.weight": (2, 4), "layer.bias": None, "extra": (1, 2)}

    summary = cs._summarise_keys(
        ["layer.weight", "layer.bias", "extra"],
        module_shapes,
        incoming_shapes,
        limit=2,
    )
    assert "layer.weight" in summary[0]
    assert "expected=(2, 3)" in summary[0]
    assert summary[-1].startswith("...(+")

    unexpected = cs._summarise_unexpected(
        ["extra", "missing"],
        incoming_shapes,
        limit=1,
    )
    assert "shape=(1, 2)" in unexpected[0]
    assert unexpected[-1].startswith("...(+")
