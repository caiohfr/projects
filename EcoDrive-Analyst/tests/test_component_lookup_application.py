import pytest

from src.vde_core.component_lookup_application import (
    REQUIRES_USER_BASELINE_INPUT,
    apply_absolute_component_lookup,
)


def test_lookup_without_recalculation_preserves_total_exactly() -> None:
    result = apply_absolute_component_lookup(
        old_total_abc={"A": 120.0, "B": 0.02, "C": 0.01},
        new_component_abc={"A": 8.0, "B": 0.004, "C": 0.0008},
        recalculate_total_abc=False,
    )

    assert result.status == "OK"
    assert result.new_total_abc == {"A": 120.0, "B": 0.02, "C": 0.01}
    assert result.baseline_component_abc is None
    assert result.delta_component_abc is None


def test_lookup_recalculation_uses_new_minus_baseline() -> None:
    result = apply_absolute_component_lookup(
        old_total_abc={"A": 120.0, "B": 0.02, "C": 0.01},
        new_component_abc={"A": 8.0, "B": 0.004, "C": 0.0008},
        baseline_component_abc={"A": 10.0, "B": 0.005, "C": 0.001},
        baseline_source="user_input",
        recalculate_total_abc=True,
    )

    assert result.delta_component_abc == pytest.approx({"A": -2.0, "B": -0.001, "C": -0.0002})
    assert result.new_total_abc == pytest.approx({"A": 118.0, "B": 0.019, "C": 0.0098})
    assert result.baseline_source == "user_input"


def test_lookup_recalculation_requires_baseline_and_never_uses_absolute_as_delta() -> None:
    result = apply_absolute_component_lookup(
        old_total_abc={"A": 120.0, "B": 0.02, "C": 0.01},
        new_component_abc={"A": 8.0, "B": 0.004, "C": 0.0008},
        recalculate_total_abc=True,
    )

    assert result.status == REQUIRES_USER_BASELINE_INPUT
    assert result.new_total_abc == {"A": 120.0, "B": 0.02, "C": 0.01}
    assert result.delta_component_abc is None
