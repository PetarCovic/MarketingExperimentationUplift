import pytest
pytest.importorskip("duckdb")
from marketing_uplift.experiment import difference_in_means


def test_difference_in_means_direction_and_interval():
    result = difference_in_means(10000, 0.12, 0.1056, 10000, 0.10, 0.09)
    assert result["absolute_effect"] > 0
    assert result["ci_95_lower"] < result["absolute_effect"] < result["ci_95_upper"]
    assert round(result["relative_lift_percent"], 6) == 20.0

