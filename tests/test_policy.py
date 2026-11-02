import numpy as np

from marketing_uplift.models import evaluate_uplift


def test_policy_evaluation_returns_expected_fields():
    y = np.array([1, 0, 1, 0, 1, 0, 0, 0])
    treatment = np.array([1, 0, 1, 0, 0, 1, 0, 1])
    effect = np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    result = evaluate_uplift(y, treatment, effect, 0.5)
    assert result["selected_rows"] == 4
    assert "qini_area_per_row" in result
    assert "separate_relative_auuc" in result
    assert "incremental_outcomes_vs_random_percent_ipw_diagnostic" in result
