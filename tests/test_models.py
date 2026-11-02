import numpy as np
import pandas as pd
import pytest

from marketing_uplift.models import evaluate_uplift, fit_and_evaluate


def test_ranking_metric_matches_official_formula():
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    t = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    score = np.array([0.9, 0.7, 0.5, 0.3, 0.8, 0.6, 0.4, 0.2])
    result = evaluate_uplift(y, t, score, 0.5)
    assert result['separate_relative_auuc'] == pytest.approx(0.0)
    assert result['selected_rows'] == 4
    assert result['selected_treated_rows'] == 2
    assert result['selected_control_rows'] == 2


def test_invalid_budget_and_nonbinary_values_are_rejected():
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    t = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    score = np.linspace(0, 1, len(y))
    with pytest.raises(ValueError, match='budget_fraction'):
        evaluate_uplift(y, t, score, 0)
    with pytest.raises(ValueError, match='treatment'):
        evaluate_uplift(y, np.where(t == 1, 2, 0), score, 0.5)
    with pytest.raises(ValueError, match='finite'):
        evaluate_uplift(y, t, np.full(len(y), np.nan), 0.5)


def test_stratified_holdout_validation_and_selection():
    rng = np.random.default_rng(31)
    rows = 1200
    features = rng.normal(size=(rows, 12))
    t = rng.binomial(1, 0.8, rows)
    p = 1 / (1 + np.exp(-(-1.5 + 0.4*features[:, 0] + 0.3*t*features[:, 1])))
    y = rng.binomial(1, p)
    df = pd.DataFrame(features, columns=[f'f{i}' for i in range(12)])
    df['treatment'] = t
    df['visit'] = y
    m, predictions = fit_and_evaluate(df, 'visit')
    assert m['train_rows'] + m['validation_rows'] + m['test_rows'] == rows
    assert m['test_rows'] == len(predictions)
    assert predictions.sample_row_index.is_unique
    assert m['selected_model_by_validation_separate_relative_auuc'] in ('s_learner', 't_learner', 'x_learner')
    assert m['model_selection_rule'].startswith('Maximum')
    for name in ['s_learner', 't_learner', 'x_learner']:
        assert np.isfinite(m['models'][name]['separate_relative_auuc'])
    m2, _ = fit_and_evaluate(df, 'visit')
    assert m2['selected_model_by_validation_separate_relative_auuc'] == m['selected_model_by_validation_separate_relative_auuc']


def test_missing_features_caught():
    df = pd.DataFrame({'f0':[0]*200, 'treatment':[0,1]*100,'visit':[0,1]*100})
    with pytest.raises(ValueError, match='Missing required'):
        fit_and_evaluate(df, 'visit')
