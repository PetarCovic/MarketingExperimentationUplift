"""Binary-outcome meta-learners and honest, held-out uplift evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from scipy.integrate import trapezoid

FEATURES = tuple(f"f{i}" for i in range(12))


def _binary(values: np.ndarray, name: str) -> np.ndarray:
    if not np.isin(values, [0, 1]).all():
        raise ValueError(f"{name} must contain only 0 and 1")
    return values.astype(int)


def _ensure_sufficient_groups(t: np.ndarray, y: np.ndarray, context: str) -> None:
    if any(np.sum((t == a) & (y == b)) == 0 for a in (0, 1) for b in (0, 1)):
        raise ValueError(f"{context} needs positive and negative outcomes in both treatment arms")


def evaluate_uplift(
    y: np.ndarray, treatment: np.ndarray, effect: np.ndarray, budget_fraction: float
) -> dict[str, float | int | bool | None]:
    """Score a pre-trained ranking on data not used to fit its models.

    Separate-relative AUUC matches Criteo's published formula. Qini/AUUC
    diagnostics employ marginal-propensity IPW. Selected-subgroup treatment
    effect uses raw randomized-arm means, with a normal-approximation CI.
    """
    y = np.asarray(y)
    treatment = np.asarray(treatment)
    effect = np.asarray(effect, dtype=float)
    if not 0 < budget_fraction <= 1 or not np.isfinite(budget_fraction):
        raise ValueError("budget_fraction must lie in (0, 1]")
    if len(y) != len(treatment) or len(y) != len(effect) or not len(y):
        raise ValueError("y, treatment, and effect must have equal, nonzero lengths")
    if not np.isfinite(effect).all():
        raise ValueError("Predicted effects must be finite")
    y = _binary(y, "outcome")
    treatment = _binary(treatment, "treatment")
    _ensure_sufficient_groups(treatment, y, "Evaluation")

    propensity = float(np.mean(treatment))
    treated_y = y[treatment == 1]
    control_y = y[treatment == 0]
    treated_auc = float(roc_auc_score(treated_y, effect[treatment == 1]))
    control_auc = float(roc_auc_score(control_y, effect[treatment == 0]))
    p1, p0 = float(treated_y.mean()), float(control_y.mean())
    separate_relative_auuc = p1 * (1 - p1) * treated_auc - p0 * (1 - p0) * control_auc + (p1**2 - p0**2) / 2

    transformed = treatment * y / propensity - (1 - treatment) * y / (1 - propensity)
    order = np.argsort(-effect, kind="stable")
    ranked = transformed[order]
    cumulative = np.cumsum(ranked)
    fractions = np.arange(1, len(y) + 1) / len(y)
    random_curve = cumulative[-1] * fractions
    # Keep the origin so the trapezoid is defined on the full interval [0, 1].
    qini_per_row = float(trapezoid(np.r_[0, cumulative - random_curve], np.r_[0, fractions]) / len(y))
    auuc_per_row = float(trapezoid(np.r_[0, cumulative / len(y)], np.r_[0, fractions]))

    selected_n = max(1, int(np.ceil(len(y) * budget_fraction)))
    selected = order[:selected_n]
    selected_t = treatment[selected]
    selected_y = y[selected]
    n1 = int(np.sum(selected_t == 1))
    n0 = int(np.sum(selected_t == 0))
    if n1 > 1 and n0 > 1:
        y1, y0 = selected_y[selected_t == 1], selected_y[selected_t == 0]
        uplift = float(y1.mean() - y0.mean())
        standard_error = float(np.sqrt(y1.var(ddof=1) / n1 + y0.var(ddof=1) / n0))
        interval_lower = uplift - 1.96 * standard_error
        interval_upper = uplift + 1.96 * standard_error
        policy_per_100k = selected_n / len(y) * 100_000 * uplift
    else:
        uplift = interval_lower = interval_upper = policy_per_100k = None

    # IPW values are deliberately identified as diagnostics, not causal ROI.
    diagnostic_selected = float(np.sum(transformed[selected]))
    diagnostic_random = float(np.mean(transformed) * selected_n)
    diagnostic_gain = (
        100 * (diagnostic_selected - diagnostic_random) / abs(diagnostic_random)
        if diagnostic_random != 0 else None
    )
    return {
        "qini_area_per_row": qini_per_row,
        "auuc_per_row": auuc_per_row,
        "separate_relative_auuc": float(separate_relative_auuc),
        "treated_outcome_auc": treated_auc,
        "control_outcome_auc": control_auc,
        "budget_fraction": float(budget_fraction),
        "selected_rows": selected_n,
        "selected_treated_rows": n1,
        "selected_control_rows": n0,
        "selected_treatment_rate": float(np.mean(selected_t)),
        "overall_treatment_rate": propensity,
        "selected_treatment_rate_difference_percentage_points": float(100 * (np.mean(selected_t) - propensity)),
        "uplift_at_budget": uplift,
        "uplift_at_budget_ci_95_lower": interval_lower,
        "uplift_at_budget_ci_95_upper": interval_upper,
        "estimated_incremental_outcomes_per_100k_eligible": policy_per_100k,
        "estimated_incremental_outcomes_selected_ipw_diagnostic": diagnostic_selected,
        "estimated_incremental_outcomes_random_ipw_diagnostic": diagnostic_random,
        "incremental_outcomes_vs_random_percent_ipw_diagnostic": diagnostic_gain,
        "constant_propensity_ipw_is_diagnostic_only": True,
    }


def _stratify(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    return t * 2 + y


def fit_and_evaluate(
    frame: pd.DataFrame,
    outcome: str,
    budget_fraction: float = 0.10,
    random_seed: int = 42,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """60/20/20 stratified train/validation/test, validation-only selection.

    All learners train on the training split only. A learner is selected on
    validation AUUC; the independent test is evaluated once after selection.
    """
    if outcome not in ("visit", "conversion"):
        raise ValueError("outcome must be 'visit' or 'conversion'")
    if not 0 < budget_fraction <= 1 or not np.isfinite(budget_fraction):
        raise ValueError("budget_fraction must lie in (0, 1]")
    required = list(FEATURES) + ["treatment", outcome]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    clean = frame.loc[:, required].replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 100:
        raise ValueError("At least 100 complete rows are required")
    x = clean[list(FEATURES)].to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Feature values must be finite")
    t = _binary(clean["treatment"].to_numpy(), "treatment")
    y = _binary(clean[outcome].to_numpy(), outcome)
    _ensure_sufficient_groups(t, y, "Input")
    ids = np.arange(len(clean))
    labels = _stratify(t, y)
    if min(np.bincount(labels, minlength=4)) < 5:
        raise ValueError("At least five observations per treatment/outcome stratum are required")
    try:
        train_idx, holdout_idx = train_test_split(ids, test_size=0.40, random_state=random_seed, stratify=labels)
        val_idx, test_idx = train_test_split(
            holdout_idx, test_size=0.50, random_state=random_seed + 1, stratify=labels[holdout_idx]
        )
    except ValueError as exc:
        raise ValueError(f"Unable to make stratified train/validation/test splits: {exc}") from exc
    x_train, x_val, x_test = x[train_idx], x[val_idx], x[test_idx]
    t_train, t_val, t_test = t[train_idx], t[val_idx], t[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    _ensure_sufficient_groups(t_val, y_val, "Validation split")
    _ensure_sufficient_groups(t_test, y_test, "Test split")

    params = {"max_iter": 120, "learning_rate": 0.06, "max_leaf_nodes": 15, "l2_regularization": 1.0, "random_state": random_seed}
    s_model = HistGradientBoostingClassifier(**params).fit(np.column_stack([x_train, t_train]), y_train)

    def predict_s(a: np.ndarray) -> np.ndarray:
        return (
            s_model.predict_proba(np.column_stack([a, np.ones(len(a))]))[:, 1]
            - s_model.predict_proba(np.column_stack([a, np.zeros(len(a))]))[:, 1]
        )

    control_model = HistGradientBoostingClassifier(**params).fit(x_train[t_train == 0], y_train[t_train == 0])
    treatment_model = HistGradientBoostingClassifier(**params).fit(x_train[t_train == 1], y_train[t_train == 1])

    def predict_t(a: np.ndarray) -> np.ndarray:
        return treatment_model.predict_proba(a)[:, 1] - control_model.predict_proba(a)[:, 1]

    d1 = y_train[t_train == 1] - control_model.predict_proba(x_train[t_train == 1])[:, 1]
    d0 = treatment_model.predict_proba(x_train[t_train == 0])[:, 1] - y_train[t_train == 0]
    tau1 = HistGradientBoostingRegressor(**params).fit(x_train[t_train == 1], d1)
    tau0 = HistGradientBoostingRegressor(**params).fit(x_train[t_train == 0], d0)
    propensity = float(t_train.mean())

    def predict_x(a: np.ndarray) -> np.ndarray:
        return propensity * tau0.predict(a) + (1 - propensity) * tau1.predict(a)

    predictors = {"s_learner": predict_s, "t_learner": predict_t, "x_learner": predict_x}
    validation_effects = {name: fn(x_val) for name, fn in predictors.items()}
    validation_metrics = {name: evaluate_uplift(y_val, t_val, pred, budget_fraction) for name, pred in validation_effects.items()}
    selected_name = max(validation_metrics, key=lambda name: validation_metrics[name]["separate_relative_auuc"])
    test_effects = {name: fn(x_test) for name, fn in predictors.items()}
    test_metrics = {name: evaluate_uplift(y_test, t_test, pred, budget_fraction) for name, pred in test_effects.items()}
    prediction_frame = pd.DataFrame({
        "sample_row_index": test_idx, "treatment": t_test, "outcome": y_test,
        **{f"{name}_uplift": value for name, value in test_effects.items()},
    })
    result: dict[str, Any] = {
        "outcome": outcome, "features": list(FEATURES), "eligible_rows": int(len(clean)),
        "excluded_rows": int(len(frame) - len(clean)), "random_seed": random_seed,
        "split": "60% training / 20% validation / 20% independent test (stratified by assignment and outcome)",
        "train_rows": int(len(train_idx)), "validation_rows": int(len(val_idx)), "test_rows": int(len(test_idx)),
        "treatment_propensity_train": propensity,
        "validation_models": validation_metrics,
        "models": test_metrics,
        "selected_model_by_validation_separate_relative_auuc": selected_name,
        "selected_model_test_metrics": test_metrics[selected_name],
        "model_selection_rule": "Maximum separate-relative AUUC on validation; no test-set selection",
        "inference_note": "No formal uncertainty correction for model comparison or repeated experiment-level analysis",
    }
    return result, prediction_frame
