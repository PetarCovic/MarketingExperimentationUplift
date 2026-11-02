"""Assignment diagnostics and intent-to-treat difference-in-means estimates."""

from __future__ import annotations

from typing import Any

import duckdb
import numpy as np
from scipy.stats import chi2, norm

FEATURES = tuple(f"f{i}" for i in range(12))


def difference_in_means(
    treated_n: int, treated_mean: float, treated_variance: float,
    control_n: int, control_mean: float, control_variance: float,
) -> dict[str, float | None]:
    if treated_n < 2 or control_n < 2 or min(treated_variance, control_variance) < 0:
        raise ValueError("Both arms need >=2 observations and nonnegative sample variances")
    effect = treated_mean - control_mean
    standard_error = float(np.sqrt(treated_variance / treated_n + control_variance / control_n))
    z = effect / standard_error if standard_error > 0 else None
    return {
        "control_rate": float(control_mean),
        "treatment_rate": float(treated_mean),
        "absolute_effect": float(effect),
        "relative_lift_percent": float(100 * effect / control_mean) if control_mean else None,
        "standard_error": standard_error,
        "ci_95_lower": float(effect - norm.ppf(0.975) * standard_error),
        "ci_95_upper": float(effect + norm.ppf(0.975) * standard_error),
        "p_value": float(2 * norm.sf(abs(z))) if z is not None else (0.0 if effect else 1.0),
    }


def analyze_experiment(
    con: duckdb.DuckDBPyConnection, relation: str, expected_treatment_rate: float,
    outcomes: tuple[str, ...] = ("visit", "conversion"),
) -> dict[str, Any]:
    if not 0 < expected_treatment_rate < 1 or not np.isfinite(expected_treatment_rate):
        raise ValueError("expected_treatment_rate must be between 0 and 1")
    if any(outcome not in ("visit", "conversion") for outcome in outcomes):
        raise ValueError("Unsupported outcome")
    columns = [row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()]
    required = {"treatment", *outcomes, *FEATURES}
    if missing := required - set(columns):
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    validity_columns = ["treatment", *outcomes]
    for col in validity_columns:
        bad = con.execute(
            f"SELECT COUNT(*) FROM {relation} WHERE {col} IS NULL OR {col} NOT IN (0, 1)"
        ).fetchone()[0]
        if bad:
            raise ValueError(f"Column {col} contains {bad:,} non-binary or null entries")
    row_count, treated = con.execute(
        f"SELECT COUNT(*), SUM(CAST(treatment AS BIGINT)) FROM {relation}"
    ).fetchone()
    if row_count < 4 or treated < 2 or row_count - treated < 2:
        raise ValueError("Both treatment arms must have at least two rows")
    control = row_count - treated
    expected_treated = row_count * expected_treatment_rate
    expected_control = row_count - expected_treated
    srm_stat = (treated - expected_treated) ** 2 / expected_treated + (control - expected_control) ** 2 / expected_control
    result: dict[str, Any] = {
        "rows": int(row_count), "treated_rows": int(treated), "control_rows": int(control),
        "observed_treatment_rate": float(treated / row_count),
        "expected_treatment_rate": expected_treatment_rate,
        "sample_ratio_mismatch_chi_square": float(srm_stat),
        "sample_ratio_mismatch_p_value": float(chi2.sf(srm_stat, df=1)),
        "srm_caveat": "Pooled experiments; 85% allocation is a configured assumption, not independently verified per trial",
        "outcomes": {},
    }
    for outcome in outcomes:
        stats = con.execute(
            f"SELECT treatment, COUNT(*) AS n, AVG(CAST({outcome} AS DOUBLE)) AS mean, "
            f"VAR_SAMP(CAST({outcome} AS DOUBLE)) AS variance "
            f"FROM {relation} GROUP BY treatment ORDER BY treatment"
        ).fetchall()
        by_group = {int(t): (int(n), float(mean), float(var or 0)) for t, n, mean, var in stats}
        n0, mean0, var0 = by_group[0]
        n1, mean1, var1 = by_group[1]
        effect = difference_in_means(n1, mean1, var1, n0, mean0, var0)
        pooled_rate = (n1 * mean1 + n0 * mean0) / (n1 + n0)
        mde = (norm.ppf(0.975) + norm.ppf(0.8)) * np.sqrt(
            pooled_rate * (1 - pooled_rate) * (1 / n1 + 1 / n0)
        )
        effect["minimum_detectable_absolute_effect_80pct_power_approx"] = float(mde)
        result["outcomes"][outcome] = effect

    smds: dict[str, float | None] = {}
    for column in FEATURES:
        stats = con.execute(
            f"SELECT treatment, AVG({column}), VAR_SAMP({column}) FROM {relation} "
            f"GROUP BY treatment ORDER BY treatment"
        ).fetchall()
        _, mean0, var0 = stats[0]
        _, mean1, var1 = stats[1]
        if any(value is None for value in (mean0, mean1, var0, var1)):
            smds[column] = None
            continue
        pooled_sd = float(np.sqrt((float(var0) + float(var1)) / 2))
        smds[column] = (float(mean1) - float(mean0)) / pooled_sd if pooled_sd > 0 else (0.0 if mean1 == mean0 else None)
    result["covariate_balance"] = {
        "standardized_mean_differences": smds,
        "maximum_absolute_smd": max((abs(v) for v in smds.values() if v is not None), default=None),
    }
    return result
