# Benchmarks — Marketing Experimentation & Uplift Modeling

**Benchmark date:** November 2, 2026 (UTC timestamps: visit `2026-11-02T23:35:51.274380+00:00`; conversion `2026-11-02T23:36:00.189872+00:00`)  
**Dataset:** Criteo Uplift Prediction Dataset v2.1 (official-data run)  
**Evaluation status:** Both visit and conversion runs completed; results are taken from their saved JSON artifacts. This is a single-seed benchmark, not an independently replicated experiment or production deployment.

## Executive summary

The pipeline analyzed **13,979,592 records** for randomized-assignment (intent-to-treat, ITT) effects and used a reproducible **1,000,000-record sample** for uplift modeling. S-, T-, and X-learners were trained on separate treatment/outcome-aware stratified splits. The learner for each outcome was chosen **only on validation separate-relative AUUC** and then evaluated on an independent test split.

| Result | Visit | Conversion |
|---|---:|---:|
| Full-data absolute ITT effect | +1.0342 pp | +0.1152 pp |
| Full-data relative lift vs. control | 27.07% | 59.45% |
| Validation-selected learner | **S-learner** | **X-learner** |
| Selected learner's validation separate-relative AUUC | 0.010074 | 0.001059 |
| Selected learner's independent-test separate-relative AUUC | **0.010297** | **0.000998** |
| Selected learner's test top-10% subgroup effect | +6.4223 pp | +0.8569 pp |

**Important:** Conversion's X-learner was selected correctly on validation results, although the S-learner achieved a *slightly higher* test separate-relative AUUC (`0.000997993` versus `0.000997627`). This retrospective test comparison is **not** a reason to replace the previously selected winner. No formal test of the difference between the models was performed.

## Dataset, preparation, and environment

| Property | Recorded value |
|---|---|
| Dataset / provenance flag | `Criteo Uplift Prediction Dataset v2.1` / `official_criteo_v2.1` |
| Full experimental records | 13,979,592 |
| Treatment-assigned records | 11,882,655 |
| Control-assigned records | 2,096,937 |
| Observed treatment assignment rate | 85.0000% |
| Configured expected treatment assignment rate | 85.0000% |
| Modeling records | 1,000,000 |
| Missing values in recorded columns | **0** for each of the 16 columns |
| Baseline predictors | `f0`–`f11` (12 anonymized features) |
| Assignment variable | `treatment` (binary) |
| Outcomes | `visit`, `conversion` (binary) |
| Other recorded variable | `exposure` (post-treatment; excluded from predictors) |
| Sampling / splitting random seed | `42` |
| Python | `3.11.9` |
| DuckDB | `1.5.6` |
| Operating system | `Windows-10-10.0.26200-SP0` |
| Manifest-recorded source SHA-256 | `2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc` |

The full dataset was used for the ITT experiment analysis. The modeled sample was drawn with a fixed-seed DuckDB reservoir sample. **Zero records were excluded for missing/non-finite modeling inputs** in either outcome run. The source SHA-256 shown above is the hash *recorded by the supplied manifest*, not a new independent calculation against the original compressed CSV for this report.

### Model-development split

| Stage | Rows | Fraction | Purpose |
|---|---:|---:|---|
| Training | 600,000 | 60% | Fit the S-, T-, and X-learners |
| Validation | 200,000 | 20% | Select the learner with maximum separate-relative AUUC |
| Independent test | 200,000 | 20% | Report held-out quality of the validation-selected learner |

The split is stratified by treatment assignment and the relevant outcome. A separate outcome-specific split is constructed for visit versus conversion, so the test partitions are **not necessarily identical** across the two analyses. The test-set scores for every learner are provided for transparency; they were not used for model selection.

### Model configuration

All three candidate learners use scikit-learn histogram gradient boosting. The implementation uses `max_iter=120`, `learning_rate=0.06`, `max_leaf_nodes=15`, `l2_regularization=1.0`, and `random_state=42` for the underlying estimators. No systematic hyperparameter-search result is contained in the benchmark artifacts.

## Full-dataset randomized-assignment results

Treatment is interpreted as **assignment**, not actual ad exposure. The primary quantity is the difference in average observed outcomes between the assigned arms (ITT).

| Outcome | Control rate | Treatment rate | Absolute effect | Relative lift | Approximate 95% CI, absolute effect | Reported p-value |
|---|---:|---:|---:|---:|---:|---:|
| Visits | 3.8201% | 4.8543% | +1.0342 pp | +27.07% | [+1.0056 pp, +1.0629 pp] | Underflowed to `0.0` |
| Conversions | 0.1938% | 0.3089% | +0.1152 pp | +59.45% | [+0.1085 pp, +0.1219 pp] | 3.195e-246 |

The reported `0.0` visit p-value is a numerical underflow from an extremely small computed value; **it does not mean the probability is literally zero**. Both full-sample results have narrow reported normal-approximation confidence intervals, but the released data's pooling and sampling limitations still apply.

### Assignment and baseline diagnostics

- **Sample-ratio mismatch (SRM):** χ² = `1.81777585e-06` and p = `0.998924` against the *configured* 85% expected assignment proportion.
- **Maximum absolute standardized mean difference:** `0.048836` across the 12 anonymized predictors (the largest absolute imbalance was at `f3`).
- **Interpretation:** No large pooled allocation discrepancy is detected against the assumed 85% target, and the largest reported SMD is small. These diagnostics do **not** independently verify randomization or assignment probabilities in each original trial; trial identifiers are unavailable.
- **Post-hoc approximate 80%-power MDE:** visits `+0.0444 pp`; conversions `+0.0113 pp`. These are descriptive approximations, **not** prospective power guarantees.

## Visit uplift: learner comparison

**Selection criterion:** highest validation separate-relative AUUC. **Selected:** **S-learner**.

| Learner | Validation separate-relative AUUC | Independent-test separate-relative AUUC | Test Qini area/row | Test AUUC/row | Test uplift at 10% |
|---|---:|---:|---:|---:|---:|
| S-learner **(selected)** | 0.010074 | 0.010297 | 0.004974 | 0.010623 | +6.4223 pp |
| T-learner | 0.010059 | 0.009916 | 0.004519 | 0.010168 | +6.2301 pp |
| X-learner | 0.009637 | 0.010222 | 0.004887 | 0.010537 | +6.3873 pp |

The validation-selected **S-learner** ranked the top **20,000 of 200,000** independent-test rows (a **10%** targeting budget). The selected group comprised 17,296 treated and 2,704 control records. Its observed treated-minus-control outcome-rate difference was **+6.4223 pp** (reported approximate 95% CI: **[+4.6509 pp, +8.1936 pp]**). The pipeline's extrapolation is **642.2 incremental visits per 100,000 eligible records** if the top 10% were treated in a similarly distributed benchmark population. This is **not** a measured business deployment result or a revenue estimate.

At the chosen 10% budget, the visit result represents an estimated subgroup **absolute outcome-rate difference**, not a 6.42% increase in revenue or a 6.42% population-wide conversion improvement.

## Conversion uplift: learner comparison

**Selection criterion:** highest validation separate-relative AUUC. **Selected:** **X-learner**.

| Learner | Validation separate-relative AUUC | Independent-test separate-relative AUUC | Test Qini area/row | Test AUUC/row | Test uplift at 10% |
|---|---:|---:|---:|---:|---:|
| S-learner | 0.000930 | 0.000997993 | 0.000451 | 0.001006 | +0.9005 pp |
| T-learner | 0.000725 | 0.000920631 | 0.000371 | 0.000926 | +0.7204 pp |
| X-learner **(selected)** | 0.001059 | 0.000997627 | 0.000438 | 0.000993 | +0.8569 pp |

The validation-selected **X-learner** ranked the top **20,000 of 200,000** independent-test rows (a **10%** targeting budget). The selected group comprised 17,232 treated and 2,768 control records. Its observed treated-minus-control outcome-rate difference was **+0.8569 pp** (reported approximate 95% CI: **[+0.3763 pp, +1.3374 pp]**). The pipeline's extrapolation is **85.7 incremental conversions per 100,000 eligible records** if the top 10% were treated in a similarly distributed benchmark population. This is **not** a measured business deployment result or a revenue estimate.

Conversion is a relatively rare outcome in this dataset. The conversion winner is therefore discussed as an **uplift-ranking benchmark** rather than evidence that individual treatment effects can be precisely known or that an operational targeting policy has been established.

## Mathematical definitions and metric interpretation

- **Intent-to-treat estimate:** $\widehat{\Delta}_{ITT} = \bar Y_{T=1}-\bar Y_{T=0}$. The corresponding **relative lift** is $100\widehat{\Delta}_{ITT}/\bar Y_{T=0}$ (percent). A percentage-point difference must not be confused with relative percent growth.
- **Reported standard error:** $\widehat{SE}(\widehat{\Delta})=\sqrt{s_1^2/n_1+s_0^2/n_0}$, with normal-approximation confidence interval $\widehat{\Delta}\pm1.96\widehat{SE}$.
- **Conditional average treatment effect (CATE):** $\tau(x)=E[Y(1)-Y(0)\mid X=x]$. A unit's two potential outcomes are never both observed, so this effect is *estimated*, not directly labeled per example.
- **S-learner:** one model with treatment as an additional input; uplift is the difference between its predictions under $T=1$ and $T=0$.
- **T-learner:** separate outcome models for treated and control arms; uplift is the difference between predicted outcome probabilities.
- **X-learner:** model potential outcomes, construct pseudo-effects for the opposite arm, then model and combine estimated effects. Here the implementation uses a **constant training-sample treatment proportion** as its combination weight; it is not a cross-fitted, trial-specific propensity approach.
- **Separate-relative AUUC:** the **model-selection metric**. The implementation uses the published two-arm ranking expression $p_1(1-p_1)AUC_1-p_0(1-p_0)AUC_0+(p_1^2-p_0^2)/2$, where each $AUC_g$ compares binary outcomes with uplift ranks within arm $g$. **These values are not classification accuracy or percentage lifts.**
- **Qini area/row and AUUC/row:** normalized areas calculated from cumulative, uplift-score-ranked inverse-propensity-weighted outcome gains. They depend on the scale, outcome, and propensity approximation. Compare learners *within the same benchmark*, rather than interpreting the raw magnitudes as business ROI.
- **Uplift at 10% budget:** estimated observed treatment-minus-control outcome rate among the **20,000 highest-scored test records**. CIs use the implementation's ordinary normal-approximation two-sample standard error. The implied outcomes per 100k eligible records multiply that subgroup effect by the targeting fraction (10%).
- **Classifier AUC:** treated/control within-arm outcome discrimination is also logged. High outcome-prediction AUC does **not** by itself establish good uplift ranking or causal policy value.

## Supplementary IPW policy diagnostics (not headline performance claims)

The pipeline also reports constant-propensity inverse-probability-weighting (IPW) sums for the selected top-budget group and a random-baseline diagnostic:

| Outcome | Selected learner | IPW selected diagnostic | IPW random diagnostic | Relative difference |
|---|---|---:|---:|---:|
| Visit | s-learner | 1,869.16 | 225.97 | +727.2% |
| Conversion | x-learner | 197.09 | 22.20 | +787.6% |

**Do not describe the relative differences above as a validated percentage increase in actual conversions, sales, campaign ROI, or real-world revenue.** They are **diagnostics** computed on a released, pooled, nonuniformly subsampled dataset using a constant treatment propensity. The records do not establish real-world policy performance for deployment.

## Limitations and interpretation guardrails

1. **Generalizability:** The published Criteo release is nonuniformly subsampled and pooled across experiments. Results characterize the *released analysis dataset* rather than a representative marketing population or business return on advertising spend.
2. **Propensity assumptions:** The 85% assignment rate is a **configured pooled assumption**, not independently documented for every underlying experiment. This matters particularly for the constant-propensity IPW Qini/policy diagnostics.
3. **Treatment versus exposure:** `treatment` indicates assignment; `exposure` is post-treatment. Conditioning on exposure in the primary assignment-based effect would change the estimand and can induce bias.
4. **Holdout discipline:** The selected learner was determined solely from validation scores. Although independent-test scores of all learners are shown, the best test score must **not** be retroactively used as the winner.
5. **Statistical uncertainty:** The reported CIs are ordinary normal approximations. The run does not report cluster-aware, trial-aware, bootstrap, or multiple-model-comparison-adjusted uncertainty. Confidence intervals for the selected top-scoring subset should not be interpreted as validated deployment uncertainty.
6. **X-learner nuisance estimation:** The implemented simplified X-learner constructs pseudo-effects using nuisance predictions from the same training records, not an out-of-fold/cross-fitted variant; this may affect bias or generalization.
7. **Single sample and split:** One modeling sample and fixed random seed (`42`) were recorded. The archive does not include multiple-seed stability estimates, formal learner-difference tests, external validation, calibration, or deployment measurements.
8. **Outcome rarity:** Conversions have low absolute event rates, so even visually large relative lifts can correspond to small absolute percentage-point changes.
9. **Not measured here:** Per-model training time, peak memory, inference latency, hardware configuration, experiment revenue, and online A/B-test performance are **not included** in the supplied benchmark archive.

## Reproduce the benchmarks locally (PowerShell)

Run from the repository root with the official Criteo v2.1 compressed file in `data/raw/`. The dataset is **not** committed to Git. Review its license before obtaining or redistributing it.

```powershell
$python = ".\.venv\Scripts\python.exe"

# Prepare once, from the existing official source file
& $python -m marketing_uplift.pipeline prepare `
    --source data/raw/criteo-uplift-v2.1.csv.gz `
    --full data/processed/criteo_full.parquet `
    --sample data/processed/criteo_model_sample.parquet `
    --sample-rows 1000000

if ($LASTEXITCODE -ne 0) { throw "Preparation failed" }

# Visits
& $python -m marketing_uplift.pipeline benchmark `
    --outcome visit `
    --budget-fraction 0.10 `
    --output outputs/metrics.json

if ($LASTEXITCODE -ne 0) { throw "Visit benchmark failed" }

# Conversions
& $python -m marketing_uplift.pipeline benchmark `
    --outcome conversion `
    --budget-fraction 0.10 `
    --output outputs/conversion_metrics.json

if ($LASTEXITCODE -ne 0) { throw "Conversion benchmark failed" }
```

Saved input artifacts provided for this report were `metrics.json`, `conversion_metrics.json`, and `data_manifest.json` (plus the pipeline-generated candidate résumé summaries). The complete experiment results are repeated in both outcome JSON files, as they analyze the same full prepared dataset. The model and policy results are distinct by outcome.

## Portfolio-ready wording

> Analyzed 13.98 million records from the Criteo Uplift Prediction Dataset v2.1 to estimate assignment-based effects on visits and conversions, finding 1.034 and 0.115 percentage-point absolute differences, respectively. Compared S-, T-, and X-learners on a reproducible 1-million-record sample, selected models using validation-only AUUC, and reported independent-test performance (separate-relative AUUC: 0.010297 for visits, S-learner; 0.000998 for conversions, X-learner).

This wording describes **measured benchmark outcomes**, not deployed gains or realized ROI. Any shorter résumé/LinkedIn claim should retain the distinction between full-dataset ITT effects and held-out model-ranking scores.

---

**Source of truth:** JSON and manifest supplied in `benchmark_results.zip` from the November 2, 2026 run. Metrics are reported at the precision indicated above; rounding is for presentation only. This document should be updated whenever the code, data sample, seed, or benchmark results change.
