# Marketing Experimentation & Uplift Modeling

**Python · DuckDB · pandas/NumPy · SciPy · scikit-learn · randomized experiments · causal uplift**

A reproducible, noncommercial research portfolio project for analyzing randomized advertising-assignment data and deciding **who to target**, rather than merely predicting **who is likely to convert**. It combines full-dataset experiment diagnostics, average treatment effects, S-/T-/X-learner benchmarks, validation-based model selection, an untouched test set, and budget-aware policy diagnostics.

> **Status and disclosure:** The original uploaded project included historical Criteo v2.1 metrics dated 2026-08-31. The revised code changes the splitting/evaluation protocol. Those old model-ranking numbers **must not be presented as results from this revision**. Run the benchmark to regenerate new results. See [docs/HISTORICAL_RESULTS.md](docs/HISTORICAL_RESULTS.md) for correctly labeled original evidence and [docs/RELEASE_AUDIT.md](docs/RELEASE_AUDIT.md) for repaired issues and verification limits.

## What the system does

1. **Load/validate** the licensed Criteo uplift v2.1 dataset (13,979,592 rows) or an explicitly synthetic offline smoke-test dataset.
2. **Profile data** and create a deterministic reservoir sample for model training. The full prepared data remains local.
3. **Audit randomization:** expected-versus-observed assignment ratio (SRM, based on a configurable allocation assumption) and covariate standardized mean differences.
4. **Estimate experimental effects:** intent-to-treat differences in visit/conversion rates; absolute/relative lift; normal-approximation 95% confidence intervals and p-values; approximate 80%-power MDE.
5. **Train three causal meta-learners** using histogram gradient-boosted trees and 12 anonymized baseline features.
6. **Select models honestly:** stratified 60% training / 20% validation / 20% independent test. Select the highest validation separate-relative AUUC; report the selected model's separate test metrics.
7. **Analyze targeting:** Qini/IPW ranking diagnostics, uplift in the selected budget segment, indicative incremental outcomes per 100k eligible records (not revenue or causal deployment proof).

```
External Criteo v2.1 (not distributed)
            |
       download / sample
            |
         prepare -----> full-data Parquet -> experimental ATE + randomization checks
            |
    1M-row reservoir sample
            |
   stratified train / validation / test
            |
       S, T, X learners
            |
   validation AUUC model selection
            |
   independent holdout evaluation
            |
   outputs/metrics.json + predictions.parquet + report.md
```

## Data ownership and restrictions

**The dataset is from Criteo AI Lab, not created by this project.** The original source and publisher notes are at [Criteo AI Lab](https://ailab.criteo.com/criteo-uplift-prediction-dataset/); the [research reference code](https://github.com/criteo-research/large-scale-ITE-UM-benchmark) provides the separate-relative AUUC reference formula. The data are licensed **CC BY-NC-SA 4.0** (attribution, noncommercial, share-alike), are aggregated from several incrementality tests, and have anonymized/transformed features. The source describes nonuniform subsampling, so observed treatment effects are benchmark-specific, not automatically transferable to the original ad audience. **No raw/processed dataset or row-level prediction file is committed.** The code's MIT license, if used, does **not** relicense the Criteo data.

Cite: Eustache Diemert, Artem Betlei, Christophe Renaudin, and Massih-Reza Amini, *A Large Scale Benchmark for Uplift Modeling*, AdKDD/TargetAd Workshop, 2018.

## Quick start

Prerequisites: Python 3.10+ (3.11 or 3.12 recommended), enough RAM for model sample and several GB of free disk. 16 GB RAM is recommended for the default one-million-row run.

### Windows PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pytest -q
```

### Safe offline end-to-end smoke test

This generates and runs **synthetic** observations in separate `data/smoke` paths. These outputs are never acceptable as real-data project results.

```bash
python -m marketing_uplift.pipeline sample --source data/smoke/synthetic-smoke.csv.gz --rows 50000
python -m marketing_uplift.pipeline all --source data/smoke/synthetic-smoke.csv.gz --full data/smoke/full.parquet --sample data/smoke/model_sample.parquet --output outputs/smoke/metrics.json --sample-rows 30000 --outcome visit
```

### Official Criteo analysis (only after reading and accepting the license)

```bash
python -m marketing_uplift.pipeline download --i-accept-license
python -m marketing_uplift.pipeline prepare --sample-rows 1000000
python -m marketing_uplift.pipeline benchmark --outcome visit --budget-fraction 0.10
python -m marketing_uplift.pipeline benchmark --outcome conversion --budget-fraction 0.10 --output outputs/conversion_metrics.json
```

Smaller machine: `python -m marketing_uplift.pipeline prepare --sample-rows 250000`; **sample size must always be disclosed**. `prepare --skip-full-parquet` skips a required analysis file, so an independent full-data file must exist before `benchmark` can run. To use the optional Makefile, run `make setup`, `make smoke`, `make download`, `make prepare`, `make benchmark`, `make test` on a platform with GNU Make.

## Outputs and definitions

All outputs are local/gitignored. Results under a custom output name use a matching filename prefix.

| Output | What it contains |
| --- | --- |
| `data/processed/data_manifest.json` | Source identity, local data quality profile, sample seed and sizes |
| `outputs/metrics.json` | Experiment diagnostics, per-learner validation/test metrics, **validation-selected** winner |
| `outputs/holdout_uplift_predictions.parquet` | Only test-split rank scores, assignment, outcome, sample row indices |
| `outputs/resume_metrics.md` | Provenance-labeled possible portfolio claims (verify before reuse) |

Definitions: **ITT ATE** = observed treatment-assigned outcome rate minus control-assigned rate. **Separate-relative AUUC** = Criteo benchmark's class-conditional AUC transformation, not regular ROC AUC. **Qini** = cumulative IPW gain above a random-targeting diagonal. **Uplift at budget** = top-score subgroup's empirical treated-minus-control rate. The fixed policy value is only a retrospective diagnostic; no incremental cash value, advertising cost, realized ROI, or external deployment outcome is measured.

## Honest technical limitations

- Criteo pools randomized experiments without a trial identifier or time/group metadata. Its 85% overall allocation should **not** be treated as proof of individually identical assignment propensity for every record.
- The separate-relative AUUC formula matches the reference utility; other IPW/Qini metrics use the **marginal** assignment share and are descriptive under heterogeneous trial propensities.
- The 95% confidence intervals are normal approximations that ignore possible cross-trial clustering and model-selection/inference multiplicity. The model-level metrics do not have bootstrap CIs.
- Choosing a learner on validation avoids ranking on the test set, but repeated manual tuning on the reported test set would still leak information.
- Covariates are anonymized; there is no real marketing cost/revenue, customer segmentation explanation, longitudinal outcome, or online deployment.
- Production improvements: cross-fitting, experiment-level propensities/cluster-aware inference if IDs become available, AIPW policy estimation, bootstrap uncertainty, and independent temporal/external validation.

For mathematical derivations and recruiter interview answers, use the **private guide included in the ZIP**, which is ignored by Git and should not be uploaded publicly.

## Quality controls

The included `tests/` check estimators, invalid inputs, metric definitions, and repeatable model selection. GitHub Actions performs a clean installation, runs all tests, and executes a synthetic end-to-end smoke benchmark. Full real-data reproduction requires accepting Criteo's dataset terms and is not run in public CI.
