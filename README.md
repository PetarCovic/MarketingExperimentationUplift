# Marketing Experimentation & Uplift Modeling

**Causal inference, randomized-experiment analysis, and uplift modeling on 13.98 million advertising records.**

`Python` · `SQL` · `DuckDB` · `pandas` · `NumPy` · `SciPy` · `scikit-learn` · `Parquet` · `A/B Testing` · `Uplift Modeling`

This project implements an end-to-end, reproducible analysis pipeline for the **Criteo Uplift Prediction Dataset v2.1**. It estimates the effects of randomized advertising **assignment** on visits and conversions, benchmarks three approaches to heterogeneous treatment-effect estimation, and evaluates which audience segments might benefit most from treatment.

The central question is **“Who is likely to respond *because of* an intervention?”**, not simply “Who is likely to convert?”

> **Status:** Official-data benchmarks completed **November 2, 2026**. The current evaluation protocol selects a learner on validation data and reports its performance on an independent test set. See **[BENCHMARKS.md](BENCHMARKS.md)** for full results, metric definitions, and caveats. This is an offline research/portfolio project, not a deployed advertising system.

## Results at a glance

### Randomized-assignment effects — full 13,979,592-row dataset

| Outcome | Control rate | Treatment-assigned rate | Absolute difference | Relative lift | Approx. 95% CI for absolute difference |
|:--|--:|--:|--:|--:|:--|
| **Visits** | 3.8201% | 4.8543% | **+1.0342 percentage points** | **+27.07%** | [+1.0056, +1.0629] pp |
| **Conversions** | 0.1938% | 0.3089% | **+0.1152 percentage points** | **+59.45%** | [+0.1085, +0.1219] pp |

These are **intent-to-treat (ITT)** comparisons between assignment groups in the *released benchmark data*, not evidence of realized revenue, advertising return on investment, or a production targeting campaign. Criteo's pooled and nonuniformly subsampled release limits generalization.

### Uplift models — 1,000,000-row sample

| Outcome | Learner selected on validation | Independent-test separate-relative AUUC | Test top-10% subgroup effect |
|:--|:--|--:|--:|
| **Visits** | **S-learner** | **0.010297** | **+6.4223 pp** |
| **Conversions** | **X-learner** | **0.000998** | **+0.8569 pp** |

- Model development uses **600,000 training**, **200,000 validation**, and **200,000 independent-test** records, stratified by assignment and the relevant outcome.
- The selected learner is the one with the **highest validation separate-relative AUUC**, not the best retrospective test score.
- At a **10% targeting budget**, the subgroup effect is the observed treatment-minus-control outcome-rate difference among the **20,000 highest-ranked test observations**. It is not a directly observed gain from deploying a targeting policy.
- **Nuance:** For conversions, the validation-selected X-learner scored `0.000997627` on test separate-relative AUUC, while the S-learner scored `0.000997993`. The test result does **not** change the preselected winner, and no statistical significance test of that small difference was performed.

[**Read the complete benchmark report →**](BENCHMARKS.md)

## What the pipeline does

| Component | Implementation |
|:--|:--|
| **Data ingestion & preparation** | Reads the compressed Criteo source; checks required columns and binary outcomes; writes ZSTD-compressed Parquet; builds a fixed-seed DuckDB reservoir sample. |
| **Experiment diagnostics** | Checks the observed treatment allocation against a configured 85% expectation, summarizes missingness, and calculates baseline standardized mean differences. |
| **Causal effect estimation** | Computes ITT visit/conversion differences, relative lifts, approximate confidence intervals, hypothesis tests, and descriptive minimum-detectable-effect calculations. |
| **Uplift modeling** | Implements S-, T-, and simplified X-learners using scikit-learn histogram gradient-boosting models. |
| **Model selection** | Separates training, validation, and independent test data; chooses each outcome's learner using validation separate-relative AUUC. |
| **Targeting analysis** | Reports AUUC/Qini ranking diagnostics, top-budget subgroup estimates, and illustrative incremental outcomes under explicit assumptions. |
| **Reproducibility & testing** | Records seeds, sample provenance, model results, and environment metadata; runs unit tests and an offline synthetic smoke test in GitHub Actions. |

### Architecture

```text
                   Criteo v2.1 CSV.gz (external; not committed)
                                      |
                                      v
                         Validation + DuckDB preparation
                             /                    \
                            v                      v
              Full-data Parquet             1M-row modeling Parquet
                     |                              |
                     v                              v
             Assignment checks              60% train / 20% validation
             Covariate balance                  / 20% test split
             Visit/conversion ITT                    |
                                                     v
                                                S / T / X learners
                                                     |
                                                     v
                                        Select by validation AUUC
                                                     |
                                                     v
                                        Independent-test evaluation
                                                     |
                                                     v
                                        Ranking + top-10% diagnostics
                                                     |
                                                     v
                                      Local JSON, Parquet, Markdown
```

## Statistical methodology

**Average assignment effect (ITT).** For a binary outcome $Y$ and assignment indicator $T$, the estimate is

$$
\widehat{\Delta}_{\mathrm{ITT}}=\overline{Y}_{T=1}-\overline{Y}_{T=0}.
$$

**Conditional treatment effect.** Uplift modeling estimates

$$
\tau(x)=\mathbb{E}[Y(1)-Y(0)\mid X=x],
$$

where $X$ contains the dataset's 12 anonymized baseline features (`f0`–`f11`). Each person's counterfactual outcome is unobserved, so individual uplift scores are **estimates**, not known labels.

- **S-learner:** Fits one outcome model that includes treatment assignment; computes the difference between predicted treated and untreated outcomes.
- **T-learner:** Fits independent outcome models for the two assignment arms and subtracts their predicted probabilities.
- **X-learner:** Constructs imputed treatment-effect targets and trains effect models for both arms; this implementation uses the **pooled training treatment fraction** as a constant combination weight and **does not use cross-fitting**.
- **Evaluation:** Uses Criteo's **separate-relative AUUC** for validation selection and reports Qini, per-arm outcome AUC, and top-budget effect diagnostics on an untouched test split.

**Assignment is not exposure:** The `exposure` field is post-treatment and is deliberately excluded from baseline model features and the primary ITT comparison.

For equations, assumptions, and implementation details, see [docs/METHODS.md](docs/METHODS.md). For full values and model-by-model comparisons, see [BENCHMARKS.md](BENCHMARKS.md).

## Getting started

### Requirements

- **64-bit Python 3.11 recommended**; the package declares compatibility with Python **3.10–3.13**. Python 3.14+ is unsupported by the pinned PyArrow range.
- Enough free disk space for the compressed source and intermediate Parquet files; **16 GB RAM recommended** for the default one-million-record modeling run.
- For official-data analysis, obtain the **Criteo Uplift Prediction Dataset v2.1** subject to its separate license (see [Data provenance and licensing](#data-provenance-and-licensing)).

### Windows PowerShell setup

From the repository root:

```powershell
py -3.11 -m venv .venv
$python = ".\.venv\Scripts\python.exe"

& $python -m pip install --upgrade pip setuptools wheel
& $python -m pip install --only-binary=:all: "pyarrow>=15,<22"
& $python -m pip install -e ".[dev]"
& $python -m pytest -q
```

The binary-only PyArrow installation avoids the accidental CMake source build encountered with unsupported Python versions. If `py -3.11` is unavailable, install the 64-bit Python 3.11 interpreter first.

### Offline smoke test (no external dataset needed)

This generates a **synthetic**, 50,000-row source and runs the entire pipeline on a 30,000-row modeling sample. Synthetic results must **not** be represented as Criteo findings.

```powershell
& $python -m marketing_uplift.pipeline sample `
    --source data/smoke/synthetic-smoke.csv.gz `
    --rows 50000

& $python -m marketing_uplift.pipeline all `
    --source data/smoke/synthetic-smoke.csv.gz `
    --full data/smoke/full.parquet `
    --sample data/smoke/model_sample.parquet `
    --output outputs/smoke/metrics.json `
    --sample-rows 30000 `
    --outcome visit
```

### Official Criteo benchmarks

If you **already have** `criteo-uplift-v2.1.csv.gz`, put it in `data/raw/` and **skip the download command**. If you need to obtain it, first review the terms at the [Criteo AI Lab dataset page](https://ailab.criteo.com/criteo-uplift-prediction-dataset/), then use the built-in downloader:

```powershell
# Optional; not needed when the local compressed dataset is already present.
& $python -m marketing_uplift.pipeline download --i-accept-license
```

The revised downloader points to Criteo's Hugging Face distribution and checks the configured source SHA-256. Network access and availability of the upstream host are required; download failures do not indicate that model training failed.

Prepare the official data **once**:

```powershell
& $python -m marketing_uplift.pipeline prepare `
    --source data/raw/criteo-uplift-v2.1.csv.gz `
    --full data/processed/criteo_full.parquet `
    --sample data/processed/criteo_model_sample.parquet `
    --sample-rows 1000000

if ($LASTEXITCODE -ne 0) { throw "Data preparation failed" }
```

Run both independent outcome benchmarks:

```powershell
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

A smaller trial can use `--sample-rows 250000`, but any reported results must disclose the smaller sample and should not be described as reproducing the benchmark above.

### macOS / Linux setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pytest -q
```

Use `python -m marketing_uplift.pipeline` in place of `& $python -m marketing_uplift.pipeline` for the commands above. The project also declares an `uplift-benchmark` console entry point.

## Outputs and repository layout

```text
MarketingExperimentationUplift/
├── README.md
├── BENCHMARKS.md                  # Public summary of the completed real-data runs
├── pyproject.toml
├── .github/workflows/ci.yml     # Unit tests + offline smoke test
├── docs/
│   ├── METHODS.md
│   ├── HISTORICAL_RESULTS.md     # Original pre-refactor results, clearly separated
│   └── RELEASE_AUDIT.md         # Changes and verification limits
├── src/marketing_uplift/
│   ├── experiment.py
│   ├── models.py
│   └── pipeline.py
├── tests/
├── data/                        # Local data; contents excluded from Git
└── outputs/                     # Generated metrics/predictions; excluded from Git
```

For the default output names, the pipeline generates:

| Local artifact | Contents |
|:--|:--|
| `data/processed/data_manifest.json` | Data provenance, missingness, row counts, fixed sampling seed |
| `outputs/metrics.json` | Complete visit experiment/validation/test metrics |
| `outputs/conversion_metrics.json` | Complete conversion experiment/validation/test metrics |
| `outputs/holdout_uplift_predictions.parquet` | Visit holdout treatment assignments, outcomes, and model scores |
| `outputs/conversion_metrics_holdout_uplift_predictions.parquet` | Conversion holdout predictions and scores |
| `outputs/resume_metrics.md` | Generated visit-run portfolio summary |
| `outputs/conversion_metrics_resume_metrics.md` | Generated conversion-run portfolio summary |

**Only the curated, aggregate [`BENCHMARKS.md`](BENCHMARKS.md) is intended to be committed publicly.** Do not commit the licensed row-level data, holdout predictions, virtual environment, or private interview guide. The included `.gitignore` excludes those paths.

## Important limitations

1. **Released data ≠ target population.** Criteo's release pools experiments and was nonuniformly subsampled. Full-data ITT effects characterize the released benchmark sample and may not transport to the original advertising population.
2. **Propensity assumption.** The 85% treated share is a configured *pooled* assumption. Per-experiment propensity scores and experiment IDs are unavailable; constant-propensity IPW/Qini policy numbers are **diagnostic only**.
3. **Uncertainty.** The reported 95% intervals are normal approximations, not trial-clustered or selection-adjusted intervals. Model-ranking differences were not formally tested.
4. **Holdout integrity.** Learners are selected on validation, not test, though repeatedly using test results for future tuning would undermine its independence.
5. **Simplified X-learner.** Nuisance models are not cross-fitted; the model is not claimed to be a doubly robust or trial-aware causal estimator.
6. **No business deployment.** The project does not measure actual campaign revenue, ad spend, production latency, online lift, or realized return on investment. Top-budget results are retrospective assignment-group comparisons in held-out data.

See [BENCHMARKS.md](BENCHMARKS.md) for a more detailed interpretation of each reported metric and its assumptions.

## Data provenance and licensing

- **Dataset:** [Criteo Uplift Prediction Dataset v2.1](https://ailab.criteo.com/criteo-uplift-prediction-dataset/), provided by Criteo AI Lab. The data are subject to **CC BY-NC-SA 4.0** and are **not** redistributed here.
- **Reference methodology:** [Criteo Research — Large Scale ITE/Uplift Modeling Benchmark](https://github.com/criteo-research/large-scale-ITE-UM-benchmark).
- **Citation:** Diemert, E., Betlei, A., Renaudin, C., and Amini, M.-R. (2018). *A Large Scale Benchmark for Uplift Modeling*. AdKDD/TargetAd Workshop.
- **Project source code:** [MIT License](LICENSE). This license applies to this project's original code, **not** the third-party Criteo dataset or dependencies.

## Further reading

- [Full benchmark results](BENCHMARKS.md)
- [Statistical methods and formulas](docs/METHODS.md)
- [Release audit and verification](docs/RELEASE_AUDIT.md)
- [Earlier pre-refactor results (historical only)](docs/HISTORICAL_RESULTS.md)
