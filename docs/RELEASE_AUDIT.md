# Code audit and release notes

**Revision date:** 2026-11-02. **Scope:** Uploaded `MarketingExperimentationUplift - Copy.zip` inspected and repackaged for a noncommercial public source-code release.

## Confirmed problems and changes

| Priority | Original behavior | Resolution / current behavior |
| --- | --- | --- |
| Critical | Model winner selected from the exact 30% holdout used to report its result; this makes the reported winner susceptible to selection optimism. | New stratified train/validation/test (60/20/20); winning S/T/X learner is selected by *validation* separate-relative AUUC; independent test reports are clearly distinguished. |
| High | Synthetic `sample` command defaulted to the official-data input path and could replace a downloaded local file. | Synthetic input defaults to `data/smoke/synthetic-smoke.csv.gz`; smoke outputs have separate files and explicit synthetic provenance. |
| High | No robust validation of binary treatment/outcome values and insufficient per-arm classes, which could yield silently distorted estimates or cryptic AUC exceptions. | Validate binary outcome and treatment, nonzero group representation, finite predictions/features, required features, size and budget. Errors are explicit. |
| High | Old policy estimates were presented alongside results without obvious labels for constant-propensity assumptions. | IPW-derived values are suffixed `_ipw_diagnostic`; empirical subgroup ITT and normal-approximation CI are separate, and outcomes per 100k eligible are derived from subgroup ITT × selected fraction. Still not revenue. |
| High | `.venv` and large licensed datasets included in distribution ZIP. | Source-only ZIP excludes environments, all external data, predictions and private preparation from Git; CI downloads dependencies, not Criteo data. |
| Medium | Report generator tied to old winner selected on test set. | Report uses validation-selected winner and its independent-test metrics. |
| Medium | Null or invalid features/constant group variance could propagate misleading or invalid values. | Required-feature checks, binary checks, nonfinite removal in the modeling sample, and explicit `None` for undefined group SMD or zero-control-rate relative lift. |
| Medium | Wrong/ambiguous treatment of nearly-zero p-values and MDE. | CI uses exact 97.5% normal quantile; MDE is explicitly labeled post-hoc, approximate; p-value can numerically underflow and should be reported as a bound. |
| Medium | Dependency range allowed NumPy 1.26 but ranking metrics depended on `numpy.trapezoid` (NumPy 2+). | Switch to `scipy.integrate.trapezoid` (SciPy is an explicit dependency). |
| Medium | Download streamed straight into final file, risking a partial trusted-looking data file on an interrupted transfer. | Download streams into a temporary file and only replaces the target on success. SHA256 is recorded in a manifest. |
| Medium | Thin test coverage and no automated CI. | Added model and negative-path tests, plus an optional end-to-end DuckDB integration test and GitHub Actions synthetic smoke. |
| Medium | Original documentation mixed benchmark results and portfolio claims. | New README, mathematical overview, historical-results disclosure, and private recruiter interview guide. |

## Verification performed

- Source inspection across `experiment.py`, `models.py`, `pipeline.py`, tests, packaging and original outputs.
- Python syntax/compile check: passed.
- Modeling/metric tests: passed in this review environment, using installed pandas, SciPy, scikit-learn and NumPy.
- **Important environment limit:** DuckDB and PyArrow were not installed in the reviewer environment and network package installation failed. This means the revised end-to-end CSV/Parquet path and full real-data experiment **were not executed here**. The Docker/GitHub CI integration checks are prepared but must complete on GitHub (or in an environment with dependencies) before the revised project can be described as fully validated.
- Original 2026-08-31 metrics were examined as **historical uploaded artifacts**, not recalculated. They cannot be used to assert revised model performance.

## Open technical/research limitations (not fixed by refactor)

1. **No experiment IDs or true per-trial assignment propensities** are available in the public data; the aggregate 85% assignment check is a diagnostic, not a full design audit.
2. **Nonuniform dataset subsampling** can limit external validity and does not license a real advertising ROI estimate.
3. **Normal CIs** ignore trial clustering; subgroup and ranked policy inference remains exploratory and not adjusted for repeated comparisons.
4. **No AIPW / doubly robust off-policy evaluation**, experiment-specific weights, or cross-fitted nuisance functions.
5. **No pre-registration, online deployment, time split, cost/revenue data, or external replication**.
6. **No guarantee of identical byte-for-byte model outputs** across Python/scikit-learn/compiler/CPU versions; record package versions for full reproducibility.
7. **License review** is necessary for public/noncommercial redistribution of derived artifacts; no Criteo rows or held-out prediction rows are shipped.

## First actions before GitHub publication

1. Run `python -m pip install -e ".[dev]"`, `python -m pytest -q`, and both offline smoke-test commands on your own computer.
2. Run the official Criteo `download`, `prepare`, and both `benchmark` commands. Verify new output and selected validation winner; do not copy historic model ranking figures into new descriptions.
3. `git status --short` after `git add .` to confirm no `.venv/`, `data/*` files, `outputs/*`, or `private/` files have entered tracking.
4. Verify the GitHub Actions test workflow passes on the newly created remote repository.
5. If using the MIT license, verify the copyright name/date before public release. Criteo's data remain under CC BY-NC-SA 4.0.
