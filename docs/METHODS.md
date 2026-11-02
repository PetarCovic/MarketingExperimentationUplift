# Methods reference (public-facing)

For the extensive technical walkthrough, assumptions, mathematical derivations, interview questions and personal preparation notes, see the privately included `private/Marketing_Uplift_Interview_Guide.docx` from the distribution ZIP; this path is excluded from Git tracking.

## Core definitions

- Feature vector: `X = (f0,...,f11)`, 12 anonymized baseline numeric covariates.
- Assignment: `T ∈ {0,1}`; outcomes `Y_visit,Y_conversion ∈ {0,1}`.
- Potential outcomes: `Y(1), Y(0)`; individual causal effect `tau(x) = E[Y(1)-Y(0) | X=x]` is **not** directly observed for any one record.
- ITT: `E[Y|T=1] - E[Y|T=0]` under randomized assignment and assumptions; exposure is post-treatment and is not conditioned upon in the primary analysis.

### Experiment and diagnostics

1. `ATE_hat = p_t - p_c` with sample outcome rates in assignment arms.
2. `SE_hat = sqrt(s_t^2/n_t + s_c^2/n_c)`, normal CI `ATE_hat ± 1.96 SE_hat` (approximate).
3. `relative_lift = 100 * ATE_hat/p_c` where `p_c > 0`. A +1 p.p. effect is **not** equal to a +1% relative lift.
4. SRM Pearson statistic `sum_g (observed_g - expected_g)^2 / expected_g`, 1 d.f. The configured expected treated proportion is 0.85.
5. `SMD_j = (mean_t,j - mean_c,j) / sqrt((s_t,j^2 + s_c,j^2)/2)`, ignoring undefined features.
6. Approximate 80%-power, two-sided alpha 5% minimum detectable **absolute** effect: `(z_.975 + z_.8) * sqrt(pbar*(1-pbar)*(1/n_t+1/n_c))`.

### Causal learners

- **S-learner:** fit `m(X,T) ≈ P(Y=1|X,T)`; estimate `tau_S(X) = m(X,1) - m(X,0)`.
- **T-learner:** fit two models `m1(X),m0(X)` on assigned treatment and control separately; `tau_T(X) = m1(X)-m0(X)`.
- **X-learner:** form pseudo-effects `D1=Y1-m0(X1)` on treated and `D0=m1(X0)-Y0` on controls; fit regressors `tau1,tau0` and combine `tau_X(X) = e*tau0(X) + (1-e)*tau1(X)`, using training-arm share `e` here. This is a simplified constant-propensity X-learner, not a fully cross-fitted trial-aware estimator.
- All implementations use histogram gradient-boosted estimators, fixed seed and fixed hyperparameters; features are baseline, not post-treatment.

### Ranking and policy

- S-/T-/X-learners train on 60% of eligible sample. Validation (20%) selects via the **separate-relative AUUC** formula published by Criteo; the final 20% independent test measures the selected learner.
- Let `p1,p0` be outcome rates in treatment/control, `AUC1,AUC0` their within-arm label-versus-uplift-score ROC AUCs. `AUUC_sep_rel = p1(1-p1) AUC1 - p0(1-p0) AUC0 + (p1^2-p0^2)/2`. This is a **ranking-quality benchmark**, not a 0-to-1 binary classifier accuracy.
- For Qini diagnostics define IPW response `Z_i = T_iY_i/e - (1-T_i)Y_i/(1-e)` with *pooled* assignment share `e`, and cumulative sorted gain `G(k)=Σ_{i≤k} Z_(i)`. The random-target diagonal is `k/n * G(n)`. Qini area integrates `G(k) - random(k)` and divides by `n`.
- At budget fraction `b`, select `ceil(b*n)` highest estimated-uplift test rows. Estimate subgroup assignment effect `mean(Y|T=1, selected)-mean(Y|T=0, selected)`, if both arms have enough rows. Multiply by `b_actual * 100000` for indicative incremental outcomes per 100k eligible records **in this dataset**, with all external-validity limitations above.
- Constant-propensity IPW sums are **diagnostics only**, particularly because the original experiment identifier and local propensities were not published.
