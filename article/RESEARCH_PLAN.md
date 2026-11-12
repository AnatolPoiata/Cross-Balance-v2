# Cross-Balance — research & development plan (v3, October 2026)

This plan responds to the two reviews in `article/` (`recenzie.docx`, a
major-revision verdict; `Cross_Balance_ECOFORUM_Review.docx`, an editorial
review of the template rewrite). It also covers the new data in
`data/raw/` and the reference material in `references/` and `examples/`.
Section 4 lists the outcomes. All numbers in it come from
`results/v3/analysis/`.

## 1. Diagnosis of v2 (iteration 7), verified in code

| Finding | Evidence | Consequence |
|---|---|---|
| **Precision 0.90 / recall 0.04 at 0.5 on 99:1** (the reviewer's point 4) is caused by the `'boosting'` update, not by a sign or weighting bug. | `classifier.py` doubles the weight of every majority sample that **one member** scores above 0.5, never decreases weights, and resamples **with replacement**. Weights compound as 2^k, so late chunks are made of a few duplicated borderline (and, in the synthetic data, label-noise) samples. Diagnostic on the 99:1 set: `bagging` P=0.19 / R=0.27; `boosting` P=1.00 / R=0.03. | The ensemble learns that "everything near the minority is majority". AP rises because the top of the ranking is cleaner, but scores collapse below 0.5 for most positives. |
| Scores are not probabilities. | Mean score on negatives is 0.31 at 1.5% prevalence. Members train at ~33% prevalence and no correction is applied. | Cost-based decisions with v2 lose money: savings on ULB are −154% at €1 per investigation. |
| The "density-adaptive" ladder is one choice per dataset. | ρ is computed once, from the **capped** chunk size (not the nominal one, as the draft stated), so all 50 members share one estimator family. | The density heuristic adds 4 tuned thresholds that act as a per-dataset model choice. |
| OOF member weights are almost uniform. | 50 members get weights 0.017–0.026 for 4× the fitting cost. Weights are computed on chunks drawn with replacement, so duplicates leak across internal folds (editorial review). | Remove: no measurable gain (ablation below). |
| Baselines were handicapped. | 8 of 10 baselines were unscaled logistic regression (e.g. `unix_time` ≈ 1.6e9 on Sparkov). No gradient-boosting baseline. EasyEnsemble's default AdaBoost was used, but the LR tier dominated the table. | The "6 of 11" claim could not separate imbalance handling from model capacity (reviewer point 1). |
| Ailerons positive share 9.66% ≠ 5%. | `Goal` has 35 distinct values; 1,084 rows tie at the 95th percentile and `>=` keeps them all. | Documented in `DATA_SOURCES.md`. |

## 2. Work packages and status

| WP | Content | Reviewer item | Status |
|---|---|---|---|
| WP1 | Fair baselines: scaled LR (class weights), SMOTE+scaled LR, RF `balanced_subsample`, Balanced RF, EasyEnsemble (AdaBoost), RUSBoost, XGBoost / LightGBM with `scale_pos_weight`, unweighted LightGBM, Self-Paced Ensemble (ported from `imbalanced-ensemble`, which is incompatible with scikit-learn ≥ 1.7) | R1 §1, Rev2 "comparison fairness" | done (`benchmarks/run_v3.py`, `benchmarks/spe.py`) |
| WP2 | Algorithm v3: self-paced hardness sampling on the ensemble-so-far, heterogeneous HistGB/ExtraTrees members, r = 4, uniform weights, prior-corrected probabilities, optional F1 / cost threshold | R1 §3-4 | done (`src/cross_balance`, v0.2.0; v2 kept as `LEGACY_V2_PARAMS`, reproduced bit-for-bit) |
| WP3 | Clean validation: design frozen on development datasets; 17 **hold-out** datasets (ULB credit card + 16 Park & Im sets) first loaded after the freeze | R1 §2, Rev2 "validation independence" | done |
| WP4 | Statistics: 5-fold × 3 repeats, Friedman + Iman-Davenport, Nemenyi CD diagram, Wilcoxon signed-rank with Holm correction, average ranks with ties | R1 §6 | done (`benchmarks/analyze_v3.py`) |
| WP5 | Economic layer: Precision@k / Recall@k / amount recall at fixed review budgets (0.1–1% of transactions), example-dependent cost rule p·amount > c_inv, sensitivity c_inv ∈ {1…50} € on ULB | R1 "ECOFORUM" | done |
| WP6 | Ablation: bagging vs self-paced vs v2 boosting, HistGB-only, density ladder, r = 2, OOF weights, no cap | R1 "ablation" | done |
| WP7 | Article restructure: final method plus ablation in the body, iteration history in Appendix A, 2015–2025 literature, JEL G21, first-person methods text, code-availability statement | R1 §5, Rev2 | done (EN + RO .docx) |
| WP8 | Open items for the next round: isotonic / Platt calibration chosen on development data, full-size PaySim and Sparkov reruns, temporal (out-of-time) validation on ULB, tuned baselines (equal Optuna budget), MESA comparison, Zenodo DOI | — | planned |

## 3. Protocol rules (to keep the evidence clean)

1. No hyper-parameter tuning for any method on hold-out data. v3 defaults
   were frozen after three grids on 10 development datasets (3-fold CV,
   seed 0; `results/v3_devgrid/`).
2. Every method sees identical folds. Scaling and resampling happen inside the
   training fold.
3. Each failure is logged per fold, never dropped silently (`*.json` with `ok=false`).
4. Prevalence is preserved: ULB runs at full size. The 50k subsample of
   PaySim raises its prevalence (0.13% → 0.40%), and the article says so.
5. Negative results stay in the paper (e.g. the v2 boosting failure mode,
   and any hold-out dataset where v3 loses).

## 4. Outcomes (results/v3, 9,600 fits: 12 methods + 8 variants × 32 datasets × 15 folds)

| Question | Answer |
|---|---|
| Does v3 beat the strong baselines on unseen data? | **Yes, on ranking.** On the 17 hold-out datasets v3 has average AP rank 2.65 vs 4.76 for the next method (RF with class weights) and mean AP 0.589 vs 0.561. Wilcoxon-Holm p ≤ 0.033 against all 11 competitors, with 12–17 wins of 17. Friedman / Iman-Davenport p < 0.001. Nemenyi CD = 4.04, so the conservative test separates v3 only from methods ranked worse than 6.69. |
| …and on the development data? | Rank 2.77 (best), but the gaps to XGBoost, LightGBM-w and LightGBM are **not** significant after Holm (p = 0.074). |
| Per hold-out dataset | Rank 1 on 7/17, top 3 on 11/17. Clear losses on winequality-red 8 vs 6 (rank 6) and yeast-2 vs 8 (rank 5), both with ≤ 20 positives. |
| Which components matter? (ablation, 32 sets) | Self-paced vs uniform bagging +0.048 AP; mixed vs HistGB-only +0.032; dropping the v2 density ladder +0.052; r = 4 vs 2 +0.010; vs v2 boosting update +0.007 (all p_holm < 0.01). OOF weights: n.s. (+0.003 AP for the weighted variant, at 2.5× cost). Cap: n.s. (tie on 23/32). |
| Probabilities | Prior correction: mean Brier 0.079 (v2) → 0.037 (v3), close to XGBoost (0.041). It over-shrinks on ULB, though: mean p 0.039% vs prevalence 0.173%. |
| Thresholds | At 0.5, v3 F1 = 0.24 (calibrated probabilities are cautious). With `decision_threshold='f1'` (internal 3-fold), F1 = 0.66 and MCC = 0.64, the best of all methods (XGBoost 0.63 / 0.60). |
| Fraud economics (ULB, 284,807 tx) | Best AP (0.867) and best precision at a 0.1% review budget (0.993); 81% of defrauded amount found at a 0.5% budget. Under the cost rule p·amount > c, v2 loses money (−154% at c = €1); v3 saves 75%, but XGBoost (82%) and RF (84%) save more because they are better calibrated. |
| Cost | v3 fit 6.1 s on average vs XGBoost 1.1 s, LightGBM 0.8 s, RF-300 16.2 s (one core). |

### Next steps (WP8, prioritised)

1. **Calibration**: add `calibration='isotonic'` fitted on internal OOF scores and select it on development data only. Then re-test the ULB cost rule. This is the only place v3 currently loses money relative to XGBoost.
2. **Equal-budget tuning**: give every method the same Optuna budget (for example 30 trials, inner CV) to test whether the hold-out advantage survives tuned XGBoost / LightGBM.
3. **Out-of-time validation** on ULB (split by `Time`), and full-size PaySim and Sparkov.
4. **Speed**: fit members in parallel (`n_jobs`), and early-stop K when the hardness distribution stabilises.
5. Small-data regime (< 25 positives): consider r adaptive to n_min, and report fold variance.
6. Deposit code and fold-level predictions on Zenodo (DOI) before submission; add the copyright form.

## 5. Revision 1 (6 Oct 2026): response to the 19-point internal review

Files: `article/CrossBalance_ECOFORUM_{EN,RO}_rev1.docx` (the previous version is kept as
`CrossBalance_ECOFORUM_{EN,RO}.docx`).

New evidence produced for this revision:
- **Fair threshold comparison** (`benchmarks/thresholds_v3.py`, `results/v3/thresholds_f1/`;
  analysis `python -m benchmarks.analyze_v3 --thresholds`). Every baseline gets the same
  internal 3-fold F1-optimal threshold. Results:
  - Over 32 datasets, Cross-Balance still has the best mean F1 (0.657) and MCC (0.640); the next best is RF-w at 0.635 / 0.616.
  - On the hold-out group it is significant after Holm against only 4 of 10 baselines for F1 and 5 for MCC.
  - The earlier claim "best values of all methods" was dropped.
- **r/K sensitivity** (`results/v3_devgrid/sensitivity_rK.*`, same 10 selection sets and
  protocol): r ∈ {5, 6, 8} and K ∈ {75, 100} change mean AP by at most +0.002. The default was kept.
- **ULB Brier ×10³:** 1.39 corrected, 1.27 uncorrected, 1.73 for an always-zero model, 0.41 for XGBoost. The prior correction over-shrinks under hardness sampling (mean p 0.039% vs 0.173%).
- **Development ranks without the 4 near-balanced sets:** Cross-Balance 2.05 vs LightGBM 3.14.

Text changes:
- Abstract and conclusion now say "default-configured", "significantly higher AP in paired Wilcoxon-Holm tests" and "prior correction for the probability distortion".
- The ULB result is presented as parity with XGBoost.
- The "32 datasets" results and the ablation are flagged as optimistic or post hoc.
- The unsupported claims were softened or given sources.
- Seven dataset sources were added, plus the UCI, Holm and Pedregosa references.
- The 10-of-15 selection subset is stated consistently.
- A credit-risk limitation was added.
- Equations were fixed (real OMML subscripts, italic variables). The `*` markup bug for the literal `***` references was fixed too.
