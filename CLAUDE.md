# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Cross-Balance is an ensemble method for imbalanced binary classification (99:1-style
ratios). It partitions the majority class into many balanced-ratio chunks, pairs each
chunk with the full minority class, trains one base estimator per chunk, and combines
predictions via out-of-fold performance-weighted averaging. `src/cross_balance/` is
the library; `benchmarks/` is a separate harness that benchmarks it against 10
established imbalance-handling methods (SMOTE, ADASYN, Balanced Random Forest,
EasyEnsemble, etc.) on synthetic and real-world datasets. The two are decoupled —
`benchmarks/` imports `cross_balance` as a normal dependency, it isn't part of the
package.

This is v2 of the project — a clean, packaged, tested rebuild of an informal
prototype that lives in the sibling `../Cross-Balance/` folder (`cbc_v1.py` →
`cbc_v2.py`), which is left untouched as an archive.

## Environment

**Always use `./.venv/bin/python` / `./.venv/bin/pip`, never the global `python3`.**
This project's `.venv` is intentionally isolated from the machine's shared
Anaconda `base` environment — an earlier editable install accidentally ran against
the global environment and downgraded its `numpy`, which is exactly what the
isolated venv exists to prevent. If the venv is ever missing: `python3 -m venv
.venv && ./.venv/bin/pip install -e ".[boost,plot,bench,dev]"`.

**This project lives on a network volume that generates macOS AppleDouble sidecar
files** (`._foo.py`, `._data.csv`, etc.) for every file written, including inside
`.venv/` and after every `pip install`. These can crash `pip` (it globs
`*.dist-info` and chokes on the binary sidecar) and get picked up by naive
`glob("*.csv")` calls in dataset loaders. Loaders already filter `._*`-prefixed
files; if you add a new one that globs a directory, filter them too. Periodically
`find . -name "._*" -delete` before committing.

**Never edit `src/cross_balance/` while a background benchmark run depends on
it.** The package is installed editable, so any change to library source takes
effect immediately for new subprocesses — including ones already queued in a
background script. This caused real data contamination once (see the "Fix data
contamination" commit): a mid-run edit to the base-estimator selection logic
silently changed which model several already-launched benchmark iterations used.
Finish (or explicitly pause) any in-flight `benchmarks.run_benchmark` /
`benchmarks.tune_thresholds` run before touching `src/`.

## Commands

Install (editable, with all extras):
```bash
./.venv/bin/pip install -e ".[boost,plot,bench,dev]"
```

Run the full test suite:
```bash
./.venv/bin/python -m pytest tests/ -q
```

Run a single test:
```bash
./.venv/bin/python -m pytest tests/test_classifier.py::test_defaults_match_validated_config -v
```

Run the benchmark harness (10 baseline methods + Cross-Balance, k-fold CV):
```bash
./.venv/bin/python -m benchmarks.run_benchmark --dataset synthetic:99-1 --folds 10
./.venv/bin/python -m benchmarks.run_benchmark --dataset synthetic:all --n-samples 50000
./.venv/bin/python -m benchmarks.run_benchmark --dataset real:credit_card_fraud --folds 10
./.venv/bin/python -m benchmarks.run_benchmark --dataset real:all --max-samples 50000
```
Pass `--out-root results/iteration_N` to avoid overwriting a prior snapshot (see
"Iteration workflow" below). Real datasets require the corresponding file(s) to
already exist under `data/raw/<name>/` — see `DATA_SOURCES.md` for download links
and expected layout; nothing under `data/raw/` is committed.

Re-tune `CrossBalanceClassifier`'s `auto_density_thresholds` against the held-out
synthetic tuning suite (writes `results/threshold_tuning.txt`):
```bash
./.venv/bin/python -m benchmarks.tune_thresholds
```

Sweep `ratio` in `{1,2,3,4,5}` against the same held-out suite, thresholds held
fixed (writes `results/ratio_tuning.txt`):
```bash
./.venv/bin/python -m benchmarks.tune_ratio
```

Regenerate the article's summary figures (iteration-progression line chart, ranked
method comparison, full-scale-vs-subsampled comparison, tier-ladder and
architecture diagrams) into `article/figures/`:
```bash
./.venv/bin/python -m article.make_figures
```

Export the article to Word (pandoc doesn't render Mermaid, so the mermaid fence in
`draft2.md` must first be swapped for `figures/architecture.png` in a throwaway
copy — see `article/draft2_docx_src.md` for the last-generated one, or redo the
swap after editing `draft2.md`):
```bash
pandoc article/draft2_docx_src.md -o article/Cross-Balance_draft2.docx \
  --resource-path=article --from=markdown+tex_math_dollars --toc --toc-depth=2
```

Large runs (`real:all`, `synthetic:all`) can take minutes; launch with
`nohup ... & disown` and poll the process rather than blocking foreground Bash,
since long-running foreground/background tool calls in this environment have been
observed to get killed on their own timeout even when the underlying process is
still healthy.

## v3 (October 2026) — read this first

The library defaults changed in v0.2.0 (see `article/RESEARCH_PLAN.md`):
`ensemble_type='self_paced'`, `base_estimator='mixed'` (alternating
HistGB / ExtraTrees), `ratio=4.0`, `weighting='uniform'`,
`calibration='prior'`, `decision_threshold=0.5` (or `'f1'`). The
iteration-7 config is `LEGACY_V2_PARAMS` and reproduces old results
bit-for-bit. `test_defaults_match_validated_config` pins the v0.2 defaults.

Evaluation discipline: the v0.2 design was frozen on the development
datasets only (`results/v3_devgrid/`). `real_datasets.HOLDOUT_DATASETS`
(ULB `creditcard_ulb` + 16 `rr_*` Park & Im sets) were first loaded after
the freeze. **Do not tune anything on hold-out results.** Choose new
components on `DEVELOPMENT_DATASETS` and report hold-out once.

Pipeline: `benchmarks.run_v3` (one `.json` + `.npz` of test-fold predictions
per (dataset, method, repeat, fold) in `results/v3/preds/`; resumable,
since completed jobs are skipped) → `benchmarks.analyze_v3` (metrics, Friedman /
Nemenyi / Wilcoxon-Holm, ablation, ULB cost analysis) →
`benchmarks.analyze_v3 --article` → `article.make_figures_v3` → `article.build_article` (EN + RO ECOFORUM
.docx from `docx_builder.py` + `text_front.py`; all results numbers are
read from `results/v3/analysis_article`, never typed; the article presents the method as new, so the legacy v2 config is excluded there). `benchmarks/spe.py` is a
port of Self-paced Ensemble, because `imbalanced-ensemble` cannot be installed:
it pins scikit-learn < 1.7 and **silently downgraded the venv once**. If
you install anything, re-check `sklearn.__version__ == 1.9.0`.

## Architecture

### The library (`src/cross_balance/`)

- `classifier.py` — `CrossBalanceClassifier`, an sklearn-compatible
  `(ClassifierMixin, BaseEstimator)` (mixin **must** come first in that base-class
  order, or `__sklearn_tags__` resolves to `BaseEstimator`'s version and breaks
  scorer response-value selection — this exact bug happened once). Key knobs:
  - `base_estimator='auto'` picks per-chunk from a 5-tier capacity ladder by
    samples-per-feature density: `OMPClassifier` → `LogisticRegression` →
    `DecisionTreeClassifier` → `RandomForestClassifier` → `GradientBoostingClassifier`,
    boundaries set by `auto_density_thresholds` (4 floats). Defaults were chosen by
    tuning against a held-out synthetic suite disjoint from the benchmark datasets
    (`benchmarks/tune_thresholds.py`), not by fitting on the benchmark itself.
  - `ensemble_type`: `'bagging'` (sequential wrap-around chunks) or `'boosting'`
    (cost-sensitive weighted resampling that upweights majority samples the
    ensemble-so-far misclassifies).
  - `max_chunk_size` caps total per-chunk samples (scaling minority and majority
    subsamples down proportionally). Needed because "pair full minority with
    ratio×minority majority" doesn't bound chunk size when the minority class
    itself is large (e.g. a near-balanced dataset) — without it, the
    RandomForest/GradientBoosting tiers can become impractically slow.
  - `weighting='oof'` scores each chunk's estimator via an internal k-fold
    out-of-fold split (`internal_splits`), then weights that member's contribution
    to `predict_proba` by its OOF average-precision.
- `estimators.py` — `OMPClassifier`, wraps sklearn's `OrthogonalMatchingPursuit`
  (a regressor, no native classifier form) as a binary classifier: fits on a
  `{0,1}`-encoded target, clips predictions to `[0,1]` as a probability surrogate.
- `_partition.py` / `_weighting.py` — chunk-sampling strategies and the
  out-of-fold scoring helper, factored out of `classifier.py` for testability.

### The benchmark harness (`benchmarks/`)

- `methods.py` — the fixed registry of 10 baseline methods + `CrossBalance`,
  built fresh per run via `build_methods(random_state=...)`.
- `datasets.py` — synthetic dataset generation (`make_synthetic_imbalanced`,
  cached to `data/processed/synthetic/`). Named ratios are keyed by actual
  majority:minority percentage pairs that sum to 100 (`"99-1"`, `"98-2"`,
  `"95-5"`) — an earlier version used `"99-2"`/`"99-5"`, which was wrong, and got
  renamed; don't reintroduce mismatched pairs.
- `real_datasets.py` — loaders for the real-world datasets in `DATA_SOURCES.md`
  (fraud/crime/telemetry datasets from the original benchmark plus four UCI
  datasets added during extended validation: `default_credit_card_clients`,
  `mammographic`, `mushrooms`, `bank_marketing`), plus `stratified_subsample()`
  for capping huge datasets while preserving class ratio (with a minority-count
  floor so extreme-imbalance sets still have enough positives for k-fold CV).
  `_drop_id_like_columns`'s all-unique-values heuristic only applies to
  non-float columns — a naturally all-unique continuous feature (e.g.
  transaction `amount`) must not be dropped as if it were an ID column.
- `metrics.py` — AUCPR, precision/recall, macro-F1, G-mean, MCC, balanced
  accuracy.
- `report.py` — writes `results_summary.csv`, `report.html`, and `pr_curves.png`
  (Cross-Balance highlighted, the strongest baseline highlighted, the other 9
  folded into one muted "Other baselines" line — see the dataviz skill for why).
  Note: the PR curve legend's AP is computed on **pooled** out-of-fold
  predictions, while `results_summary.csv`'s `aucpr_mean` is the **mean of
  per-fold** AUCPR — these are both valid but not numerically identical, and can
  diverge noticeably on small datasets (see `article/draft2.md` §5.4/§7 for a
  worked example on the 1,994-row Communities & Crime dataset).
- `run_benchmark.py` — CLI entry point tying the above together with
  `StratifiedKFold`; every (dataset × method × fold) exception is caught and
  logged rather than crashing the run.
- `tune_thresholds.py` — independent threshold search for
  `auto_density_thresholds` (see below).
- `tune_ratio.py` — same held-out tuning suite and constants (imported from
  `tune_thresholds.py`), sweeping `ratio` instead with thresholds held fixed at
  their tuned value. Confirmed `ratio=2.0` (used unchanged since the original
  prototype, never previously tested) was already the best choice.

### Iteration workflow (`results/`)

Algorithm development here proceeds as numbered iterations, each changing one
aspect of `CrossBalanceClassifier`'s config in `methods.py`'s registry, then
re-running the full benchmark with `--out-root results/iteration_N`. `git log`
is the record of what each iteration changed and why — read recent commits
before assuming what the "current" configuration is or should be.
`results/phase_synthetic/` and `results/phase_real/` are the very first
(pre-iteration) baseline run. An iteration is only adopted as the library's
shipped default if it's a validated net improvement over the prior best; negative
results (an iteration that regresses more than it improves) are still committed
and reported, not discarded — see iteration 6 in `article/draft2.md` for the
canonical example, and iteration 7 for how its regression was partly resolved by
tuning thresholds independently rather than reverting the idea outright.

`results/iteration_7_fullsize/` holds the extended-validation reruns of the
shipped (iteration 7) config, unchanged, at true full scale — the actual
568,630 / 1,852,394 / 6,362,620 rows for the three largest fraud datasets,
not the 50,000-row `stratified_subsample()` used everywhere else. This is where
Cross-Balance's win over Balanced Random Forest reverses on PaySim (see
`max_chunk_size` above) — a limitation invisible in every other results
directory, which all use the subsampled comparison.

### Tests (`tests/`)

`test_classifier.py` includes a regression-guard test
(`test_defaults_match_validated_config`) that pins `CrossBalanceClassifier()`'s
default parameters to the currently-best-validated configuration. If you change
what the library ships as defaults, update this test deliberately — a failure
here means either the change was unintentional or the test needs updating to
reflect a newly-validated config, not that the test is wrong.

### The article (`article/`)

- `draft.md` is the original full draft (Phase C + extended validation, prose and
  tables only). `draft2.md` is a from-scratch rewrite with the same verified
  numbers plus diagrams/graphics woven in — a Mermaid flowchart of the fit-time
  pipeline (§3), and four matplotlib figures generated by `make_figures.py`
  (iteration-progression line chart, ranked 11-method bar chart, full-scale-vs-
  subsampled grouped bars, density-tier-ladder diagram), on top of the three
  `pr_curves.png` figures `benchmarks/report.py` already produces per dataset.
  Treat `draft.md` as superseded prose-only history, `draft2.md` as current.
- `make_figures.py` reuses the exact palette constants from
  `benchmarks/report.py::plot_pr_curves` (`_BLUE`/`_RED`/`_OTHER_GRAY`/ink and
  gridline tokens) so the new charts match the existing PR-curve figures. Every
  number plotted is copied from the `results/` CSVs it cites in-line, not
  retyped from memory — verify against source before editing.
- `Cross-Balance_draft2.docx` is a pandoc export of `draft2.md` for readers who
  want Word rather than Markdown. Pandoc can't render the Mermaid fence, so the
  actual conversion source is `draft2_docx_src.md` (same content, Mermaid block
  replaced by `figures/architecture.png`) — regenerate that swap after editing
  `draft2.md`'s method section before re-exporting (see Commands above).

## Data sources

Real-world datasets are not committed (`data/raw/` is gitignored). See
`DATA_SOURCES.md` for exact download links, expected file layout per dataset, and
documented schema caveats (e.g. `credit_card_fraud` ended up being a different
Kaggle dataset than originally planned, and its loader was written for the
schema that's actually there). `REFERENCES.md` tracks external reading material
for the article, separate from the ~239-paper corpus in `references/`
(gitignored, copied from the archived v1 project).
