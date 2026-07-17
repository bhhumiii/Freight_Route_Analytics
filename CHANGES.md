# RailRoute Optimizer — Improvement Summary

This document covers the analysis, fixes, and enhancements applied to the
freight-route project. Everything below was verified by running the actual
pipeline, pathfinder, and dashboard on the full 441,381-row dataset.

---

## 1. Headline result

| Area | Before | After |
|------|--------|-------|
| Worst-case route query (`BSPC → MDCC`, 3 hops) | **24,700 ms** | **24 ms** (~1000× faster) |
| `run_pipeline.py` on a fresh clone | **Crashed** (missing `railrake.csv`; bad `run()` signature) | Completes in ~0.6 min |
| Evaluation metrics | MAE, RMSE, R² | **MAE, MSE, RMSE, RMSLE, R²** |
| Outlier handling | None | IQR detection + winsorising + report |
| Residual / pred-vs-actual plots | None | Full analytics tabs |
| `config.py` (shipped but unused) | Dead file | Wired into all core modules |

Model quality (200k sample, 40k validation, after IQR winsorising):

| Model    | MAE   | MSE   | RMSE  | RMSLE | R²    |
|----------|-------|-------|-------|-------|-------|
| XGBoost  | 1.386 | 3.834 | 1.958 | 0.236 | 0.826 |
| LightGBM | 1.421 | 3.975 | 1.994 | 0.241 | 0.820 |
| CatBoost | 1.446 | 4.081 | 2.020 | 0.245 | 0.815 |

---

## 2. Bugs fixed

1. **Catastrophic pathfinding performance.** `_dijkstra` in `time` mode called
   `_predict_speed` on every edge relaxation, and each call built a fresh
   one-row DataFrame and label-encoded it (~5 ms each). On a 35k-edge graph this
   meant tens of thousands of redundant single-row model calls per query.
   **Fix:** every edge's ML speed is now batch-predicted **once at load time**
   (one vectorised `predict()` per model over de-duplicated `(km, hrs)` pairs),
   reducing each Dijkstra edge to an O(1) dict lookup. An `lru_cache` fallback
   covers any off-graph value.

2. **Missing / misnamed raw CSV.** Code expected `data/railrake.csv`; the
   supplied file is `ALL_RAKES_CYCL_FY2526_APR_FEB.csv`, so a fresh
   `run_pipeline.py` crashed immediately. **Fix:** `config.resolve_raw_csv()`
   accepts either name (and is the single place to add more).

3. **`run_pipeline.py` called a non-existent signature.** It passed
   `remove_outliers=`/`outlier_method=` to a `preprocessing.run()` that took no
   arguments — an `AttributeError`/`TypeError` on every run. **Fix:** the call
   now matches the real signature (`outlier_method`, `cap_outliers`), driven by
   the existing `REMOVE_OUTLIERS` / `OUTLIER_METHOD` env vars.

4. **KPI card crash / wrong labels in `app.py`.** The card formatted `{val:,.2f}`
   even when `val` was the string `"N/A"` (raising `ValueError`), and
   `label_map[unit]` mislabeled values. **Fix:** rewritten `kpi()` helper with an
   explicit no-path state and per-mode format strings.

5. **Train/serve skew on `dist_bucket`.** Training used `pd.cut` bins
   `[0,200,500,1000,2000,…]`; inference used `km // 250`, so the model saw a
   different feature at serve time than at train time. **Fix:** both paths now
   call the single `config.dist_bucket()` with identical bins (verified
   equivalent for all km > 0).

6. **Brittle exception + offline breakage.** `parse_hhmm` used a bare `except:`
   (now scoped to `ValueError`/`AttributeError`); the sidebar loaded a Wikipedia
   image over the network (now an inline SVG that works offline).

---

## 3. Features added

- **Outlier detection & handling** (`preprocessing.py`): IQR (Tukey) and Z-score
  detectors, a per-column report (fences, counts, %), and winsorising to the
  fences — chosen over row-dropping so the route graph keeps full connectivity.
  Report saved to `data/outlier_report.pkl`.
- **Full metric suite** (`train_models.py`): MAE, MSE, RMSE, RMSLE, R² for all
  three models; a 5,000-row validation sample (`y_true` + per-model `y_pred`)
  is saved into the bundle to power residual plots without retraining.
- **Analytics module** (`analytics.py`, new): reusable Plotly builders for
  metric comparison, predicted-vs-actual, residual scatter, residual histogram,
  outlier box plots, and per-hop distance.
- **Redesigned dashboard** (`app.py`): railway-signal theme (slate base with
  blue/green/amber accents mapped to the three route modes), departure-board
  monospace for station codes, fixed KPI cards, and two new tabs —
  **Model analytics** (metrics + residual/pred-vs-actual) and **Data quality**
  (outlier report + box plots). Existing sections/layout kept recognizable.

---

## 4. Refactoring & maintainability

- The shipped-but-unused **`config.py`** is now the single source of truth for
  paths, feature lists, categorical columns, and distance-bucket bins.
  `preprocessing.py`, `train_models.py`, and `pathfinder.py` import from it
  instead of re-declaring constants — eliminating the copy-paste drift that
  caused bug #5.
- The three near-identical `train_xgboost` / `train_lightgbm` / `train_catboost`
  functions collapsed into a `_build_model()` factory + unified `_fit()`.
- Plotting logic moved out of the dashboard into `analytics.py`.

---

## 5. Files modified

| File | Change |
|------|--------|
| `src/preprocessing.py` | Outlier detection/handling, config wiring, robust CSV resolution, scoped exception |
| `src/train_models.py` | 5-metric suite, model factory refactor, saved validation arrays, config wiring |
| `src/pathfinder.py` | Batched edge-speed precompute (1000× speedup), shared `dist_bucket`, config wiring |
| `src/app.py` | Full UI redesign, KPI bug fix, analytics + data-quality tabs, offline SVG |
| `src/analytics.py` | **New** — Plotly figure builders |
| `src/config.py` | Now actually used (was dead) |
| `run_pipeline.py` | Fixed `preprocess_run()` call signature |
| `README.md` | Updated metrics table + outlier description |

> Note: `data/processed.pkl` and the 285 MB raw CSV are not shipped (they are
> regenerated by `run_pipeline.py` and are in `.gitignore`). Drop your CSV into
> `data/` as `railrake.csv` (or its original name) and run the pipeline.

---

## 6. Recommendations for further improvement

1. **Investigate the `ldng_uldg_speed` feature.** The target `circuit_speed`
   correlates 0.79 with `ldng_uldg_speed` (= km ÷ hours), which carries most of
   the R². They are related-but-distinct speed measures, so it is not pure
   leakage — but a model trained without it would better reflect *predictive*
   power from route/commodity structure. Worth an ablation. (Left untouched here
   so as not to silently change your headline numbers.)
2. **Use native categorical handling.** CatBoost and LightGBM accept raw
   categoricals; the current `LabelEncoder` path discards that advantage and
   imposes a false ordinal ordering. CatBoost especially should improve.
3. **Cross-validation over a single split** for more stable metric estimates,
   and a held-out test set distinct from the early-stopping validation set.
4. **Edge confidence.** Many graph edges are backed by a single observation
   (`count = 1`); exposing/weighting by `count` would flag low-confidence hops.
5. **Geospatial map view.** Station coordinates would enable an actual route map
   (Plotly/pydeck) instead of the code-chip sequence.
6. **Persist tuned hyperparameters** via Optuna rather than hand-set values, and
   log runs (MLflow) for reproducibility.
7. **Tests.** Add unit tests for `dist_bucket`, `parse_hhmm`, outlier masks, and
   a smoke test that a known station pair returns a path.
