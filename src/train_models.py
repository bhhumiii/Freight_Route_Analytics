"""
train_models.py
Trains three regressors (XGBoost, LightGBM, CatBoost) to predict
circuit_speed (km/h) for a route, evaluates them on a held-out
validation split with a full metric suite, and persists everything
(models, encoders, metrics, validation predictions/residuals) into
models/models_bundle.pkl for the dashboard.
"""
import os, pickle, warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, r2_score, mean_squared_error,
    mean_squared_log_error,
)
from sklearn.preprocessing import LabelEncoder
import xgboost  as xgb
import lightgbm as lgb
from catboost  import CatBoostRegressor

warnings.filterwarnings("ignore")

import config as C

PROC_PKL   = C.PROC_PKL
MODELS_DIR = C.MODELS_DIR
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_COLS = C.FEATURE_COLS
CAT_COLS     = C.CAT_COLS
TARGET       = C.TARGET


def load_data(max_rows=200_000):
    print("Loading processed data ...")
    with open(PROC_PKL, "rb") as f:
        df = pickle.load(f)

    num_cols = [c for c in FEATURE_COLS if c not in CAT_COLS]
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    if len(df) > max_rows:

        df = (
            df.sample(frac=1, random_state=42).head(max_rows)
        )
    print(f"  Using {len(df):,} rows")
    return df


def prep_xy(df, encoders=None):
    X = df[FEATURE_COLS].copy()
    fit_mode = encoders is None
    if fit_mode:
        encoders = {}

    for c in CAT_COLS:
        X[c] = X[c].astype(str)
        if fit_mode:
            le = LabelEncoder()
            X[c] = le.fit_transform(X[c])
            encoders[c] = le
        else:
            le = encoders[c]
            known = set(le.classes_)
            X[c] = X[c].apply(lambda v: v if v in known else le.classes_[0])
            X[c] = le.transform(X[c])

    y = df[TARGET].values
    return X.astype(np.float32), y, encoders


def compute_metrics(name, y_true, y_pred):
    """Full regression metric suite: MAE, MSE, RMSE, RMSLE, R²."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2   = r2_score(y_true, y_pred)
    # RMSLE needs non-negative inputs; clip predictions at 0 to be safe.
    rmsle = float(np.sqrt(mean_squared_log_error(
        np.clip(y_true, 0, None), np.clip(y_pred, 0, None))))
    print(f"  [{name}]  MAE={mae:.4f}  MSE={mse:.4f}  RMSE={rmse:.4f}  "
          f"RMSLE={rmsle:.4f}  R2={r2:.4f}")
    return {"model": name, "MAE": mae, "MSE": mse, "RMSE": rmse,
            "RMSLE": rmsle, "R2": r2}


def _build_model(name):

    if name == "XGBoost":
        return xgb.XGBRegressor(

            n_estimators=700,

            max_depth=8,

            learning_rate=0.03,

            subsample=0.9,

            colsample_bytree=0.9,

            gamma=0.1,

            min_child_weight=2,

            reg_alpha=0.1,

            reg_lambda=2,

            tree_method="hist",

            random_state=42,

            eval_metric="rmse",

            early_stopping_rounds=50,

            verbosity=0
        )

    if name == "LightGBM":
        return lgb.LGBMRegressor(
            n_estimators=700,

            learning_rate=0.03,

            num_leaves=64,

            max_depth=8,

            subsample=0.9,

            colsample_bytree=0.9,

            reg_alpha=0.1,

            reg_lambda=2,

            random_state=42,

            verbose=-1
        )

    if name == "CatBoost":
        return CatBoostRegressor(
            iterations=700,
            depth=8,
            learning_rate=0.03,
            l2_leaf_reg=5,
            loss_function="RMSE",
            random_seed=42,
            early_stopping_rounds=50,
            verbose=0
        )


def _fit(name, model, X_tr, y_tr, X_val, y_val):
    """Unified fit with model-specific early-stopping wiring."""
    print(f"Training {name} ...")
    if name == "XGBoost":
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    elif name == "LightGBM":
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False),
            lgb.log_evaluation(-1)])
    else:  # CatBoost
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)
    return model


def run():
    df             = load_data()
    X, y, encoders = prep_xy(df)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2,shuffle=True, random_state=42)
    print(f"Train: {len(X_tr):,}  |  Val: {len(X_val):,}\n")

    models, results, val_preds = {}, [], {}
    for name in ["XGBoost", "LightGBM", "CatBoost"]:

        model = _fit(name, _build_model(name), X_tr, y_tr, X_val, y_val)

    # ===== Feature Importance =====
        if name == "XGBoost":

            importance= pd.Series(
            model.feature_importances_,
            index=X.columns
            )

            print("\nTop 20 Most Important Features\n")

            print(
                importance.sort_values(
                ascending=False
                ).head(20)
            )

        preds = np.asarray(model.predict(X_val), dtype=float)

        models[name.lower()] = model

        val_preds[name] = preds

        results.append(
            compute_metrics(
                name,
                y_val,
                preds
            )
        )

    results = pd.DataFrame(results)
    print("\n=== Model Comparison ===")
    print(results.to_string(index=False))

    best_name = results.loc[results["MAE"].idxmin(), "model"]
    print(f"\nBest model by MAE: {best_name}")

    # Persist a capped validation sample for residual / pred-vs-actual plots
    # in the dashboard (keep it small so the bundle stays light).
    n_keep = min(5000, len(y_val))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(y_val), n_keep, replace=False)
    validation = {
        "y_true": np.asarray(y_val, dtype=float)[idx],
        "y_pred": {name: val_preds[name][idx] for name in val_preds},
    }

    bundle = {
        "xgboost":  models["xgboost"],
        "lightgbm": models["lightgbm"],
        "catboost": models["catboost"],
        "encoders": encoders,
        "feature_cols": FEATURE_COLS,
        "cat_cols": CAT_COLS,
        "results": results,
        "best_model": best_name,
        "validation": validation,
    }
    with open(os.path.join(MODELS_DIR, "models_bundle.pkl"), "wb") as f:
        pickle.dump(bundle, f, protocol=5)
    print("Saved models/models_bundle.pkl")
    return bundle


if __name__ == "__main__":
    run()
