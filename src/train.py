"""Split, compare models with cross-validation, then evaluate the best one on the test set once.

Usage:
    python scripts/train.py data/kc_house_data.csv
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_validate, train_test_split

from house_prices import build_model_pipeline, remove_bad_data

SCORING = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}


def candidate_models():
    return {
        "ridge":               Ridge(alpha=1.0),
        # Tunes alpha and l1_ratio with its own 5-fold CV, inside each training fold
        "elastic_net":         ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8, 1.0],
                                            alphas=[0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03],
                                            cv=5, max_iter=5000, n_jobs=-1),
        "random_forest":       RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=42),
        "hist_gradient_boost": HistGradientBoostingRegressor(random_state=42),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", help="raw kc_house_data CSV")
    parser.add_argument("--model-out", default="models/house_price_model.joblib")
    args = parser.parse_args()

    # 1. Remove bad rows, then split
    df = remove_bad_data(pd.read_csv(args.data))
    X, y = df.drop(columns="price"), df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"{len(df)} rows after cleaning → train {len(X_train)}, test {len(X_test)}\n")

    # 2. Compare models with 5-fold CV on TRAIN only (same folds for every model)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    models = candidate_models()
    rows = []
    for name, regressor in models.items():
        print(f"cross-validating {name} ...")
        s = cross_validate(build_model_pipeline(regressor), X_train, y_train, cv=cv, scoring=SCORING)
        rows.append({
            "model": name,
            "cv_rmse": -s["test_rmse"].mean(), "rmse_std": s["test_rmse"].std(),
            "cv_mae": -s["test_mae"].mean(),
            "cv_r2": s["test_r2"].mean(), "r2_std": s["test_r2"].std(),
            "fit_time_s": s["fit_time"].mean(),
        })
    results = pd.DataFrame(rows).sort_values("cv_rmse").reset_index(drop=True)
    with pd.option_context("display.float_format", "{:,.3f}".format, "display.width", 120):
        print("\n" + results.to_string(index=False))

    # 3. Fit the best model on all training data and evaluate on TEST once
    best = results.loc[0, "model"]
    final_model = build_model_pipeline(models[best]).fit(X_train, y_train)
    pred = final_model.predict(X_test)
    print(f"\nBest model: {best}")
    fitted = final_model["model"].regressor_
    if hasattr(fitted, "alpha_"):
        print(f"Tuned ElasticNet: alpha={fitted.alpha_}, l1_ratio={fitted.l1_ratio_}")
    print(f"Test RMSE: {root_mean_squared_error(y_test, pred):,.0f}")
    print(f"Test MAE:  {mean_absolute_error(y_test, pred):,.0f}")
    print(f"Test R²:   {r2_score(y_test, pred):.3f}")

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, args.model_out)
    print(f"\nSaved model to {args.model_out}")


if __name__ == "__main__":
    main()
