# House Prices Project: Work Summary

## What was built

A Python package, `house-prices`, that predicts King County house prices from raw data, with one pipeline from start to finish:

```
remove_bad_data          ← before the split (drops rows)
train_test_split (80/20)
model pipeline
 ├─ cleaning     fix_date, calc_sqft_basement, calc_last_known_change
 ├─ features     WealthCenterDistance, WaterfrontDistance
 ├─ preprocess   ColumnTransformer: view/waterfront → most frequent,
 │               numbers → median + scale, zipcode → one-hot (or ordinal for trees)
 └─ model        TransformedTargetRegressor (log price) + regressor
```

## Key decisions

- **Rows vs. columns:** steps that remove rows run before the split. Steps that change columns go inside the pipeline.
- **Fixed rules vs. learned steps:** fixed formulas use `FunctionTransformer` or simple classes. Anything that learns from data (medians, scaling, waterfront locations, zipcode averages) is fitted on training data only, to avoid leakage.
- **`view` / `waterfront`:** filled with the most frequent value inside the `ColumnTransformer`.
- **New features:**
  - `wealth_center_distance`: distance to a fixed centre of wealth (47.62774, -122.24194)
  - `waterfront_distance`: distance to the nearest waterfront house, learned from training data
  - `zipcode_price_per_sqft` (optional, not used): target encoding with cross-fitting and smoothing
- **Target:** log(price) via `TransformedTargetRegressor`, wrapping only the model step so the feature steps still see the raw price.
- **Model selection:** 5-fold cross-validation on training data with the same folds for every model. The test set is used once, at the end.

## Results

- Compared Ridge, ElasticNet (`ElasticNetCV`, self-tuning), Random Forest and HistGradientBoosting. **HistGradientBoosting performed best.**
- A tree-specific pipeline (no scaling, zipcode as a native category via `OrdinalEncoder`) gave the **same score** as one-hot encoding. Location is already captured by `lat`/`long` and the distance features, so the simpler and faster native version is preferred.

## Package contents

| File | Contents |
|---|---|
| `cleaning.py` | `remove_bad_data()`: duplicates, implausible rows (before the split) |
| `features.py` | `fix_date`, `calc_sqft_basement`, `calc_last_known_change` |
| `feature_engineering.py` | `WealthCenterDistance`, `WaterfrontDistance`, `ZipcodePricePerSqft`, `dist()` |
| `preprocessing.py` | `build_preprocessor()`, column lists `NUM_COLS`, `CAT_COLS`, `MODE_FILL_COLS` |
| `pipeline.py` | `build_cleaning_pipeline()`, `build_feature_pipeline()`, `build_model_pipeline()` |
| `train.py` | Split, compare models with CV, test once, save the model |


## Next steps

1. Tune HistGradientBoosting with `RandomizedSearchCV`.
2. Check which features matter (ablation or permutation importance).
3. Analyze errors by price band, by zipcode and for the biggest misses.
4. Do the final test evaluation, save the model, and add a `predict.py` script.
5. Create an API for the same
