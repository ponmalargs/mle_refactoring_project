"""Preprocessing: steps that LEARN from the training data.
 
Runs after the cleaning pipeline and after train_test_split.
Any column not listed below (id, date, yr_renovated, price) is dropped.
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler ,OrdinalEncoder
import numpy as np
 
# Filled with the most frequent value, learned from the training data
MODE_FILL_COLS = ["view", "waterfront"]
 
NUM_COLS = [
    "bedrooms", "bathrooms", "floors", "condition", "grade",
    "sqft_living", "sqft_lot", "sqft_above", "sqft_basement",
    "sqft_living15", "sqft_lot15",
    "last_known_change",
    "center_distance",          # from build_feature_pipeline()
    "waterfront_distance",      # from build_feature_pipeline()
]
 
CAT_COLS = ["zipcode"]
 
 
def build_preprocessor(
    num_cols: list[str] = NUM_COLS,
    cat_cols: list[str] = CAT_COLS,
    mode_fill_cols: list[str] = MODE_FILL_COLS,
) -> ColumnTransformer:
    """ColumnTransformer: fill view/waterfront with the most frequent value, impute + scale numbers,
    one-hot encode categories."""
    preprocessor = ColumnTransformer(
        [
            ("mode_fill", SimpleImputer(strategy="most_frequent"), mode_fill_cols),
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale",  StandardScaler()),
            ]), num_cols),
 """           ("cat", Pipeline([
                #("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), cat_cols), """
             ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                                   encoded_missing_value=np.nan), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    # Return a DataFrame with column names instead of a NumPy array
    return preprocessor.set_output(transform="pandas")