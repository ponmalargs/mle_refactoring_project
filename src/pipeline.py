import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import TransformedTargetRegressor
from .cleaning_functions import fix_date_format, fix_str_format
from .features import (
    WaterfrontDistance,
    calc_last_known_change,
    calc_sqft_basement,
    calc_wealth_center_distance,
)
from .preprocessing import build_preprocessor

def build_cleaning_pipeline():
    pipeline = Pipeline([
        ('fix_date_format', FunctionTransformer(fix_date_format, kw_args={'date_col': ['date']})),
        ('fix_str_format', FunctionTransformer(fix_str_format, kw_args={'str_col': ['zipcode']})),
        ('calc_sqft_basement', FunctionTransformer(calc_sqft_basement)),
        ('calc_last_known_change', FunctionTransformer(calc_last_known_change))
    ])
    return pipeline

def build_feature_pipeline():
    pipeline = Pipeline([
        ('wealth_center_distance', FunctionTransformer(calc_wealth_center_distance)),
        ('waterfront_distance', WaterfrontDistance())
    ])
    return pipeline

def build_model_pipeline(regressor):
    return Pipeline([
        ("cleaning",   build_cleaning_pipeline()),
        ("features",   build_feature_pipeline()),
        ("preprocess", build_preprocessor()),
        ("model",      TransformedTargetRegressor(regressor=regressor,
                                                  func=np.log1p, inverse_func=np.expm1)),
    ])