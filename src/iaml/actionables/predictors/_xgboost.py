"""Feature-name adaptation shared by the XGBoost predictors."""
import pandas as pd


_FEATURE_NAME_ESCAPES = str.maketrans({"%": "%25", "[": "%5B", "]": "%5D", "<": "%3C"})


def xgboost_features(X):
    """Escape names without collisions, preserving data and native name validation."""
    if not isinstance(X, pd.DataFrame):
        return X

    renamed = X.copy(deep=False)
    # Escaping '%' also distinguishes a literal escape sequence from its source.
    renamed.rename(columns=lambda name: str(name).translate(_FEATURE_NAME_ESCAPES), inplace=True)
    return renamed
