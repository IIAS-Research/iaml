import hashlib
import pickle
from typing import Any

import numpy as np
import pandas as pd

def hash_df(df: pd.DataFrame) -> str:
    h = hashlib.sha256()

    vals = pd.util.hash_pandas_object(df, index=False, categorize=True).values.tobytes()
    h.update(vals)

    h.update(pd.util.hash_pandas_object(df.index, categorize=True).values.tobytes())

    h.update(pd.util.hash_pandas_object(df.columns, categorize=True).values.tobytes())

    dtype_idx = pd.Index([getattr(dt, "name", str(dt)) for dt in df.dtypes])
    h.update(pd.util.hash_pandas_object(dtype_idx).values.tobytes())

    return h.hexdigest()


def hash_dataset(
    X: pd.DataFrame,
    y: Any,
    groups: pd.DataFrame | None = None,
    columns_types: dict | None = None,
    target_type: str | None = None,
) -> str:
    """Hash all inputs that can affect a supervised step or evaluation.

    Targets are positional, as in Dataset, so their pandas index is not used.
    Serialization preserves their shape and dtype, including structured survival
    targets. This only serializes local inputs. No pickle is loaded here.
    """
    payload = (
        "iaml-dataset-v2",
        hash_df(X),
        np.asarray(y),
        hash_df(groups) if groups is not None else None,
        columns_types,
        target_type,
    )
    return hashlib.sha256(pickle.dumps(payload, protocol=5)).hexdigest()


def hash_evaluation_context(*values: Any) -> str | None:
    """Identify serializable evaluation settings, or decline to cache them.

    Partial splitters include their arguments. Local functions and lambdas that
    cannot be serialized are evaluated without partition or score caching.
    """
    try:
        return hashlib.sha256(pickle.dumps(values, protocol=5)).hexdigest()
    except (pickle.PicklingError, TypeError, AttributeError, ValueError):
        return None
