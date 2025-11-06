import hashlib
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

