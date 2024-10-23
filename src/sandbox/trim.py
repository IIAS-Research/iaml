import pandas as pd
from pandas.api.types import is_object_dtype
data = {
  " stringos  ": ['1', '     ezr', 'aezr zer ze    ', ''],
  "intos": [1 ,2 ,3, 4],
  "f loatos": [1.0 ,2.2 ,3.5, 4.0],
  "calories": ['1', 2, True , '   '],
  "durat ion": ['    1 ', 2, False, '      ']
}


df = pd.DataFrame(data)

print(df.head())
print(df.info(verbose=True))

df = df.rename(columns=lambda x: x.strip())
for col in df.columns:
    if isinstance(df[col].dtype, str) or is_object_dtype(df[col]):
        print(f"Triming {col=}")
        df[col] = df[col].astype(str)
        df[col] = df[col].str.strip()
        # df[col] = df[col].astype(str).str.strip()
        # print(df[col].str.strip())


print(df.head())
for col in df.columns:
    print(f"{col=} {df[col].isnull().values.any()}")