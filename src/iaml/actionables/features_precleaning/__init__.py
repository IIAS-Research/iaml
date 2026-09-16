"""
Usualy first step of a pipeline, clean/transform features.
Example : Convert SHORT_TEXT column into datetime if possible
"""
from .act_date_converter import ActDateConverter
from .act_trim_space import ActTrimSpaces
from .act_normalize_column_names import ActNormalizeColumnNames
from .act_coerce_numeric_strings import ActCoerceNumericStrings
from .act_drop_high_missing_columns import ActDropHighMissingColumns
from .act_drop_duplicate_rows import ActDropDuplicateRows
from .act_sentinel_to_na_n import ActSentinelToNaN
from .act_drop_id_like_columns import ActDropIdLikeColumns
