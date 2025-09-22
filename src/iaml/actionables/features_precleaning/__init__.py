"""
Usualy first step of a pipeline, clean/transform features.
Example : Convert SHORT_TEXT column into datetime if possible
"""
from .act_date_converter import ActDateConverter
from .act_drop_bad_quality_rows import ActDropBadQualityRows
from .act_trim_space import ActTrimSpaces
