"""Types of data used in Dataset"""
from enum import IntEnum


class DataType(IntEnum):
    """Types of data used in Dataset"""
    CATEGORICAL = 0
    TEXT = 1
    SHORT_TEXT = 2
    NUMERIC = 3
    DATE = 4
