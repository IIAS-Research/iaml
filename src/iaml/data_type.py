"""Types of data used in Dataset"""
from enum import Enum


class DataType(Enum):
    """Types of data used in Dataset"""
    CATEGORICAL = 0
    TEXT = 1
    SHORT_TEXT = 2
    NUMERIC = 3
    DATE = 4
