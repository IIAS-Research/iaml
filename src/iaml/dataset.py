"""Encapsulate X, y data to be used by Steps
Add features like data type detection and splitting 
"""
import copy
from copy import deepcopy
from typing import Any, Iterator, TYPE_CHECKING
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
import numpy as np
import pandas as pd

from .cache_keys import hash_df
from .data_type import DataType
from .type_of_target import type_of_target
from .logger import Logger


if TYPE_CHECKING:
    from .iaml_pipeline import IAMLPipeline


class Dataset:
    """Encapsulate X, y data to be used by Steps
    Add features like data type detection and splitting
    
    :param pd.DataFrame X: The dataframe used in this Dataset.
    :param list, optional y: The dataframe target. Default to None.
    :param pd.DataFrame, optional groups: If set, will be used when splitting to avoid
        having data rows for a similar ID in two different splits. Default to None.
    :param list[str], optional groups_columns: Same as groups, but with column names. 
        Default to None.
    :param dict, optional columns_types: Specify column types instead of detecting them 
        automaticaly. Default to None.
    
    """

    def __init__(self,
        X: pd.DataFrame,
        y: list = None,
        groups: pd.DataFrame = None,
        groups_columns: list[str] = None,
        columns_types: dict = None
    ):
        if groups is not None and groups_columns:
            raise ValueError("groups and groups_columns are not None. Only one must be set")

        if groups is not None and set(groups.columns).intersection(X.columns):
            raise ValueError("Group columns present in dataset!")

        if groups_columns is None:
            groups_columns = []

        self.__X: pd.DataFrame = X.drop(columns=groups_columns)
        """The dataframe used in this Dataset object without groups columns if provided"""

        self._fingerprint: str | None = None
        """Cached hash of X for step caching."""

        # Make sure y is either None or single column
        if isinstance(y, pd.DataFrame):
            if len(y.columns) > 1:
                raise ValueError("IAML only handle single column labels !")
            y = y.values.ravel()
        self.__y: np.array = np.array(y)
        """The dataframe target"""

        if groups_columns:
            self.groups = X[groups_columns]
        else:
            if type(groups) in [pd.Series, list, np.array]:
                self.groups = pd.DataFrame(groups)
            else:
                self.groups = groups

        if self.groups is not None and self.groups.shape[1] > 1:
            # Create a combined group label by concatenating all columns into tuples
            Logger().warning("You are using multiple columns as groups. \
                Be careful, as these columns will serve as a composite key.")
            self.groups = pd.DataFrame(pd.Series(
                list(zip(*[self.groups[col] for col in self.groups.columns]))),
                columns=['groups']
                )

        self.columns_types: dict = columns_types if columns_types else {}
        """Columns types to be applied to our dataframe columns"""

        self.__detect_columns_types()

        self.type_of_target: str = None
        """Type of target to predict"""

        if y is not None:
            self.type_of_target: str = type_of_target(self.__y)

    @property
    def features(self) -> list[str]:
        """List columns names of X data

        :return: columns names.
        """
        return self.X.columns.to_list()

    @property
    def X(self) -> pd.DataFrame:
        """X data getter

        :return: X data
        """
        return self.__X

    @property
    def y(self) -> np.ndarray:
        """y data getter

        :return: y data
        """
        return self.__y

    def copy(self, deep: bool = True) -> 'Dataset':
        """Copy Dataset into a new instance

        :param bool, optional deep: Perform a deep copy. Defaults to True.
        :return: Copied Dataset.
        """
        if deep:
            return copy.deepcopy(self)
        return copy.copy(self)

    def decline(self, X: pd.DataFrame, y: list, groups: pd.DataFrame = None) -> 'Dataset':
        """Create a new Dataset with columns_types based on self.
        Avoid time consuming columns_types computing
        
        :param pd.DataFrame X: The dataframe to use for our newly created Dataset.
        :param list y: The target to use for our newly created Dataset.
        :param pd.DataFrame, optional groups: The groups to use for our newly created Dataset.
        :return: New Dataset.
        """
        if groups is None:
            groups = self.groups
        return Dataset(X, y, groups=groups, columns_types=self.columns_types)

    def sample(self, n: int) -> 'Dataset':
        """Return a dataset with a sample of data

        :param int n: Number of line in the sample dataset
        :return: dataset with a sample of data
        """
        if isinstance(n, float):
            n = int(self.X.shape[0]*n)

        if n >= self.X.shape[0]:
            return self.decline(self.X, self.y)

        if self.type_of_target == 'continuous':
            _, test_idx = next(
                ShuffleSplit(n_splits=1, test_size=n, random_state=42
                ).split(self.X, self.y))
        else:
            try:
                _, test_idx = next(
                    StratifiedShuffleSplit(n_splits=1, test_size=n, random_state=42
                    ).split(self.X, self.y))
            except ValueError:
                _, test_idx = next(
                    ShuffleSplit(n_splits=1, test_size=n, random_state=42
                    ).split(self.X, self.y))

        return self.decline(self.X.iloc[test_idx], self.y[test_idx])

    def transform(self, method: callable) -> None:
        """Apply transform method to X or y data based on the method signature

        :param callable method: Callable to apply. Will be call with X or y as parameter.
        :return: Transformed dataset.
        """
        self.__X = method(self.__X)
        self._fingerprint = None
        self.__detect_columns_types()

    @property
    def has_groups(self) -> bool:
        """Groups exists ?
        
        :return: Exists ?
        """
        return self.groups is not None and not self.groups.empty

    def resample(self, resampler: callable) -> 'Dataset':
        """Apply a resampler on X, y and groups data. 

        :param callable resampler: Resampler method.
        :return: Resampled Dataset.
        """

        if self.has_groups:
            # Merge groups with X
            X = self.X.reset_index(drop=True).join(self.groups.reset_index(drop=True))

            # Resampler
            X, y = resampler(X, self.y)

            # Split groups and X
            return self.decline(X.drop(columns=self.groups.columns),
                            y,
                            groups=X[self.groups.columns])
        return self.decline(*resampler(self.X, self.y))

    def fingerprint(self) -> str:
        """Stable hash of X for caching."""
        if self._fingerprint is None:
            self._fingerprint = hash_df(self.__X)
        return self._fingerprint

    def split(self, splitter: callable, *args, **kwargs) -> Iterator[tuple['Dataset', 'Dataset']]:
        """Use splitter to split dataset into a list of tuple (train set, test set) 

        :param callable splitter: The splitter function to perform.
        :param tuple, optional \\*args: Additional parameters.
        :param dict, optional \\**kwargs: Additional parameters.
        :return: Train set and Test set iterator.
        """
        # Split the dataset as many times as the splitter requires it
        y = self.y if self.type_of_target is not None else None
        for i_train, i_test in splitter(self.X, y, *args, **kwargs):
            X_train = self.X.iloc[i_train].copy()
            X_test = self.X.iloc[i_test].copy()

            if self.has_groups:
                groups = self.groups.iloc[i_train].copy()
            else:
                groups = None

            if y is not None:
                y_train = self.__y[i_train].copy()
                y_test = self.__y[i_test].copy()
            else:
                y_train = None
                y_test = None

            yield (self.decline(X_train, y_train, groups=groups),
                   self.decline(X_test, y_test))

    def x_with_groups(self) -> pd.DataFrame:
        """Return X dataframe with groups columns if not None. Return X otherwise.
        
        :return: The X dataframe with or without the groups columns.
        """
        if self.has_groups:
            return self.X.reset_index(drop=True).join(self.groups.reset_index(drop=True))
        return self.X

    def get_columns_names_by_type(self, types: list[DataType]) -> list[str]:
        """Get names of all the columns with DataType in types

        :param list[DataType] types: List of Datatype to search.
        :return: columns names.
        """
        if not isinstance(types, list):
            types = [types]

        return [
            column
            for column, (dtype, type) in self.columns_types.items()
            if type in types
        ]

    def __detect_data_type(self, column_name: str) -> DataType:
        """Detect data type of a column

        :param str column_name: Name of the column to analyse.
        :return  Type of the columns.
        """
        column_value = self.X[column_name]
        detected: DataType = None
        if column_value.dtype == object:
            if (len(column_value.unique()) / len(column_value) < 0.05 \
                or len(column_value.unique()) < 7):
                detected = DataType.CATEGORICAL
            elif column_value.astype(str).apply(len).max() <= 85:
                detected = DataType.SHORT_TEXT
            else:
                detected = DataType.TEXT
        elif np.issubdtype(column_value.dtype, np.number):
            detected = DataType.NUMERIC
        elif np.issubdtype(column_value.dtype, np.datetime64):
            detected = DataType.DATE

        return column_value.dtype, detected

    @property
    def needed_estimator(self) -> str:
        """Kind of estimator needed for this dataset
        
        :return: Estimator type.
        """
        if self.type_of_target == 'continuous':
            return 'regressor'

        if self.type_of_target == 'survival':
            return 'survival'

        return 'classifier'

    def __detect_columns_types(self) -> None:
        """Detect column type of all features in X"""
        new_types = {}
        for column in self.features:
            if column not in self.columns_types \
                or self.columns_types[column][0] != self.X[column].dtype:
                new_types[column] = self.__detect_data_type(column)
            else:
                new_types[column] = self.columns_types[column]

        self.columns_types = new_types

    def to_survival(self) -> None:
        """Turn dataframe to survival compatibility"""
        return Dataset.fix_survival(self.X, self.y)

    @staticmethod
    def _normalize_survival_pair(value: Any) -> tuple[bool, float]:
        """Normalize a single survival sample to a (event, time) tuple."""
        if isinstance(value, np.void):
            if value.dtype.names and \
                'event' in value.dtype.names and 'time' in value.dtype.names:
                return bool(value['event']), float(value['time'])
            value = value.tolist()

        if isinstance(value, dict):
            if 'event' not in value or 'time' not in value:
                raise KeyError("Survival sample dictionary must include 'event' and 'time'.")
            return bool(value['event']), float(value['time'])

        if isinstance(value, np.ndarray):
            if value.shape == ():
                return Dataset._normalize_survival_pair(value.item())
            if value.ndim >= 1 and value.shape[0] >= 2:
                return bool(value[0]), float(value[1])

        if isinstance(value, (tuple, list)):
            if len(value) < 2:
                raise ValueError("Survival sample must provide event indicator and time.")
            return bool(value[0]), float(value[1])

        raise TypeError(f"Unsupported survival sample format: {type(value)}")

    @classmethod
    def normalize_survival_target(cls, y: Any) -> list[tuple[bool, float]]:
        """Return survival targets as a list of (event, time) tuples."""
        if y is None:
            return []

        if isinstance(y, pd.DataFrame):
            if not len(y.columns):
                return []
            if {'event', 'time'}.issubset(y.columns):
                iterator = zip(y['event'], y['time'])
            elif len(y.columns) >= 2:
                iterator = (row[:2] for row in y.itertuples(index=False, name=None))
            else:
                raise ValueError("Survival DataFrame must contain at least two columns.")
            return [cls._normalize_survival_pair(sample) for sample in iterator]

        if isinstance(y, pd.Series):
            return cls.normalize_survival_target(y.to_frame())

        if isinstance(y, np.ndarray):
            if y.dtype.names and 'event' in y.dtype.names and 'time' in y.dtype.names:
                return [cls._normalize_survival_pair((row['event'], row['time'])) for row in y]
            if y.ndim == 0:
                return [cls._normalize_survival_pair(y.item())]
            if y.ndim == 1:
                return [cls._normalize_survival_pair(sample) for sample in y.tolist()]
            if y.ndim >= 2 and y.shape[1] >= 2:
                return [cls._normalize_survival_pair(sample[:2]) for sample in y]

        if isinstance(y, (list, tuple)):
            return [cls._normalize_survival_pair(sample) for sample in y]

        if hasattr(y, '__iter__'):
            return cls.normalize_survival_target(list(y))

        raise TypeError(f"Unsupported survival target format: {type(y)}")

    @classmethod
    def fix_survival(cls, X: pd.DataFrame, y: Any) -> tuple[pd.DataFrame, np.ndarray]:
        """Turn dataframe to survival compatibility
        
        :param pd.DataFrame X: The dataframe to fix.
        :param Any y: The dataframe target to fix.
        :return: Fixed dataframe
        """
        from sksurv.util import Surv

        samples = cls.normalize_survival_target(y)
        if samples:
            events, times = zip(*samples)
            y_surv = Surv.from_arrays(
                event=np.asarray(events, dtype=bool),
                time=np.asarray(times, dtype=float)
            )
        else:
            y_surv = np.array([], dtype=[('event', 'bool'), ('time', 'float')])
        
        X = deepcopy(X)
        X[X.select_dtypes(include=['float64']).columns] = \
                X.select_dtypes(include=['float64']).astype('float32')

        return X, y_surv

    @classmethod
    def fix_y_survival(cls, y: Any, y_train: Any) -> list[tuple[bool, float]]:
        """Adjust survival targets to avoid censoring beyond the training horizon."""
        y_samples = cls.normalize_survival_target(y)
        y_train_samples = cls.normalize_survival_target(y_train)

        if not y_train_samples:
            return y_samples

        _, times = zip(*y_train_samples)
        censure_time = max(times)

        new_y: list[tuple[bool, float]] = []
        for event, time in y_samples:
            if time >= censure_time:
                time = censure_time
                event = False
            new_y.append((event, time))

        return new_y
