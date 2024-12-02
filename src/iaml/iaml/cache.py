"""Singleton used by IAML to cache results"""
from copy import deepcopy
from typing import Any
import pandas as pd
from .meta_singleton import MetaSingleton


class Cache(metaclass=MetaSingleton):
    """Singleton used by IAML to cache results"""

    def __init__(self) -> None:
        self.saved: list = []
        """List of saved result"""

        self.max_cache_size: int = 100
        """Maximum number of cached results"""

        self.__disable: bool = False
        """Check if cache is disabled"""

    def disable(self) -> None:
        """Disable cache everywhere"""
        self.__disable = True

    def enable(self) -> None:
        """Enable cache everywhere"""
        self.__disable = False

    def __get_from_fingerprint(self, fingerprint: str) -> list:
        """Return cache from fingerprint
        
        :param str fingerprint: The fingerprinted cache to retrieve.
        :return: list of cached item
        """
        return [item for item in self.saved if item[0] == fingerprint]

    def __delete(self, fingerprint: str, dataset: pd.DataFrame) -> None:
        """Delete item from cache
        
        :param str fingerprint: Fingerprint used to identify task
        :param pd.DataFrame dataset: Dataset to compare cache to.
        """
        for idx, item in enumerate(self.saved):
            old_fingerprint, input_data, _ = item
            if old_fingerprint == fingerprint and dataset.equals(input_data):
                del self.saved[idx]
                break

    def from_cache(self, fingerprint: str, dataset: pd.DataFrame) -> Any:
        """Get data from cache
        
        :param str fingerprint: Fingerprint used to identify task
        :param pd.DataFrame dataset: Dataset to compare cache to.
        :return: cached data
        """
        if self.__disable:
            return None

        for idx, item in enumerate(self.__get_from_fingerprint(fingerprint)):
            _, input_data, output = item
            if dataset.equals(input_data):
                # Put item on the top of the list
                del self.saved[idx]
                self.saved.append(item)
                # Return cached data
                return output

        return None

    def add_to_cache(self, fingerprint: str, dataset: pd.DataFrame, output: Any) -> None:
        """
        Add something to cache
        
        Parameters
        ----------
        fingerprint : str
            fingerprint of the newly added object to cache
        
        """
        if self.__disable:
            return None

        self.__delete(fingerprint, dataset)
        self.saved.append((fingerprint, dataset, deepcopy(output)))

        if len(self.saved) > self.max_cache_size:
            del self.saved[0]

        return None
