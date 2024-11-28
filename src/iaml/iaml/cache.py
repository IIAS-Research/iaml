"""
Singleton used by IAML to cache results
"""
from copy import deepcopy
from typing import Any, List
import pandas as pd
from .meta_singleton import MetaSingleton

class Cache(metaclass=MetaSingleton):
    """
    Singleton used by IAML to cache results
    """
    
    def __init__(self) -> None:
        self.saved = []
        self.max_cache_size = 100
        self.__disable = False
        
    def disable(self) -> None:
        """
        Disable cache everywhere
        """
        self.__disable = True
        
    def enable(self) -> None:
        """
        Enable cache everywhere
        """
        self.__disable = False
        
    def __get_from_fingerprint(self, fingerprint: str) -> List:
        return [item for item in self.saved if item[0] == fingerprint]
        
    def __delete(self, fingerprint: str, dataset: pd.DataFrame) -> None:
        """
        Delete item from cache
        
        Parameters
        ----------
        fingerprint : str
            Fingerprint used to identify task
        dataset : pd.DataFrame
            _description_
        
        """
        for idx, item in enumerate(self.saved):
            old_fingerprint, input_data, _ = item
            if old_fingerprint == fingerprint and dataset.equals(input_data):
                del self.saved[idx]
                break
        
    def from_cache(self, fingerprint: str, dataset: pd.DataFrame) -> Any:  
        """
        Get data from cache
        
        Parameters
        ----------
        fingerprint : str
            Fingerprint used to identify task
        dataset : pd.DataFrame
            _description_
        
        Returns
        -------
        Any
            cached data
        
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
