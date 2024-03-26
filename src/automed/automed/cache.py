"""
Singleton used by Automed to cache results
"""
import pandas as pd
from .meta_singleton import MetaSingleton

class Cache(metaclass=MetaSingleton):
    """
    Singleton used by Automed to cache results
    """
    def __init__(self) -> None:
        self.saved:list = []
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
        
    def __get_from_fingerprint(self, fingerprint:str) -> list:
        return [item for item in self.saved if item[0] == fingerprint]
        
    def __delete(self, fingerprint:str, dataset) -> None:
        """Delete item from cache"""
        for idx, item in enumerate(self.saved):
            old_fingerprint, input_data, _ = item
            if old_fingerprint == fingerprint and dataset.equals(input_data):
                del self.saved[idx]
                break
                
        return [item for item in self.saved if item[0] == fingerprint]
        
    def from_cache(self, fingerprint:str, dataset:pd.DataFrame) -> any:  
        """
        Get data from cache
        
        Args:
            fingerprint (bool): New quiet value
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

    def add_to_cache(self, fingerprint:str, dataset:pd.DataFrame, output:any) -> None:
        """
        Add something to cache
        """
        if self.__disable:
            return None
        
        self.__delete(fingerprint, dataset)
        self.saved.append((fingerprint, dataset, output))
        
        if len(self.saved) > self.max_cache_size:
            del self.saved[0]
            
        return None
