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
        self.saved:dict = {}
        
    def from_cache(self, fingerprint:str, dataset:pd.DataFrame) -> any:
        """
        Get data from cache
        
        Args:
            fingerprint (bool): New quiet value
        """
        if fingerprint in self.saved:
            cached = self.saved[fingerprint]
            for old_data, output in cached:
                if dataset.equals(old_data):
                    return output
        return None

    def add_to_cache(self, fingerprint:str, dataset:pd.DataFrame, output:any) -> None:
        """
        Add something to cache
        """
        if fingerprint not in self.saved:
            self.saved[fingerprint] = []
                
        self.saved[fingerprint].append((dataset, output))
