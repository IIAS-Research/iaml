"""
Reference class.
Contain all needed data to provide a reference for a step
"""
from typing import Dict


class Reference:  # pylint: disable=too-few-public-methods
    """
    Reference class.
    Contain all needed data to provide a reference for a step
    """
    
    def __init__(self, properties: Dict) -> None:
        """
        Instantiate all properties provided to the specific reference
        such as year of publication, authors, doi ...
        """
        for k, v in properties.items():
            setattr(self, k, v)
    
    def __repr__(self) -> str:
        """
        Return a simple string containing reference information
        """
        # return ' | '.join(f"[{k.capitalize()}] {v}" for k,v in self.__dict__.items())
        ret = ''
        try:
            ret = ret + ', '.join(self.authors) + '. '
        except AttributeError:
            pass
        try:
            ret = ret + self.name + '\n'
        except AttributeError:
            pass
        try:
            ret = ret + self.publisher + ', '
        except AttributeError:
            pass
        try:
            ret = ret + self.doi + ', '
        except AttributeError:
            pass
        try:
            ret = ret + str(self.year) +'.'
        except AttributeError:
            pass
        return ret
