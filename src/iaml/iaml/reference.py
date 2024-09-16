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
    
    def __init__(self, properties: Dict, step_name: str) -> None:
        """
        Instantiate all properties provided to the specific reference
        such as year of publication, authors, doi ...
        """
        setattr(self, 'step', step_name)
        for k, v in properties.items():
            setattr(self, k, v)
    
    def __str__(self) -> str:
        """
        Return a simple string containing reference information
        """
        structured = ''
        try:
            structured = structured + ', '.join(self.authors) + '. '
        except AttributeError:
            pass
        try:
            structured = structured + str(self.name) + '\n'
        except AttributeError:
            pass
        try:
            structured = structured + str(self.publisher) + ', '
        except AttributeError:
            pass
        try:
            structured = structured + str(self.doi) + ', '
        except AttributeError:
            pass
        try:
            structured = structured + str(self.year) +'.'
        except AttributeError:
            pass
        return structured
    
    @classmethod
    def bibliography(cls, references: list):
        """
        Format a bibliography in a string from a list of references

        Params:
            references : List of References
        Returns:
            str: Formatted bibliography
        """
        spacing = len(str(len(references)))
        return '\n'.join([f"[{i+1:>{spacing}}]  {str(reference)}\n" \
            for i, reference in enumerate(references)])
