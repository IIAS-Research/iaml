"""
Reference class.
Contain all needed data to provide a reference for a step
"""
from typing import List, Dict


class Reference:  # pylint: disable=too-few-public-methods
    """Reference class.
    Contain all needed data to provide a reference for a step
    
    :param Dict properties: a dictionnary of properties for a reference object
    :param str step_name: The step_name attached to this reference
    """

    def __init__(self, properties: Dict, step_name: str) -> None:
        """Instantiate all properties provided to the specific reference
        such as year of publication, authors, doi ...
        """
        setattr(self, 'step', step_name)
        for k, v in properties.items():
            setattr(self, k, v)

    def __str__(self) -> str:
        """Return a simple string containing reference information
        
        :return: The reference representation
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
    def bibliography(cls, references: List['Reference'], structured: bool) -> str | List[Dict]:
        """Format a bibliography in a string from a list of references

        :param List[Reference] references: List of References
        :param bool structured: Wether we want a string bibliography or a list of references

        :return: Bibliography in string or List format
        """
        if structured:
            return [vars(r) for r in references]
        spacing = len(str(len(references)))
        return '\n'.join([f"[{i+1:>{spacing}}]  {str(reference)}\n" \
            for i, reference in enumerate(references)])
