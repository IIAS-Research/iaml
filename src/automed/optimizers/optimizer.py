
"""
Base class of AutoMed Optimizer. Optimizer receive a pool of Candidates, 
optimize parameters and return a new pool of candidate
"""
from ..candidate import Candidate

class Optimizer():
    """
    Base class of AutoMed Optimizer. Optimizer receive a pool of Candidates, 
    optimize parameters and return a new pool of candidate
    """
    def __init__(self):
        self.__finished:bool = False
        
    @property
    def finished(self) -> bool:
        """
        Does optimisation is finished ?

        Returns:
            bool: finished ?
        """
        return self.__finished
    
    def run(self, candidates:list[Candidate]) -> list[Candidate]:
        """
        Run one optimisation stage. Run of Optimzer have to be overwrite (do nothing).

        Args:
            candidates (list[Candidate]): List of candidates to optimize

        Returns:
            list[Candidate]: Optimized candidates
        """
        self.__finished = True
        return candidates
