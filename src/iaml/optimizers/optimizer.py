
"""Base class of IAML Optimizer. Optimizer receive a pool of Candidates, 
optimize parameters and return a new pool of candidate
"""
from ..candidate import Candidate


class Optimizer:
    """Base class of IAML Optimizer. Optimizer receive a pool of Candidates, 
    optimize parameters and return a new pool of candidate
    """
    def __init__(self):
        self.__finished: bool = False
        """Is the optimization done ?"""

    @property
    def finished(self) -> bool:
        """Is the optimization finished ?

        :return: Finished ?
        """
        return self.__finished

    def run(self, candidates: list[Candidate]) -> list[Candidate]:
        """Run one optimisation stage. Run of Optimzer have to be overwrite (do nothing).

        :param list[Candidate] candidates: List of candidates to optimize
        :return: Optimized candidates
        """
        self.__finished = True
        return candidates
