"""
    Step.run() decorator.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from ..logger import Logger

if TYPE_CHECKING:
    from ..candidate import Candidate

def runner(func) -> callable:
    """
    runner MUST decorate your run() method. It you manage every boring things for you.
        - Store results in cache
        - Send information to Destroyers
        - Put results in good shape
        - And maybe more

    Args:
        func (callable): decorated method

    Returns:
        callable: edited method
    """
    def runner_wrapper(self, candidates: list[Candidate]) -> list[Candidate]:
        """Wrapping decorated method

        Returns:
            list[Candidate]: All generated candidates
        """
        # Avoid circular import
        from ..candidate import Candidate  # pylint: disable=import-outside-toplevel

        if candidates.__class__ in [Candidate]:
            candidates = [candidates]

        result: list[Candidate] = []

        # only print "parent" steps to reduce logs
        if hasattr(self, 'step') or hasattr(self, 'steps'):
            Logger().info(f'running step: {self.to_rich_str()}')

        for current_candidate in candidates:
            if self.suitable(current_candidate.dataset) and self.enable:
                candidate = self.from_cache(current_candidate)
                if not candidate:
                    candidate = func(self, current_candidate)
                    self.add_cache(current_candidate, candidate)
            else: # If the step is disabled or not suitable for the dataset, do nothing
                candidate = current_candidate


            result = result + ([candidate] if type(candidate) in [Candidate] else candidate)

        self.candidate = result

        return result

    return runner_wrapper
