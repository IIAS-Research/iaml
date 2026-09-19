"""Step.run() decorator."""
from __future__ import annotations
from typing import TYPE_CHECKING
import time

from ..logger import Logger

if TYPE_CHECKING:
    from ..candidate import Candidate


def runner(func: callable) -> callable:
    """runner MUST decorate your run() method. It you manage every boring things for you.

        - Store results in cache
        - Send information to Destroyers
        - Put results in good shape
        - And maybe more

    :param callable func: decorated method.
    :return: edited method.
    """
    def runner_wrapper(self, candidates: list[Candidate]) -> list[Candidate]:
        """Wrapping decorated method

        :param list[Candidate] candidates: Candidates to apply wrapper to.
        :return: All generated candidates.
        """
        # Avoid circular import
        from ..candidate import Candidate  # pylint: disable=import-outside-toplevel

        if candidates.__class__ in [Candidate]:
            candidates = [candidates]

        def _shape_from_candidate(candidate: Candidate) -> str:
            try:
                dataset = candidate.dataset
                if dataset is None or dataset.X is None:
                    return "None"
                return f"{dataset.X.shape[0]}x{dataset.X.shape[1]}"
            except Exception:  # pylint: disable=broad-except
                return "?"

        def _summarize_shapes(cands: list[Candidate]) -> str:
            if not cands:
                return "none"
            counts: dict[str, int] = {}
            for cand in cands:
                shape = _shape_from_candidate(cand)
                counts[shape] = counts.get(shape, 0) + 1
            parts = []
            for shape in sorted(counts.keys()):
                count = counts[shape]
                if count == 1:
                    parts.append(shape)
                else:
                    parts.append(f"{shape} (x{count})")
            return ", ".join(parts)

        logger = Logger()
        log_timing = logger.verbose > 1
        step_label = None
        start_time = None
        input_shapes = None
        if log_timing:
            try:
                step_label = self.to_rich_str()
            except Exception:  # pylint: disable=broad-except
                step_label = self.__class__.__name__
            logger.info(f'running step: {step_label}')
            start_time = time.perf_counter()
            input_shapes = _summarize_shapes(candidates)

        result: list[Candidate] = []

        for current_candidate in candidates:
            if self.suitable(current_candidate.dataset) and self.enable:
                candidate = self.from_cache(current_candidate)
                if not candidate:
                    candidate = func(self, current_candidate)
                    self.add_cache(current_candidate, candidate)
            else: # If the step is disabled or not suitable for the dataset, do nothing
                candidate = current_candidate


            result = result + ([candidate] if type(candidate) in [Candidate] else candidate)

        if log_timing and start_time is not None:
            elapsed = time.perf_counter() - start_time
            output_shapes = _summarize_shapes(result)
            logger.info(
                f'finished step: {step_label} in {elapsed:.2f}s '
                f'(outputs={len(result)}, data={input_shapes}->{output_shapes})'
            )

        self.candidate = result

        return result

    return runner_wrapper
