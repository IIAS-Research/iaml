"""Regression tests for candidate identity and shared step caching."""
import gc
import unittest
import weakref
from copy import deepcopy
from unittest.mock import patch

import pandas as pd

from iaml import Candidate, Dataset, Step
from iaml.step_cache import StepCache


def make_candidate(offset=0):
    """Build independent candidates with distinguishable feature values."""
    return Candidate(Dataset(pd.DataFrame({"x": [offset, offset + 1]}), [0, 1]))


class TestStepCacheIdentity(unittest.TestCase):
    def setUp(self):
        self.step = Step()
        self.addCleanup(self.step.reset_cache)

    def test_reused_address_does_not_return_a_dead_candidates_output(self):
        first_input = make_candidate()
        first_output = make_candidate(10)
        first_ref = weakref.ref(first_input)

        # Force address reuse without depending on the interpreter's allocator.
        with patch("iaml.step.id", new=lambda _: 12345, create=True):
            self.step.add_cache(first_input, first_output)
            self.assertIs(self.step.from_cache(first_input), first_output)
            del first_input
            gc.collect()
            self.assertIsNone(first_ref(), "The cache must not retain its input")

            next_input = make_candidate(20)
            self.assertIsNone(self.step.from_cache(next_input))
            self.assertEqual(self.step.caches, [])
            self.assertEqual(StepCache().size_for_step(self.step._cache_id), 0)

            next_output = make_candidate(30)
            self.step.add_cache(next_input, next_output)
            self.assertIs(self.step.from_cache(next_input), next_output)

    def test_equal_candidates_with_colliding_addresses_do_not_share_results(self):
        first_input = make_candidate()
        other_input = make_candidate(10)
        for candidate in (first_input, other_input):
            candidate.computed_metrics = {candidate.main_metric: 0.75}
        self.assertEqual(first_input, other_input)
        self.assertIsNot(first_input, other_input)
        output = make_candidate(20)

        with patch("iaml.step.id", new=lambda _: 12345, create=True):
            self.step.add_cache(first_input, output)
            self.assertIsNone(self.step.from_cache(other_input))

    def test_step_copies_share_results_but_candidate_copies_do_not(self):
        candidate = make_candidate()
        output = make_candidate(10)
        self.step.add_cache(candidate, output)

        copied_step = deepcopy(self.step)
        self.assertIs(copied_step.from_cache(candidate), output)
        self.assertIsNone(copied_step.from_cache(deepcopy(candidate)))
        copied_step.reset_cache()
        self.assertIsNone(self.step.from_cache(candidate))


class TestStepCacheStorage(unittest.TestCase):
    def setUp(self):
        # Keep the package-wide singleton and other tests' caches untouched.
        self.cache = object.__new__(StepCache)
        StepCache.__init__(self.cache, max_size=2)

    def test_lru_eviction_keeps_step_indexes_and_outputs_consistent(self):
        first, second, third = [make_candidate(n) for n in (0, 10, 20)]
        first_key = ("first-step", "config", 1)
        second_key = ("second-step", "config", 2)
        third_key = ("first-step", "config", 3)
        first_output, second_output, third_output = object(), object(), object()
        self.cache.put(first_key, first_output, "first-step", first)
        self.cache.put(second_key, second_output, "second-step", second)
        self.assertIs(self.cache.get(first_key, first), first_output)
        self.cache.put(third_key, third_output, "first-step", third)

        self.assertIsNone(self.cache.get(second_key, second))
        self.assertEqual(self.cache.size_for_step("second-step"), 0)
        self.assertEqual(self.cache.values_for_step("second-step"), [])
        self.assertEqual(self.cache.total_size(), 2)
        self.assertEqual(
            self.cache.values_for_step("first-step"), [first_output, third_output]
        )
        self.cache.clear("first-step")
        self.assertEqual(self.cache.total_size(), 0)
        self.assertEqual(self.cache.size_for_step("first-step"), 0)

    def test_none_input_remains_cacheable_and_cannot_match_a_dead_input(self):
        key = ("step", "config", None)
        output = object()
        self.cache.put(key, output, "step", None)
        self.assertIs(self.cache.get(key, None), output)

        candidate = make_candidate()
        self.cache.put(key, output, "step", candidate)
        del candidate
        gc.collect()
        self.assertIsNone(self.cache.get(key, None))
        self.assertEqual(self.cache.total_size(), 0)
        self.assertEqual(self.cache.size_for_step("step"), 0)
