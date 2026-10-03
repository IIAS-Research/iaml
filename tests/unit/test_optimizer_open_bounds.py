"""Open numeric search domains remain usable without inventing finite bounds."""
from copy import deepcopy
import random
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from iaml.cache import Cache
from iaml.candidate import Candidate
from iaml.dataset import Dataset
from iaml.flow import Const, use
from iaml.logger import Logger
from iaml.optimizers import BayesianOptimizer, GeneticOptimizer, RandomOptimizer
from iaml.step import Step
from iaml.steps import SurvivalTree






class OpenBoundsOptimizerTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(setattr, Logger(), 'verbose', Logger().verbose)
        Logger().verbose = 0
        Cache().configure(None)
        self.X = pd.DataFrame({'x': np.arange(60, dtype=float),
                               'z': np.sin(np.arange(60))})
        self.y = np.array([(bool(i % 3), float(i + 1)) for i in range(60)], dtype=object)

    def candidate(self, bounds, value=3):
        step = Step()
        step.optimizable = True
        step.configuration = {'depth': {'value': value, 'range': list(bounds)}}
        candidate = Candidate(Dataset(self.X, self.y))
        candidate.pipeline.set_model(step)
        candidate.computed_metrics = {candidate.main_metric: 0.8}
        return candidate

    def assert_bounds(self, value, bounds):
        self.assertTrue(np.isfinite(value))
        low, high = bounds
        if low is not None:
            self.assertGreaterEqual(value, low)
        if high is not None:
            self.assertLessEqual(value, high)

    def test_random_and_genetic_explore_open_ranges_and_preserve_sources(self):
        ranges = ((1, None), (None, 10), (None, None),
                  (1, np.inf), (-np.inf, 10), (-np.inf, np.inf))
        for bounds in ranges:
            for optimizer in (RandomOptimizer(), GeneticOptimizer(nb_candidate=4)):
                with self.subTest(bounds=bounds, optimizer=type(optimizer).__name__):
                    source = self.candidate(bounds)
                    before = source.pipeline.fingerprint()
                    # Force both broad proposals and genetic local mutations.
                    def uniform(low, high):
                        return 0.5 if low < 0 else 3.0
                    with patch('random.getrandbits', return_value=0), \
                            patch('random.uniform', side_effect=uniform):
                        outputs = optimizer.run([source])
                    values = [item.pipeline.predictor[1].get_config('depth') for item in outputs]
                    self.assertTrue(any(value != 3 for value in values))
                    for item, value in zip(outputs, values):
                        self.assert_bounds(value, bounds)
                        self.assertEqual(item.pipeline.predictor[1].configuration['depth']['range'],
                                         list(bounds))
                    self.assertEqual(source.pipeline.fingerprint(), before)
                    self.assertEqual(source.pipeline.predictor[1].get_config('depth'), 3)

    def test_open_ranges_still_enforce_the_present_bound(self):
        for bounds, value, decrease in (((None, 4), 3, False), ((2, None), 3, True)):
            with self.subTest(bounds=bounds):
                source = self.candidate(bounds, value)
                with patch('random.getrandbits', return_value=int(decrease)), \
                        patch('random.uniform', return_value=0.01 if decrease else 5.0):
                    proposal = RandomOptimizer().run([source])[-1]
                self.assertEqual(proposal.pipeline.predictor[1].get_config('depth'), value)
        optimizer = GeneticOptimizer()
        source = self.candidate((2, None), value=2)
        with patch('random.uniform', return_value=-0.5):
            proposal = optimizer._GeneticOptimizer__mutate(source)
        self.assertEqual(proposal.pipeline.predictor[1].get_config('depth'), 2)

    def test_bayesian_keeps_open_range_candidates_without_finite_dimensions(self):
        for bounds in ((1, None), (None, 10), (None, None),
                       (1, np.inf), (-np.inf, 10), (-np.inf, np.inf)):
            with self.subTest(bounds=bounds):
                candidate = self.candidate(bounds)
                optimizer = BayesianOptimizer()
                self.assertEqual(optimizer.run([candidate]), [candidate])
                self.assertIs(optimizer.run([candidate])[0], candidate)
                self.assertEqual(optimizer.skopt_optimizers, {})

    def test_open_range_const_is_preserved_by_every_optimizer(self):
        recipe = use(SurvivalTree, max_depth=Const(3))
        recipe.configure(**{key: Const(param.value) for key, param in recipe.parameters.items()})
        source = Candidate(Dataset(self.X, self.y))
        source.pipeline.set_model(recipe.instantiate())
        source.computed_metrics = {source.main_metric: 0.8}
        for optimizer in (RandomOptimizer(), GeneticOptimizer(nb_candidate=4), BayesianOptimizer()):
            with self.subTest(optimizer=type(optimizer).__name__):
                outputs = optimizer.run([deepcopy(source)])
                self.assertTrue(outputs)
                for item in outputs:
                    self.assertEqual(item.pipeline.predictor[1].get_config('max_depth'), 3)



if __name__ == '__main__':
    unittest.main()
