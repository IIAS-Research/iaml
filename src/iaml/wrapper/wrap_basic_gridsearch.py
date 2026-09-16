"""[WRAPPER] Wrap a step to apply Grid Search configuration parameters"""
from copy import deepcopy
from itertools import product

from ..step_wrapper import StepWrapper
from ..candidate import Candidate
from ..decorators.all import is_step, runner


@is_step('wrapper')
class WrapBasicGridSearch(StepWrapper):
    """[WRAPPER] Wrap a step to apply Grid Search configuration parameters"""

    _usage: str = "Use when you want a quick, simple grid around current values and prefer it over WrapGeneticGridSearch or WrapIterativeGridSearch. Applicable to numeric, categorical, and boolean hyperparameters. Avoid when search space is large or needs adaptive or iterative exploration."
    to_avoid: list[str] = ['random_state'] # List of ignored key

    @runner
    def run(self, candidate: Candidate) -> list[Candidate]:
        """Super basic GridSearch.
        
        - Numeric values -> 11 runs with 10% (50% to 150%)
        - Categorial values -> 1 run each
        - Boolean values -> Run with True and False
        - Other -> keep current value

        :param Candidate candidate: Candidate data
        :return: All transformed candidates
        """

        to_explore = {}

        for key, item in self.step.configuration.items():
            if key in self.to_avoid or item.get('no_gridsearch'):
                continue

            if 'categorical' in item.keys(): # Categorical
                to_explore[key] = item['categorical']
            elif type(item['value']) in [int, float]: # Numeric
                current_value = item['value']

                tmp = [factor * current_value for factor in
                       [.5, .6, .7, .8, .9, 1, 1.1, 1.2, 1.3, 1.4, 1.5]]

                # Keep int
                if isinstance(item['value'], int):
                    tmp = [round(x) for x in tmp]

                # Check range
                if 'range' in item.keys():
                    lower, upper = item['range']
                    tmp = [value for value in tmp
                           if (lower is None or lower <= value)
                           and (upper is None or value <= upper)]

                to_explore[key] = list(dict.fromkeys(tmp))

            elif isinstance(item['value'], bool): # Bool
                to_explore[key] = [True, False]
            else: # Other
                to_explore[key] = [item['value']]

        candidates = []
        for values in product(*to_explore.values()):
            step = deepcopy(self.step)
            step.configure(dict(zip(to_explore, values)))
            results = step.run(candidate.to_output())
            candidates.extend([results] if isinstance(results, Candidate) else results)
        return candidates
