"""[WRAPPER] Wrap a step to apply an Iterative Grid Search implementation"""
from copy import deepcopy
from math import isfinite
from typing import Any, Callable
from ..step_wrapper import StepWrapper
from ..candidate import Candidate
from ..step import Step
from ..decorators.all import is_step, runner


# Wrapper : Implementation of an interative GridSearch
@is_step('wrapper')
class WrapIterativeGridSearch(StepWrapper):
    """[WRAPPER] Wrap a step to apply an Iterative Grid Search implementation
    
    :param Step step: Step to apply grid search on.
    :param evaluator: Callback returning metric scores, for example using
        Candidate.training_evaluate with the original dataset and desired splitter.
        Required unless the wrapped step supplies evaluation scores itself.
    """

    name = "Wrap : Iterative GridSearch"
    _usage = "Use when you need iterative narrowing of numeric grids, faster than WrapBasicGridSearch. Applicable to numeric-heavy configs with ordered ranges and small categorical/bool sets. Avoid when spaces are huge or unstructured; consider WrapGeneticGridSearch."
    def __init__(self, step: Step, *, evaluator: Callable[[Candidate], dict] = None) -> None:
        self.evaluator = evaluator
        # Evaluation callbacks may capture data that changes between runs.
        self.use_cache = False
        self.configuration = {
            'modificator': {
                'description': 'Value modificator for each iteration',
                'default': 0.5
            },
            'max_iterations': {
                'description': 'Maximum number of iterations',
                'default': 10
            },
            'patience': {
                'description': 'Stop iterations after N tries without improvements',
                'default': 3
            }
        }
        """Iterative grid search configuration"""

        self.step: Step = step
        """Step to perform grid search on"""

    to_avoid = ['random_state']

    @runner
    def run(self, candidate: Candidate) -> list[Candidate]:
        """Iterative GridSearch.

        Search numeric parameters as follows::

            Numeric values
                -> First run -> 100% of the value
                -> Next runs -> +10% and - 10% (100% * modificator value)
                -> RUN
                    -> Result is better ? Keep best result and try again with modificator
                    -> Result is worst ? Keep previous result, update modificator
                -> STOP Conditions ? -> Number of iterations OR no improvement since X interations
            -> Remember the range and then do a dichotomous to find the best parameters

        Use the basic grid search for other parameter types::

            Categorical values -> 1 run each
            Boolean values -> Run with True and False
            Other -> keep current value

        :param Candidate candidate: The candidate on which we run the grid search.
        :return: All generated Candidates.
        """
        step = deepcopy(self.step)
        config = deepcopy({key: item for key, item in step.configuration.items()
                           if key not in self.to_avoid and not item.get('no_gridsearch', False)})
        pending = [GridIteration(
            step, self.get_config('modificator'), copy_config=config,
            patience=self.get_config('patience'), max_iterations=self.get_config('max_iterations'),
            evaluator=self.evaluator,
        )]
        results = []
        while pending:
            iteration = pending.pop(0)
            current_results, siblings = iteration.run(candidate)
            results.extend(current_results)
            pending.extend(siblings)
        return sorted(results, reverse=True)[:10]


class GridIteration:  # pylint: disable=too-many-instance-attributes
    """One iteration of Iterative grid search
    
    :param Step step: The step we are working on.
    :param float modificator_rate: The modification ratio
    :param list, optional value_range: Range used to compute modificator. Default to None.
    :param int, optional patience: Patience. Default to 5.
    :param dict, optional copy_config: Config to copy. Default to None.
    :param str, optional key: Key to perform gridsearch on. Default to None.
    :param Any, optional value: Value of the key. Default to None.
    :param int, optional max_iterations: Maximum number of iterations. Default to 10.
    :param int, optional number_of_results: Maximum number of results. Default to 10.
    :param float, optional minimal_range_diff: Minimal range diff. Default to None.
    :param float, optional best_result: Best ranking score. Default to negative infinity.
    """
    # pylint: disable=too-many-arguments,too-many-locals,too-many-branches,R0915,R0917
    def __init__(
        self,
        step: Step,
        modificator_rate: float,
        value_range: list = None,
        patience: int = 5,
        copy_config: dict = None,
        key: str = None,
        value: Any = None,
        max_iterations: int = 10,
        number_of_results: int = 10,
        minimal_range_diff: float = None,
        best_result: float = float('-inf'),
        evaluator: Callable[[Candidate], dict] = None) -> None:
        self.step: Step = step
        """The step we are working on"""
        self.evaluator = evaluator

        self.modificator_rate: float = modificator_rate
        """Modificator ratio"""

        self.patience: int = patience
        """Patience"""

        self.config: dict = deepcopy(step.configuration) if copy_config is None else copy_config
        """Config used"""

        if key:
            self.key = key
        elif any(self.config.keys()):
            self.key = list(self.config.keys())[0]
        else:
            self.key = None

        self.max_iterations: int = max_iterations
        """Maximum number of iterations"""

        self.count_iterations: int = 0
        """Iteration counter"""

        self.iterations_without_improvement: int = 0
        """Iterations without improvement counter"""

        self.children: list = []
        """Childrens"""

        self.ways: list = []
        """List of ways to obtain possible values"""

        self.values: list = []
        """List of possible values"""

        if self.key:
            self.value = self.config[self.key]['value'] if value is None else value

            self.modificator = None
            self.minimal_range_diff = None
            if minimal_range_diff is not None:
                self.minimal_range_diff = minimal_range_diff

            # Type
            self.can_generate_sibling = False
            if 'categorical' in self.config[self.key]:
                self.values = list(self.config[self.key]['categorical'])
            elif type(self.value) in [int, float]:
                self.can_generate_sibling = True
                if value_range:
                    self.modificator = abs(value_range[1]-value_range[0])/2 * modificator_rate
                else:
                    self.modificator = abs(self.value) * self.modificator_rate

                for way_ind, way in enumerate([1, -1]):
                    way_values = [self.value+(self.modificator*ind*way) \
                        for ind in range(way_ind, self.max_iterations)]
                    if isinstance(self.value, int):
                        way_values = [round(v) for v in way_values]

                    if ('range' in self.config[self.key]) or value_range:
                        limits = value_range or self.config[self.key]['range']
                        way_values = [v for v in way_values
                                      if (limits[0] is None or limits[0] <= v)
                                      and (limits[1] is None or v <= limits[1])]

                    self.ways.append(list(dict.fromkeys(way_values)))

                self.__next_way()

                if self.minimal_range_diff is None:
                    self.minimal_range_diff = self.modificator * 0.1 # TODO improve this

            elif isinstance(self.value, bool): # Bool
                self.values = [True, False]
            else: # Other
                self.values = [self.value]

            self.candidates: list[Candidate] = []
            """List of returned candidates"""

            self.results: list = []
            """List of results"""

            self.number_of_results: int = number_of_results
            """Number of results"""

            self.best_result: float = best_result
            """Best result"""        

    def done(self) -> bool:
        """Iteration finished ?
        
        :return: Finished ?
        """
        if not self.can_generate_sibling:
            return self.count_iterations >= len(self.values)
        return (self.iterations_without_improvement >= self.patience) \
            or (self.count_iterations >= self.max_iterations) \
            or (len(self.values)-1 < self.count_iterations)

    def go_deeper(self) -> bool:
        """Need to go deeper ? (Remains parameter to optimize)
        
        :return: Deeper ?
        """
        return len(self.config.keys()) > 1

    def __generate_child(self):
        """Generate new childs"""
        child_config = deepcopy(self.config)
        del child_config[self.key]

        self.children.append(self.__class__(
            self.step,
            self.modificator_rate,
            patience=self.patience,
            best_result = self.best_result,
            copy_config=child_config,
            max_iterations=self.max_iterations,
            evaluator=self.evaluator))

    def __get_best_range(self) -> tuple:
        """Get the best range using previous results. 
        The best range will be the higher results + his highest neighbour.
        
        :return: best range.
        """
        if len(self.results) < 2:
            return None, None

        ordered = sorted(self.results, key=lambda result: result['value'])
        max_index = max(range(len(ordered)), key=lambda index: ordered[index]['result'])
        around = ordered[max(0, max_index-1):(max_index+2)]
        around = sorted(around, key=lambda x: x['result'])
        return tuple(sorted((around[-2]['value'], around[-1]['value'])))

    def __generate_siblings(self) -> list:
        """Generate siblings
        Siblings will be next iterator at the same level (same key, same step).
        They only explore the best range.
        
        :return: list of siblings
        """
        if not self.can_generate_sibling:
            return []

        mini, maxi = self.__get_best_range()

        if maxi is None or mini is None:
            return []

        current_range = [mini, maxi]
        middle = mini+(maxi-mini)/2

        if self.minimal_range_diff >= (maxi-mini):
            return []

        if isinstance(maxi, int):
            if (maxi-mini) <= 1:
                return []

            middle = round(middle)

        next_iter = self.__class__(
            self.step,
            self.modificator_rate,
            value=middle,
            value_range=current_range,
            patience=self.patience,
            copy_config=self.config,
            best_result = self.best_result,
            key=self.key,
            max_iterations=self.max_iterations,
            minimal_range_diff=self.minimal_range_diff,
            evaluator=self.evaluator)

        return [next_iter]

    def __next_way(self) -> None:
        """Go to the next direction. 
        Iterator values will go up and then go down.
        """
        while self.ways:
            self.values = self.ways.pop(0)
            self.count_iterations = 0
            self.iterations_without_improvement = 0
            if self.values:
                break

    def next_iteration(self) -> None:
        """
        Go to the next iteration or the next way if current one is finished
        """
        self.count_iterations = self.count_iterations + 1
        if self.done():
            self.__next_way()

    def current_value(self) -> Any:
        """Get current value
        
        :return: Current value
        """
        return self.values[self.count_iterations]

    def __stack_results(self, results: list[Candidate]) -> None:
        """Stack, compute and save results"""
        if not any(results):
            return None

        best = max(results, key=lambda result: result.get_main_metric_score())
        best_val = best.get_main_metric_score()

        self.results.append({'value': self.current_value(), 'result': best_val})

        if best_val <= self.best_result:
            self.iterations_without_improvement = self.iterations_without_improvement + 1
        else:
            self.best_result = best_val
            self.iterations_without_improvement = 0

        self.candidates.append(best)

        # Keep only n best
        self.candidates.sort(reverse=True)
        self.candidates = self.candidates[:self.number_of_results]

        return None

    def run(
        self,
        candidate: Candidate) -> tuple[list[Candidate], list['GridIteration']]:
        """Run and Stack results
        
        :param Candidate candidate: The candidate we run.
        :return: tuple of lists of candidates and GridIteration
        """
        # Run and Stack results
        if not self.key:
            return self.step.run(candidate.to_input()), []

        while not self.done():
            results = []
            self.step.configure(self.key, self.current_value())

            if self.go_deeper():
                self.__generate_child()
                results = []
                while self.children:
                    child = self.children.pop(0)
                    current_results, siblings = child.run(candidate)
                    results = results + current_results

                    if siblings:
                        self.children = self.children + siblings
            else:
                for result in deepcopy(self.step).run(candidate.to_input()):
                    result.main_metric = candidate.get_main_metric()
                    if self.evaluator is not None:
                        result.computed_metrics = self.evaluator(result) or {}
                    score = result.computed_metrics.get(result.main_metric)
                    if score is None or not isfinite(score):
                        if self.evaluator is None:
                            raise ValueError(
                                'WrapIterativeGridSearch needs an evaluator returning the main '
                                'metric. Pass evaluator=... using Candidate.training_evaluate '
                                'on the original dataset.'
                            )
                        continue
                    results.append(result)

            self.__stack_results(results)
            self.next_iteration()

        siblings = self.__generate_siblings()

        return self.candidates, siblings
