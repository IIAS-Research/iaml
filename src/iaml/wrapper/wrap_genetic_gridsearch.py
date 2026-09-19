"""[WRAPPER] Genetic Grid Search implementation
This wrapper is inspired by genetic algorithms. 
It'll randomly create and mutate generations Step configurations to find the best parameters
Each new generation will learn from the previous one
"""
from copy import deepcopy
from math import isfinite
import random
from typing import Callable
from ..step_wrapper import StepWrapper
from ..step import Step
from ..decorators.all import is_step, runner
from ..candidate import Candidate
from ..meta_explorer_step import MetaExplorerStep
from ..logger import Logger


@is_step('wrapper')
class WrapGeneticGridSearch(StepWrapper):
    """[WRAPPER] Genetic Grid Search implementation
    This wrapper is inspired by genetic algorithms. 
    It'll randomly create and mutate generations Step configurations 
    to find the best parameters
    Each new generation will learn from the previous one
    """
    name = "Wrap : Genetic GridSearch"
    _usage: str = "Use when you need adaptive, multi-generation search over larger spaces and want broader exploration than WrapBasicGridSearch or WrapIterativeGridSearch. Applicable to numeric, categorical, and boolean hyperparameters. Avoid when budget is tight or a fixed grid suffices."

    def __init__(self, step: Step, *, evaluator: Callable[[Candidate], dict] = None):
        # Set of configuration key to ignore.
        # For example, random_state is not a parameter to optimize
        self.ignored_configs: set[str] = {'random_state'}
        # An evaluator can capture a dataset or splitter that changes between runs.
        # Recompute the search rather than cache results by input identity alone.
        self.use_cache = False
        self.evaluator = evaluator
        """Optional callback returning metric scores for each generated candidate.

        Use Candidate.training_evaluate with the original dataset and the desired
        splitter. Without a callback, multiple generations require scores already
        supplied by the wrapped step.
        """

        self.configuration = {
            'initial_modificator': {
                'description': 'Maximum multiplier of default value to generate \
                    the first generation of steps',
                'default': 5
            },
            'nb_generations': {
                'description': 'Number of generations to create, train and test',
                'default': 5,
                'range': [1, float('inf')]
            },
            'nb_estimators': {
                'description': 'Number of Steps by generations',
                'default': 15,
                'range': [1, float('inf')]
            },
            'mutation_power': {
                'description': 'Maximum multiplier of current value when mutating',
                'default': 0.1,
                'range': [0.001, 1]
            }
        }
        """Dictionnary of genetic configuration"""

    # pylint: disable=too-many-locals
    @runner
    def run(self, candidate: Candidate) -> list[Candidate]:
        """Will iterate over generation to find best parameters

            Genetic GridSearch
                First generation
                    Generate nb_estimator Step randomly
                    Run and get result
                Next generations
                    Keep 1/4 BEST
                    Mutation 2/4 BEST to New steps
                    Generate totally new steps
        
        :param Candidate candidate: Candidate data
        :return: All generated Candidate
        """
        # If no configuration, let's run the step once. Nothing to optimize here
        if not self.__config_keys():
            return self.step.run(candidate)

        population_size = self.get_config('nb_estimators')
        generations = self.get_config('nb_generations')
        for name, value in [('nb_estimators', population_size), ('nb_generations', generations)]:
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')

        generation = [self.random_generation() for _ in range(population_size)]
        candidates = []
        for i_gen in range(generations):
            Logger().info(f"created new generation: [b]{self.step.__class__.__name__}[/] \
                (generation={i_gen})")

            meta = MetaExplorerStep() # Use MetaExplorer to run all our generation easily
            meta.add_steps(generation) # Give all steps to MetaExplorer
            for step in generation:
                step.candidate = []
            meta.run(candidate)

            # The runner keeps each step's outputs. No legacy Stack history is needed.
            ranked = []
            for step in generation:
                for result in step.candidate or []:
                    result.main_metric = candidate.get_main_metric()
                    if self.evaluator is not None:
                        result.computed_metrics = self.evaluator(result) or {}
                        score = result.computed_metrics.get(result.main_metric)
                        if score is None or not isfinite(score):
                            continue
                    elif i_gen + 1 < generations and (
                        result.main_metric not in result.computed_metrics
                        or not isfinite(result.get_main_metric_value())
                    ):
                        raise ValueError(
                            'WrapGeneticGridSearch needs an evaluator returning the main '
                            'metric to select subsequent generations. Pass evaluator=... '
                            'using Candidate.training_evaluate on the original dataset, '
                            'or use GeneticOptimizer in IAML.'
                        )
                    ranked.append((result, step))

            candidates = [result for result, _ in ranked]
            if not candidates or i_gen + 1 == generations:
                break

            ranked.sort(key=lambda item: item[0], reverse=True)
            nb_to_keep = max(1, population_size // 4)
            steps_to_keep = []
            seen = set()
            for _, step in ranked:
                fingerprint = step.fingerprint()
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                kept = deepcopy(step)
                kept.parents_steps.remove(id(meta))
                steps_to_keep.append(kept)
                if len(steps_to_keep) == nb_to_keep:
                    break

            new_generation = list(steps_to_keep)
            for _ in range(min(2 * nb_to_keep, population_size - len(new_generation))):
                new_generation.append(self.random_mutation(random.choice(steps_to_keep)))
            while len(new_generation) < population_size:
                new_generation.append(self.random_generation())

            # Compare the complete effective configuration, including every key.
            generation = []
            seen = set()
            for step in new_generation:
                fingerprint = step.fingerprint()
                if fingerprint not in seen:
                    seen.add(fingerprint)
                    generation.append(step)

        return candidates

    # Return Step with random configuration
    def random_generation(self) -> Step:
        """Randomly generate a new Step

        :return: Generated step with random configuration
        """
        # Deepcopy to avoid editing other Steps of the same generation
        new_step: Step = deepcopy(self.step)

        for key in self.__config_keys(): # For each configuration key, we'll choose a random value
            config = new_step.configuration[key]

            if 'categorical' in config:
                new_value = random.choice(config['categorical'])
            elif type(config['value']) in [int, float]: # Numeric value ? Let's apply multiplier
                is_int = isinstance(config['value'], int)

                new_value = None
                # Randomly choose a positive or negative editing
                if bool(random.getrandbits(1)):
                    # Negative -> Multiply value by something between 0.01 and 1
                    change_rate = random.uniform(0.01, 1)
                    new_value = config['value']*change_rate
                else:
                    # Positive -> Multiply vaoue by something between 1
                    # and the max modificator in configuration
                    change_rate = random.uniform(1, self.get_config('initial_modificator'))
                    new_value = config['value']*change_rate

                # Value was a int ? Round it to keep it int
                if is_int:
                    new_value = round(new_value)

                if not self.__valide_config(config, new_value):
                    # Cancel is the new value is not correct.
                    new_value = config['value']

            elif isinstance(config['value'], bool):
                # Bool value, choose randomly beetwen True and False
                new_value = random.choice([True, False])
            else: # Other value ? Just keep it
                new_value = config['value']

            new_step.configure(key, new_value) # Set new configuration in the step

        return new_step

    # Randomly mutate Step
    def random_mutation(self, step: Step) -> Step:
        """Randomly mutate some parameters of the step

        :param Step step: Step to mutate.
        :return: Mutated Step
        """
        new_step: Step = deepcopy(step) # Deepcopy to avoid editing another Step

        keys = self.__config_keys()
        if not keys:
            return new_step
        random_key = random.choice(keys) # Choose a random key to mutate
        random_item = new_step.configuration[random_key] # Get value of the random key
        new_value = None

        if 'categorical' in random_item:
            new_value = random.choice(random_item['categorical'])
        elif type(random_item['value']) in [int, float]: # Numeric value ? Apply multiplier
            is_int = isinstance(random_item['value'], int)

            # Find a multiplier between - mutation_power & + mutation_power
            change_rate = random.uniform(-self.get_config('mutation_power'), \
                self.get_config('mutation_power'))
            new_value = random_item['value']*(1+change_rate) # Apply random multiplier

            # Value was a int ? Round it to keep it int
            if is_int:
                new_value = round(new_value)

            if new_value == random_item['value']: # To be sure there is a mutation
                new_value += random.choice([-1, 1])

            if not self.__valide_config(random_item, new_value):
                # Cancel is the new value is not correct.
                new_value = random_item['value']

        elif isinstance(random_item['value'], bool): # Bool -> Choose between True and False
            new_value = random.choice([True, False])
        else: # Other -> Keep it
            new_value = random_item['value']

        new_step.configure(random_key, new_value) # Apply configuration
        return new_step

    def conf_to_rich_str_list(self) -> list[str]:
        """List of String to Rich logger
        
        :return: list of stirng
        """
        l = [f'step={self.step.__class__.__name__}']
        l.extend(super().conf_to_rich_str_list())

        return l

    def count_steps(self) -> int:
        """Estimation of remaining step count

        :return: Number of steps
        """
        estimators  = self.get_config('nb_estimators')
        generations = self.get_config('nb_generations')

        return self.step.count_steps() * estimators * generations

    def __valide_config(self, config: dict, value: float) -> bool:
        """Is the configuration valid or not.
        
        :param dict config: The configuration to check.
        :param float value: The value to check in range.
        :return: Valid ?
        """
        if 'range' not in config.keys():
            return True
        lower, upper = config['range']
        return (lower is None or lower <= value) and (upper is None or value <= upper)

    # Get configurable keys (without ignored keys)
    def __config_keys(self):
        """Get configurable keys (without ignored keys)."""
        return [key for key, config in self.step.configuration.items()
                if key not in self.ignored_configs and not config.get('no_gridsearch', False)]
