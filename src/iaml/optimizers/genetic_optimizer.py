"""Pipeline optimizer based on genetic concepts"""
from copy import deepcopy
import random
import time
from typing import Any
from ..candidate import Candidate
from .optimizer import Optimizer
from ..step import Step
from ..void_step import VoidStep


class GeneticOptimizer(Optimizer): # pylint: disable=too-many-instance-attributes
    """Pipeline optimizer based on genetic concepts
    
    :param int, optional nb_candidate: Number of candidates to optimize. Default to 35.
    :param float, optional mutation_power: Chance of a mutation hapenning. Default to 0.1.
    :param float, optional initial_modifier: Maximum modification value possible. Default to 5.
    :param int, optional duration: Used to compute a mutation ratio. Default to None.
    """

    def __init__(
        self,
        nb_candidate: int = 35,
        mutation_power: float = 0.1,
        initial_modifier: float = 5,
        duration: int = None) -> None:
        super().__init__()
        self.number_of_candidate: int = max(nb_candidate, 4)
        """Maxmimum number of candicates"""

        self.generation_count: int = 0
        """Generation coutner"""

        self.ignored_configs: set[str] = {'random_state'}
        """Set of config keys to ignore"""

        self.mutation_power: float = mutation_power
        """Chance of a mutation hapenning"""

        self.initial_modifier: float = initial_modifier
        """Maximum modification value possible"""

        self.max_generations: int = 200
        """Maximum number of generations"""

        self.first_candidate_pool: list[Candidate] = None
        """List of candidates for the first generation"""

        self.duration: int = duration
        """Used to compute a mutation ratio"""

        self.start_time: float = time.time()
        """Starting time of the first generation"""

    @property
    def __mutate_ratio(self) -> float:
        """Define a mutation ratio"""
        if not self.duration:
            return 0.5
        return min(0.9, max(0.1, ((time.time() - self.start_time) / self.duration)))

    @property
    def finished(self) -> bool:
        """Is optimization finished ?
    
        :return: finished ?
        """
        return self.generation_count >= self.max_generations

    def run(self, candidates: list[Candidate]) -> list[Candidate]:
        """Run one optimisation stage. Run of Optimzer have to be overwrite (do nothing).

        :param list[Candidate] candidates: List of candidates to optimize
        :return: Optimized candidates
        """
        candidates.sort(reverse=True)

        # Will be used for random generation
        if self.first_candidate_pool is None:
            self.first_candidate_pool = candidates[0:6]

        nb_to_keep: int = round(self.number_of_candidate / 4)
        # nb_to_keep: int = max(min(4, round(self.number_of_candidate / 4)), 1)
        mutate_ratio = self.__mutate_ratio
        self.generation_count += 1

        # Keep best pipelines
        new_generation = [deepcopy(candidate) for candidate in candidates[0:nb_to_keep]]

        for _ in range(int(self.number_of_candidate*mutate_ratio)):
            new_generation.append(self.__mutate(random.choice(candidates[0:nb_to_keep])))

        new_generation = [item for item in new_generation if item is not None] # remove None

        while len(new_generation) < self.number_of_candidate:
            new_generation.append(
                self.__random_configuration(random.choice(candidates))
            )

        new_generation = [item for item in new_generation if item is not None] # remove None

        return self.__unique(new_generation) # Remove duplicated

    def __random_configuration(self, candidate: Candidate) -> Candidate:
        """From a candidate generate a new one with a full random configuration

        :param Candidate candidate: Candidate used to generate a new one
        :return: Newly generated Candidate
        """
        # Deepcopy to avoid editing other Steps of the same generation
        new_candidate: Candidate = deepcopy(candidate)

        for current_step in new_candidate.pipeline.optimizable_step:

            # If interchangeable -> 1/2 to change the step
            if current_step.is_interchangeable and bool(random.getrandbits(1)):
                new_step = self.__interchange(current_step)
                new_candidate.pipeline.replace_step(current_step, new_step)
                current_step = new_step
            if not self.__config_keys(current_step):
                continue # Nothing to optimize

            # For each configuration key, we'll choose a random value
            for key in self.__config_keys(current_step):

                # 1/2 chance to let the default value unchanged
                if bool(random.getrandbits(1)):
                    continue

                config = current_step.configuration[key]

                if type(config['value']) in [int, float]: # Numeric value ? Let's apply multiplier
                    is_int = isinstance(config['value'], int)

                    new_value = None
                    if 'range' in config: # Random in range
                        new_value = random.uniform(*config['range'])
                    else: # Kind of strong mutate
                        # Randomly choose a positive or negative editing
                        if bool(random.getrandbits(1)):
                            # Negative -> Multiply value by something between 0.01 and 1
                            change_rate = random.uniform(0.01, 1)
                            new_value = config['value']*change_rate
                        else:
                            # Positive -> Multiply value by something between 1
                            # and the max modificator in configuration
                            change_rate = random.uniform(1, self.initial_modifier)
                            new_value = config['value']*change_rate

                    # Value was a int ? Round it to keep it int
                    if is_int:
                        new_value = round(new_value)

                    if not self.__valide_config(config, new_value):
                        # Cancel is the new value is not correct.
                        new_value = config['value']

                # Categorical value, choose randomly one of them
                elif 'categorical' in config.keys():
                    new_value = random.choice(config['categorical'])
                elif isinstance(config['value'], bool):
                    # Bool value, choose randomly beetwen True and False
                    new_value = random.choice([True, False])
                else: # Other value ? Just keep it
                    new_value = config['value']

                current_step.configure(key, new_value) # Set new configuration in the step

        return new_candidate

    def __mutate(self, candidate: Candidate) -> Candidate:
        """Mutate a candidate into a new one
        
        :param Candidate candidate: Candidate used for the mutation.
        :return: Newly created Candidate.
        """
        new_candidate: Candidate = deepcopy(candidate)
        steps = new_candidate.pipeline.optimizable_step
        if not steps:
            return None
        step_to_mutate: Step = random.choice(steps)

        mutable_keys = self.__config_keys(step_to_mutate)
        if step_to_mutate.is_interchangeable:
            mutable_keys.append("interchange")

        if not mutable_keys:
            return None # Nothing to optimize

        # Choose a random key to mutate
        random_key: str = random.choice(mutable_keys)

        if random_key == "interchange": # Mutate by interchanging the step with sibling
            new_step = self.__interchange(step_to_mutate)
            new_candidate.pipeline.replace_step(step_to_mutate, new_step)
            return new_candidate


        random_item: dict = step_to_mutate.configuration[random_key] # Get value of the random key
        new_value = None

        if type(random_item['value']) in [int, float]: # Numeric value ? Apply multiplier
            is_int = isinstance(random_item['value'], int)

            # Find a multiplier between - mutation_power & + mutation_power
            change_rate = random.uniform(-self.mutation_power, self.mutation_power)
            new_value = random_item['value']*(1+change_rate) # Apply random multiplier

            # Value was a int ? Round it to keep it int
            if is_int:
                new_value = round(new_value)

            if new_value == random_item['value']: # To be sure there is a mutation
                new_value += random.choice([-1, 1])

            if not self.__valide_config(random_item, new_value):
                # Cancel if the new value is not correct.
                new_value = random_item['value']

        elif 'categorical' in random_item.keys(): # Categorial -> Choose one
            new_value = random.choice(random_item['categorical'])
        elif isinstance(random_item['value'], bool): # Bool -> Choose between True and False
            new_value = random.choice([True, False])
        else: # Other -> Keep it
            new_value = random_item['value']

        step_to_mutate.configure(random_key, new_value) # Apply configuration

        return new_candidate

    @staticmethod
    def __interchange(step: Step) -> Step:
        """Choose another implementation, allowing optional preprocessing to be removed."""
        choices = [sibling for sibling in step.step_with_same_tags()
                   if sibling is not type(step)]
        if (not isinstance(step, VoidStep) and step.can_be_disabled
                and not hasattr(step, 'predict')
                and (hasattr(step, 'transform') or hasattr(step, 'resample'))):
            choices.append(VoidStep)
        if not choices:
            return step
        step_class = random.choice(choices)
        if step_class is VoidStep:
            replacement = VoidStep(step_to_mimic=type(step)())
        else:
            replacement = step_class()
        replacement.is_interchangeable = True
        return replacement

    def __config_keys(self, step: Step) -> list[str]:
        """Get a list of config keys used for a Step
        
        :param Step step: The step we want the config
        :return: List of config keys for this step
        """
        if not step.optimizable:
            return []
        return list(set(step.configuration.keys()) - self.ignored_configs)

    def __valide_config(self, config: dict, value: Any) -> bool:
        """Is the configuration range valid ?
        
        :param dict config: The config to validate.
        :return: Valid ?
        """
        if 'range' not in config.keys():
            return True
        return config['range'][0] <= value <= config['range'][1]

    def __unique(self, candidates: list[Candidate]) -> list[Candidate]:
        """Get a list of unique Candidates

        :param list[Candidate] candidates: List of Candidate.
        :return: List of unique Candidate
        """
        unique_candidates: list[Candidate] = []
        unique_fingerprint: list[str] = []
        for candidate in candidates:
            fingerprint: str = candidate.pipeline.fingerprint()
            if fingerprint not in unique_fingerprint:
                unique_candidates.append(candidate)
                unique_fingerprint.append(fingerprint)

        return unique_candidates
