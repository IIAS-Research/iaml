
"""
Base class of IAML Optimizer. Optimizer receive a pool of Candidates, 
optimize parameters and return a new pool of candidate
"""
from copy import deepcopy
import random
import time
from ..candidate import Candidate
from .optimizer import Optimizer
from ..step import Step
from ..logger import Logger

class RandomOptimizer(Optimizer):
    def __init__(self, duration:int=None, max_iterations=50):
        """
        Initialize the Random Search Optimizer.
        :param candidates: List of Candidate objects to optimize.
        :param max_iterations: Number of random samples to evaluate.
        :param patience: Number of iterations without improvement before stopping.
        """
        self.max_iterations = max_iterations
        self.current_iteration = 0
        self.initial_modifier:float = 5
        self.max_candidates = 40
        self.first_candidates = None
        
    
    def _randomize_hyperparameters(self, candidate):
        """Randomly modifies the hyperparameters of a given candidate."""
        for _, step in candidate.pipeline.steps:
            for key, config in step.configuration.items():
                # Get random values
                
                if type(config['value']) in [int, float]: # Numeric value ? Let's apply multiplier
                    is_int = isinstance(config['value'], int)
                    
                    new_value = None
                    if 'range' in config: # Random in range
                        new_value = random.uniform(*config['range'])
                    else: # Strong multiplier -> kind of random
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
                
                step.configure(key, new_value)
                
        return candidate
    
    def run(self, candidates):
        """
        Perform a single iteration of random search optimization.
        :param evaluate_fn: Function that takes a Candidate object and returns a performance score.
        """
        candidates = candidates[0:self.max_candidates//2]
        new_candidates = []
        for candidate in candidates:
            new_candidates.append(self._randomize_hyperparameters(deepcopy(candidate)))
        
        self.current_iteration += 1
        return candidates + new_candidates
    
    @property
    def finished(self) -> bool:
        """
        Does optimisation is finished ?

        Returns:
            bool: finished ?
        """
        return self.current_iteration >= self.max_iterations
    
    # Does the configuration is valid or not ?
    def __valide_config(self, config:dict, value:any) -> bool:
        if 'range' not in config.keys():
            return True
        return config['range'][0] <= value <= config['range'][1]