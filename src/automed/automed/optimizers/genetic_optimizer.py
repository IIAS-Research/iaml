"""
Pipeline optimizer based on genetic concepts
"""
from copy import deepcopy
import random
from ..candidate import Candidate
from .optimizer import Optimizer
from ..step import Step
from ..logger import Logger

# TODO -> Only optimize predictor for now. See if we can optimize cleaning stage

class GeneticOptimizer(Optimizer):
    """
    Pipeline optimizer based on genetic concepts
    """
    def __init__(self, nb_candidate:int=25, mutation_power:float=0.1, initial_modifier:float=5):
        super().__init__()
        self.number_of_candidate:int = max(nb_candidate, 4)
        self.generation_count:int = 0
        self.ignored_configs:list[str] = {'random_state'}
        self.mutation_power:float = mutation_power
        self.initial_modifier:float = initial_modifier
        self.max_generations:int = 200
        self.first_candidate_pool:list[Candidate] = None
        
    @property
    def finished(self) -> bool:
        """
        Does optimisation is finished ?

        Returns:
            bool: finished ?
        """
        return self.generation_count >= self.max_generations
    
    def run(self, candidates:list[Candidate]) -> list[Candidate]:
        """
        Run one optimisation stage. Run of Optimzer have to be overwrite (do nothing).

        Args:
            candidates (list[Candidate]): List of candidates to optimize

        Returns:
            list[Candidate]: Optimized candidates
        """
        candidates.sort(reverse=True)
        
        # Will be used for random generation
        if self.first_candidate_pool is None:
            self.first_candidate_pool = candidates[0:6]
            
        nb_to_keep:int = round(self.number_of_candidate / 4)
        nb_to_mutate:int = round(self.number_of_candidate / 4)
        
        self.generation_count += 1
                
        new_generation = [deepcopy(candidate) for candidate in candidates[0:nb_to_keep]]
        for candidate in candidates[0:nb_to_mutate]: # Mutate best candidates twice
            new_generation.append(self.__mutate(candidate))
            new_generation.append(self.__mutate(candidate))
            
        new_generation = [item for item in new_generation if item is not None] # remove None
        
        while len(new_generation) < self.number_of_candidate:
            new_generation += \
            [self.__random_configuration(random.choice(self.first_candidate_pool))]

        new_generation = [item for item in new_generation if item is not None] # remove None
        
        return self.__unique(new_generation) # Remove duplicated
    
    def __random_configuration(self, candidate:Candidate) -> Candidate:
        """
        From a candidate generate a new one with a full random configuration
        """
        # Deepcopy to avoid editing other Steps of the same generation
        new_candidate:Candidate = deepcopy(candidate)
        
        for current_step in new_candidate.pipeline.optimizable_step:
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
    
    def __mutate(self, candidate:Candidate) -> Candidate:
        """
        Mutate a candidate into a new one 
        """
        new_candidate:Candidate = deepcopy(candidate)
        step_to_mutate:Step = random.choice(new_candidate.pipeline.optimizable_step)
        
        mutable_keys = self.__config_keys(step_to_mutate)
        if step_to_mutate.is_interchangeable:
            mutable_keys.append("interchange")
        
        if not mutable_keys:
            return None # Nothing to optimize
        
        # Choose a random key to mutate
        random_key:str = random.choice(mutable_keys)
        
        if random_key == "interchange": # Mutate by interchanging the step with sibling
            new_step:Step = random.choice(step_to_mutate.step_with_same_tags())()
            new_step.is_interchangeable = True
            candidate.pipeline.replace_step(step_to_mutate, new_step)
            return candidate
            
        
        random_item:dict = step_to_mutate.configuration[random_key] # Get value of the random key
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

    def __config_keys(self, step:Step) -> list[str]:
        return list(set(step.configuration.keys()) - self.ignored_configs)
    
    # Does the configuration is valid or not ?
    def __valide_config(self, config:dict, value:any) -> bool:
        if 'range' not in config.keys():
            return True
        return config['range'][0] <= value <= config['range'][1]
    
    def __unique(self, candidates:list[Candidate]) -> list[Candidate]:
        unique_candidates:list[Candidate] = []
        unique_fingerprint:list[str] = []
        for candidate in candidates:
            fingerprint:str = candidate.pipeline.fingerprint()
            if fingerprint not in unique_fingerprint:
                unique_candidates.append(candidate)
                unique_fingerprint.append(fingerprint)
                
        return unique_candidates
