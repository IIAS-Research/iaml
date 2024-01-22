from ..step_wrapper import *
from ..output import *
from ..meta_explorer_step import MetaExplorerStep
from ..logger import Logger

from copy import deepcopy
import random

# TODO : Implement patience -> n generations without improvements -> Stop & keep best result

# Wrapper : Implementation of a Genetic GridSearch
#
# This wrapper is inspired by genetic algorithms. 
# It'll randomly create and mutate generations Step configurations to find the best parameters
# Each new generation will learn from the previous one
@isStep('wrapper')
class WrapGeneticGridSearch(StepWrapper):
    name = "Wrap : Genetic GridSearch"
    def __init__(self, step):
        self.ignored_configs = set(['random_state']) # Set of configuration key to ignore. For example, random_state is not a parameter to optimize
        
        self.configurations = [{
            'initial_modificator': {
                'description': 'Maximum multiplier of default value to generate the first generation of steps',
                'default': 5
            },
            'nb_generations': {
                'description': 'Number of generations to create, train and test',
                'default': 5, #10
                'range': [5, float('inf')]
            },
            'nb_estimators': {
                'description': 'Number of Steps by generations',
                'default': 15,
                'range': [5, float('inf')]
            },
            'mutation_power': {
                'description': 'Maximum multiplier of current value when mutating',
                'default': 0.1,
                'range': [0.001, 1]
            }
        }]
        
        super().__init__(step)
        self.step.keep_only_first_config() # Avoid run several config for each run
        self.step.current_configuration = self.step.configurations[0] # Will be defined when the step ".run()" but as we'll need it before, let's defined it now.
        
    @runner
    def run(self, input, callback=None):
        outputs = []
        # Genetic GridSearch
            # First generation
                # Generate nb_estimator Step randomly
                # Run and get result
            # Next generations
                # Keep 1/4 BEST
                # Mutation 2/4 BEST to New steps
                # Generate totally new steps
                
        # If no configuration, let's run the step once. Nothing to optimize here
        if not(any(self.step.current_configuration.keys())):
            return self.step.run(input, callback=callback)
                
                
        # Create the first generation of Steps. This first generation is full of random Steps configurations
        generation = []
        for i in range(0, self.get_config('nb_estimators')):
            generation.append(self.random_generation())
            
        # Loop one time by wanted generation
        for i_gen in range(0, self.get_config('nb_generations')):
            Logger().log(f"created new generation: [b]{self.step.__class__.__name__}[/] (generation={i_gen})")
            
            meta = MetaExplorerStep() # Use MetaExplorer to run all our generation easily
            meta.add_steps(generation) # Give all steps to MetaExplorer
            outputs = meta.run(input, callback=callback) # And run !
            
            # If this is not the last generation, let's create a new one
            if i_gen+1 < self.get_config('nb_generations'):
                # Generate next generation
                nb_to_get = int(self.get_config('nb_estimators')/4)
                outputs.sort()
                ordered_ids = self.__get_unique_ordered(list(map(lambda x: x.stacked_path[-2].step_id, outputs)))
                
                # Keep the 1/4 better Steps
                steps_to_keep = []
                for i in range(0, min(nb_to_get, len(generation))):
                    step = deepcopy(self.__find_step(generation, ordered_ids[i]))
                    del step.parents_steps[-1] # this step is going to switch parents
                    steps_to_keep.append(step)
                
                # Add 2/4 with mutated Steps (from the better steps of previous generation)
                new_mutations = []
                for i in range(0, nb_to_get*2):
                    new_mutations.append(self.random_mutation(random.choice(steps_to_keep)))
                
                # And add the last 1/4 with fully random Steps
                tmp_new_generation = (steps_to_keep + new_mutations)
                while len(tmp_new_generation) < self.get_config('nb_estimators'):
                    tmp_new_generation.append(self.random_generation())
                    
                # DROP Duplicated Steps
                new_generation = []
                for step in tmp_new_generation:
                    to_add = True
                    for to_filter in new_generation:
                        if self.__same_config(step.current_configuration, to_filter.current_configuration):
                            to_add = False
                            break
                        
                    if to_add:
                        new_generation.append(step)
                        
                        
                    
                generation = new_generation # Let's go for the next generation
            else:
                Logger().log(f"finished all generations: [b]{self.step.__class__.__name__}[/]")
        
        return outputs
            
            
            
    # Return Step with random configuration
    def random_generation(self):
        new_step: Step = deepcopy(self.step) # Deepcopy to avoid editing other Steps of the same generation
        
        for key in self.__config_keys(): # For each configuration key, we'll choose a random value
            config = new_step.current_configuration[key]
            
            if type(config['value']) in [int, float]: # Numeric value ? Let's apply multiplier
                is_int = type(config['value']) == int
                
                new_value = None
                # Randomly choose a positive or negative editing
                if bool(random.getrandbits(1)): # Negative -> Multiply value by something between 0.01 and 1
                    change_rate = random.uniform(0.01, 1)
                    new_value = config['value']*change_rate 
                else: # Positive -> Multiply vaoue by something between 1 and the max modificator in configuration 
                    change_rate = random.uniform(1, self.get_config('initial_modificator'))
                    new_value = config['value']*change_rate
                
                # Value was a int ? Round it to keep it int 
                if is_int: 
                    new_value = round(new_value)
                
                if not self.__valide_config(config, new_value): # Cancel is the new value is not correct.
                    new_value = config['value'] # TODO Do better (for example : Run random generation again)
                    
            elif 'categorical' in config.keys(): # Categorial value, choose randomly one of them
                new_value = random.choice(config['categorical'])
            elif type(config['value']) == bool: # Bool value, choose randomly beetwen True and False
                new_value = random.choice([True, False])
            else: # Other value ? Just keep it
                new_value = config['value']
                
            new_step.configure_one(0, key, new_value) # Set new configuration in the step
            
        return new_step
            
    
    # Randomly mutate Step
    def random_mutation(self, step):
        new_step: Step = deepcopy(step) # Deepcopy to avoid editing another Step
        
        random_key = random.choice(list(self.__config_keys())) # Choose a random key to mutate
        random_item = new_step.current_configuration[random_key] # Get value of the random key
        new_value = None
        
        if type(random_item['value']) in [int, float]: # Numeric value ? Apply multiplier
            is_int = type(random_item['value']) == int
            
            # Find a multiplier between - mutation_power & + mutation_power
            change_rate = random.uniform(-self.get_config('mutation_power'), self.get_config('mutation_power'))
            new_value = random_item['value']*(1+change_rate) # Apply random multiplier
            
            # Value was a int ? Round it to keep it int 
            if is_int:
                new_value = round(new_value)
                
            if new_value == random_item['value']: # To be sure there is a mutation
                new_value += random.choice([-1, 1])
                
            if not self.__valide_config(random_item, new_value): # Cancel is the new value is not correct.
                new_value = random_item['value'] # TODO Do better (for example : Run random generation again)
                
        elif 'categorical' in random_item.keys(): # Categorial -> Choose one
            new_value = random.choice(random_item['categorical'])
        elif type(random_item['value']) == bool: # Bool -> Choose between True and False
            new_value = random.choice([True, False])
        else: # Other -> Keep it
            new_value = random_item['value']
            
        new_step.configure_one(0, random_key, new_value) # Apply configuration
        return new_step
    

    def conf_to_rich_str_list(self):
        l = [f'step={self.step.__class__.__name__}']
        l.extend(super().conf_to_rich_str_list())

        return l
    

    def count_steps(self):
        estimators  = self.get_config('nb_estimators')
        generations = self.get_config('nb_generations')
        
        return self.step.count_steps() * sum([ estimators * (0.75 ** i) for i in range(generations) ]) # math

    
    def __same_config(self, a, b):
        if len(a.keys()) != len(b.keys()):
            return False
        for key, value in a.items():
            if isinstance(value, dict):
                return self.__same_config(value, b[key])
            elif key not in b or value != b[key]:
                return False
        return True
    
    # Return a new list with unique values in the same order. Useful because others methods with set can disorder values
    def __get_unique_ordered(self, old_list):
        new_list = []
        for item in old_list:
            if item not in new_list:
                new_list.append(item)
        return new_list
    
    # Find a step in a generation by id
    def __find_step(self, generation, step_id) -> Step | None:
        for step in generation:
            if id(step) == step_id:
                return step
        
        return None
    
    # Does the configuration is valid or not ?
    def __valide_config(self, config, value):
        if 'range' not in config.keys():
            return True
        else:
            return config['range'][0] <= value <= config['range'][1]
    
    # Get configurable keys (without ignored keys)    
    def __config_keys(self):
        # print("HERE !", set(self.step.current_configuration.keys()) - self.ignored_configs)
        return set(self.step.current_configuration.keys()) - self.ignored_configs