from ..step_wrapper import *
from ..output import *
from ..meta_explorer_step import MetaExplorerStep

from copy import deepcopy
import random

# Wrapper : Implementation of a Genetic GridSearch
@isStep('wrapper')
class WrapGeneticGridSearch(StepWrapper):
    name = "Wrap : Genetic GridSearch"
    def __init__(self, step):
        self.ignored_configs = set('random_state')
        
        self.configurations = [{
            'initial_modificator': {
                'description': 'Value modificator for each iteration',
                'default': 5
            },
            'nb_generations': {
                'description': 'TODO',
                'default': 5 #10
            },
            'nb_estimators': {
                'description': 'TODO',
                'default': 20,
                'range': [5, float('inf')]
            },
            'mutation_power': {
                'description': 'TODO',
                'default': 0.1,
                'range': [0.001, 1]
            }
        }]
        self.step = step
        self.step.keep_only_first_config() # Avoid run several config for each run
        self.step.current_configuration = self.step.configurations[0]
        
    to_avoid = ['random_state']
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
                
        # If no configuration
        if not(any(self.step.current_configuration.keys())):
            return self.step.run(input, callback=callback)
                
        # Generate first generation
        generation = []
        for i in range(0, self.get_config('nb_estimators')):
            generation.append(self.random_generation())
            
        for i_gen in range(0, self.get_config('nb_generations')):
            print("--- NEW GENERATION ---", i_gen, self.step.name)
            meta = MetaExplorerStep()
            meta.add_steps(generation)
            outputs = meta.run(input, callback=callback)
            if i_gen+1 < self.get_config('nb_generations'):
                print("GENERATE NEW !", self.step.name)
                # Generate next generation
                nb_to_get = int(len(generation)/4)
                outputs.sort()
                ordered_ids = self.__get_unique_ordered(list(map(lambda x: x.stacked_path[-2].step_id, outputs)))
                
                new_generation = []
                for i in range(0, nb_to_get):
                    new_generation.append(self.__find_step(generation, ordered_ids[i]))
                
                new_mutations = []
                for i in range(0, nb_to_get*2):
                    new_mutations.append(self.random_mutation(random.choice(new_generation)))
                    
                # Complet with random
                new_generation = new_generation + new_mutations
                while len(new_generation) < self.get_config('nb_estimators'):
                    new_generation.append(self.random_generation())
                    
                generation = new_generation
            else:
                print("FINISH ALL GENERATIONS", self.step.name)
        
        return outputs
            
            
            
    # Return Step with random configuration
    def random_generation(self):
        new_step = deepcopy(self.step)
        
        
        for key in self.__config_keys():
            config = new_step.current_configuration[key]
            if type(config['value']) in [int, float]:
                is_int = type(config['value']) == int
                
                new_value = None
                if bool(random.getrandbits(1)): # Negative
                    change_rate = random.uniform(0.01, 1)
                    new_value = config['value']*change_rate
                else: # positive
                    change_rate = random.uniform(1, self.get_config('initial_modificator'))
                    new_value = config['value']*change_rate
                
                if is_int:
                    new_value = round(new_value)
                
                if not self.__valide_config(config, new_value): # Cancel is the new value is not correct.
                    new_value = config['value'] # TODO Do better (for example : Run random generation again)
                    
            elif 'categorical' in config.keys(): # Categorial 
                new_value = random.choice(config['categorical'])
            elif type(config['value']) == bool: # Bool
                new_value = random.choice([True, False])
            else: # Other 
                new_value = config['value']
                
            new_step.configure_one(0, key, new_value)
            
        return new_step
            
    
    # Randomly mutate Step
    def random_mutation(self, step):
        new_step = deepcopy(step)
        random_key = random.choice(list(self.__config_keys()))
        random_item = new_step.current_configuration[random_key]
        new_value = None
        
        if type(random_item['value']) in [int, float]:
            is_int = type(random_item['value']) == int
            
            change_rate = random.uniform(-self.get_config('mutation_power'), self.get_config('mutation_power'))
            new_value = random_item['value']*(1+change_rate)
            
            if is_int:
                new_value = round(new_value)
                
            if new_value == random_item['value']: # To be sure there is a mutation
                new_value += random.choice([-1, 1])
                
            if not self.__valide_config(random_item, new_value): # Cancel is the new value is not correct.
                new_value = random_item['value'] # TODO Do better (for example : Run random generation again)
                
        elif 'categorical' in random_item.keys(): # Categorial 
            new_value = random.choice(random_item['categorical'])
        elif type(random_item['value']) == bool: # Bool
            new_value = random.choice([True, False])
        else: # Other 
            new_value = random_item['value']
            
        new_step.configure_one(0, random_key, new_value)
        return new_step
    
    def __get_unique_ordered(self, old_list):
        new_list = []
        for item in old_list:
            if item not in new_list:
                new_list.append(item)
        return new_list
    
    def __find_step(self, generation, step_id):
        for step in generation:
            if id(step) == step_id:
                return step
        
        return None
    
    def __valide_config(self, config, value):
        if 'range' not in config.keys():
            return True
        else:
            return config['range'][0] <= value <= config['range'][1]
        
    def __config_keys(self):
        return set(self.step.current_configuration.keys()) - self.ignored_configs