from ..step_wrapper import *
from ..output import *

from copy import deepcopy


@isStep('wrapper')
class WrapIterativeGridSearch(StepWrapper):
    name = "Wrap : Iterative GridSearch"
    def __init__(self, step):
        self.configurations = [{
            'modificator': {
                'description': 'Value modificator for each iteration',
                'default': 0.25
            },
            'max_iterations': {
                'description': 'Maximum number of iterations',
                'default': 20
            },
            'patience': {
                'description': 'Stop iterations after N tries without improvements',
                'default': 5
            }
        }]
        self.step = step
        
    to_avoid = ['random_state']
    @runner
    def run(self, input, callback=None):
        # Iterative GridSearch
            # Numeric values
            #   -> Frist run -> 100% of the value
            #   -> Next runs -> +10% and - 10% (100% * modificator value)
            #   -> RUN
            #       -> Result is better ? Keep best result and try again with modificator
            #       -> Result is worst ? Keep previous result, update modificator
            #   -> STOP Conditions ? -> Number of iterations OR no improvement since X interations
            # -> Remember the range and then do a dichotomous to find the best parameters
        # Same as basic for the others types
            # Categorial values -> 1 run each
            # Boolean values -> Run with True and False
            # Other -> keep current value
            
        results = []
        configs = deepcopy(self.step.configurations)
        self.step.keep_only_first_config() # Avoid run several config for each run
        for config in configs:
            output = self.__recursive_run(input, config, callback=callback)
            results = results + ([output] if type(output) == Output else output)
        
        return results
    
    
    def __recursive_run(self, input, config, callback=None):
        if any(list(config.keys())): # Any thing to explore ?
            key = list(config.keys())[0] # Pick a key
            item = config[key] # Item to explore
            del config[key] # Remove for next explorations
            
            results = []
            
            if key in self.to_avoid: # Nothing to explore here, continue
                output = self.__recursive_run(input, config, callback=callback)
            else:
                if type(item['value']) in [int, float]: # Numeric
                    # modificator
                    modi = self.get_config('modificator')
                    minimal_modi = item['value']*0.01
            
                    # Step 1 -> Base value
                    output = self.__recursive_run(input, config, callback=callback) 
                    results = results + ([output] if type(output) == Output else output)
                    max_result = self.__find_best(output)
                    very_max_value = max_result
                    
                    
                    # Step 2 -> Explore
                    current_iterations = [
                        {
                            'value': item['value'],
                            'result': max_result,
                            'modificator': item['value'] * modi,
                            'patience': 0,
                            'stop': False 
                        },
                        {
                            'value': item['value'],
                            'result': max_result,
                            'modificator': item['value'] * (modi * -1),
                            'patience': 0,
                            'stop': False
                        },]
                    
                    for i in range(0, self.get_config('max_iterations')):
                        change = False
                        for current_iteration in current_iterations:
                            if current_iteration['stop']:
                                continue
                            
                            value = current_iteration['value'] + current_iteration['modificator']
                            
                            if isinstance(item['value'], int):
                                value = int(value)
                                
                            if value == item['value']:
                                current_iteration['stop'] = True
                                continue
                            
                            # Check range
                            if ('range' in item.keys()) and not(item['range'][0] <= value <= item['range']):
                                current_iteration['stop'] = True
                                continue
                                
                                    
                            # RUN
                            self.step.configure_one(0, key, value)
                            output = self.__recursive_run(input, config, callback=callback) 
                            results = results + ([output] if type(output) == Output else output)
                            max_result = self.__find_best(output)
                            
                            if max_result < current_iteration['result']:
                                current_iteration['stop'] = True
                                if current_iteration['value'] >= very_max_value:
                                    change = True
                                    # Create new iterations
                                    middle = min([current_iteration['value'], value]) + abs(current_iteration['value'] - value)/2
                                    new_modificator = abs(current_iteration['value'] - value) * modi
                                    if new_modificator < minimal_modi:
                                        continue
                                    
                                    current_iterations.append({
                                        'value': middle,
                                        'result': current_iteration['result'],
                                        'modificator': new_modificator,
                                        'patience': 0,
                                        'stop': False
                                    })
                                    current_iterations.append({
                                        'value': middle,
                                        'result': current_iteration['result'],
                                        'modificator': new_modificator * -1,
                                        'patience': 0,
                                        'stop': False
                                    })
                            else:
                                change = True     
                                if max_result >= very_max_value:
                                    very_max_value = max_result
                        
                            current_iteration['result'] = max_result
                            current_iteration['value'] = value
                        
                        if not change:
                            break
            
                    
                else:
                    if 'categorical' in item.keys(): # Categorial 
                        values = item['categorical']
                    elif type(item['value']) == bool: # Bool
                        values = [True, False]
                    else: # Other 
                        values = [item['value']]
                    
                    for value in values:
                        print("->", self.step, key, value)
                        self.step.configure_one(0, key, value)
                        output = self.__recursive_run(input, config, callback=callback) 
                        results = results + ([output] if type(output) == Output else output)
                    
            return results
        else: # Nothing to explore -> RUN
            return self.step.run(input, callback=callback)
                
            
        
    def __find_best(self, outputs):
        # Find best result
        max_result = 0
        for output in outputs:
            tmp = output.metric.compute(output)
            if tmp > max_result:
                max_result = tmp
        
        return max_result