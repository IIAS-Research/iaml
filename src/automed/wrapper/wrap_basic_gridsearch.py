"""
[WRAPPER] Wrap a step to apply Grid Search configuration parameters
"""
from ..step_wrapper import StepWrapper
from ..output import Output, Input
from ..step import is_step, runner


@is_step('wrapper')
class WrapBasicGridSearch(StepWrapper):
    """
    [WRAPPER] Wrap a step to apply Grid Search configuration parameters
    """
    
    to_avoid:list[str] = ['random_state'] # List of ignored key

    @runner
    def run(self, input_data:Input, callback:callable=None) -> list[Output]:
        """
        Super basic GridSearch.
            Numeric values -> 11 runs with 10% (50% to 150%)
            Categorial values -> 1 run each
            Boolean values -> Run with True and False
            Other -> keep current value

        Args:
            input_data (Input): Input data
            callback (callable, optional): Call after each step run. Defaults to None.

        Returns:
            list[Output]: All transformed input
        """
        
        to_explore = {}
        for config in self.step.configurations:
            for key, item in config.items():
                if key in self.to_avoid:
                    continue
                
                if 'categorical' in item.keys(): # Categorial 
                    to_explore[key] = item['categorical']
                elif type(item['value']) in [int, float]: # Numeric
                    current_value = item['value']
                    tmp = map(lambda x: x*current_value, [.5, .6, .7, .8, .9, 1, 1.1, 1.2, 1.3, 1.4, 1.5]) # pylint: disable=cell-var-from-loop
                    
                    # Keep int
                    if isinstance(item['value'], int):
                        tmp = [round(x) for x in tmp]
                    
                    # Check range
                    if 'range' in item.keys():
                        tmp = filter(lambda x: item['range'][0] >= x <= item['range'][1], tmp)  # pylint: disable=cell-var-from-loop
                    
                    to_explore[key] = set(tmp)
                        
                elif isinstance(item['value'], bool): # Bool
                    to_explore[key] = [True, False]
                else: # Other 
                    to_explore[key] = [item['value']]
                    
        self.step.keep_only_first_config() # Avoid run several config for each run
                    
        outputs = self.__recursive_run(input_data, to_explore, callback=callback)
        
        return outputs
    
    
    def __recursive_run(self, input_data, to_explore, callback=None):
        if any(list(to_explore.keys())):
            results = []
            key = list(to_explore.keys())[0]
            values = to_explore[key]
            
            del to_explore[key]
            
            for value in values:
                self.step.configure_one(0, key, value)
                output = self.__recursive_run(input_data, to_explore, callback=callback) 
                results = results + ([output] if type(output) == Output else output)
                
            return results
        else:
            return self.step.run(input_data, callback=callback)
