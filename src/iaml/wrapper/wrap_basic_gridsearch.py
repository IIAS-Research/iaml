"""
[WRAPPER] Wrap a step to apply Grid Search configuration parameters
"""
from ..step_wrapper import StepWrapper
from ..candidate import Candidate
from ..decorators.all import is_step, runner


@is_step('wrapper')
class WrapBasicGridSearch(StepWrapper):
    """
    [WRAPPER] Wrap a step to apply Grid Search configuration parameters
    """
    
    to_avoid:list[str] = ['random_state'] # List of ignored key

    @runner
    def run(self, candidate:Candidate) -> list[Candidate]:
        """
        Super basic GridSearch.
            Numeric values -> 11 runs with 10% (50% to 150%)
            Categorial values -> 1 run each
            Boolean values -> Run with True and False
            Other -> keep current value

        Args:
            candidate (Candidate): Candidate data

        Returns:
            list[Candidate]: All transformed candidate
        """
        
        to_explore = {}
        
        for key, item in self.step.configuration.items():
            if key in self.to_avoid:
                continue
            
            if 'categorical' in item.keys(): # Categorical 
                to_explore[key] = item['categorical']
            elif type(item['value']) in [int, float]: # Numeric
                current_value = item['value']
                
                # pylint: disable=cell-var-from-loop
                tmp = map(lambda x: x*current_value, \
                    [.5, .6, .7, .8, .9, 1, 1.1, 1.2, 1.3, 1.4, 1.5]\
                    ) 
                
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
        
        candidates = self.__recursive_run(candidate, to_explore)
        
        return candidates
    
    
    def __recursive_run(self, candidate, to_explore):
        if any(list(to_explore.keys())):
            results = []
            key = list(to_explore.keys())[0]
            values = to_explore[key]
            
            del to_explore[key]
            
            for value in values:
                self.step.configure(key, value)
                candidate = self.__recursive_run(candidate, to_explore) 
                results = results + ([candidate] if isinstance(candidate, Candidate) else candidate)
                
            return results
        return self.step.run(candidate)
