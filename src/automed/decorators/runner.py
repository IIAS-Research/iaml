"""
    Step.run() decorator.
"""
from ..output import Input, Output
from ..logger import Logger

def runner(func) -> callable:
    """
    runner MUST decorate your run() method. It you manage every boring things for you.
        - Store results in cache
        - Send information to Destroyers
        - Put results in good shape
        - Increment Stack data
        - Call callback method
        - And maybe more

    Args:
        func (callable): decorated method

    Returns:
        callable: edited method
    """
    def runner_wrapper(self, inputs:list[Input],
                        callback:callable=None
                        ) -> list[Output]:
        """Wrapping decorated method

        Returns:
            list[Output]: All generated outputs
        """
        
        if inputs.__class__ in [Output, Input]:
            inputs = [inputs]
        
        result:list[Output] = []

        # only print "parent" steps to reduce logs
        if hasattr(self, 'step') or hasattr(self, 'steps'):
            Logger().log(f'running step: {self.to_rich_str()}')
        
        for current_input in inputs:
            if self.suitable(current_input):
                output = self.from_cache(current_input)
                if not output:
                    output = func(self, current_input, callback=callback)
                    self.add_cache(current_input, output)
                else:
                    callback(self) # Call callback manually because we used cache
                    
                self.track_output(output)
            else:
                output = current_input    
            
                
            result = result + ([output] if type(output) in [Output, Input] else output)

        self.output = result        
        if callback:
            callback(self)
            
        return result
    return runner_wrapper
