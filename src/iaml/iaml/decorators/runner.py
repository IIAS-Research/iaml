"""
    Step.run() decorator.
"""
from ..logger import Logger

def runner(func) -> callable:
    """
    runner MUST decorate your run() method. It you manage every boring things for you.
        - Store results in cache
        - Send information to Destroyers
        - Put results in good shape
        - Call callback method
        - And maybe more

    Args:
        func (callable): decorated method

    Returns:
        callable: edited method
    """
    def runner_wrapper(self, candidates:list['Candidate'],
                        callback:callable=None
                        ) -> list['Candidate']:
        """Wrapping decorated method

        Returns:
            list[Candidate]: All generated candidates
        """
        from ..candidate import Candidate # Avoid circular import
        
        
        if candidates.__class__ in [Candidate]:
            candidates = [candidates]
        
        result:list['Candidate'] = []

        # only print "parent" steps to reduce logs
        if hasattr(self, 'step') or hasattr(self, 'steps'):
            Logger().log(f'running step: {self.to_rich_str()}')
        
        for current_candidate in candidates:
            if self.suitable(current_candidate.dataset) and self.enable:
                candidate = self.from_cache(current_candidate)
                if not candidate:
                    candidate = func(self, current_candidate, callback=callback)
                    self.add_cache(current_candidate, candidate)
                else:
                    callback(self) # Call callback manually because we used cache
            else: # If the step is disabled or not suitable for the dataset, do nothing
                candidate = current_candidate    
            
                
            result = result + ([candidate] if type(candidate) in [Candidate] else candidate)

        self.candidate = result        
        if callback:
            callback(self)
        
        return result
    return runner_wrapper
