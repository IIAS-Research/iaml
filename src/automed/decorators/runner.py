"""
    Step.run() decorator.
"""
from ..candidate import Candidate
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
    def runner_wrapper(self, candidates:list[Candidate],
                        callback:callable=None
                        ) -> list[Candidate]:
        """Wrapping decorated method

        Returns:
            list[Candidate]: All generated candidates
        """
        
        if candidates.__class__ in [Candidate]:
            candidates = [candidates]
            
        print("Begin step", self)
        print("input cand", candidates)
        
        result:list[Candidate] = []

        # only print "parent" steps to reduce logs
        if hasattr(self, 'step') or hasattr(self, 'steps'):
            Logger().log(f'running step: {self.to_rich_str()}')
        
        for current_candidate in candidates:
            if self.suitable(current_candidate):
                candidate = self.from_cache(current_candidate)
                if not candidate:
                    candidate = func(self, current_candidate, callback=callback)
                    self.add_cache(current_candidate, candidate)
                else:
                    callback(self) # Call callback manually because we used cache
                    
                self.track_candidate(candidate)
            else:
                candidate = current_candidate    
            
                
            result = result + ([candidate] if type(candidate) in [Candidate] else candidate)

        self.candidate = result        
        if callback:
            callback(self)
            
        print("output cand", result)
        
        return result
    return runner_wrapper
