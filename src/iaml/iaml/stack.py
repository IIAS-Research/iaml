"""
Frozen version of Step created to be stacked in an Candidate
"""
class Stack:
    """
    Frozen version of Step created to be stacked in an Candidate
    """
    def __init__(self, step_class, configuration:dict, step_id:int):
        self.step_class = step_class
        self.configuration:dict = configuration
        self.step_id:int = step_id
        
    def __str__(self) -> str:
        return self.step_class.name
    
    def explain(self) -> str:
        """
        Return explanation string of Step
        """
        return self.step_class.explain(self.configuration)
