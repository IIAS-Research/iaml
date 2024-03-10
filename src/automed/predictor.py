"""
Last step of a pipeline -> can make prediction
"""
from .actionables import Actionable
from .decorators.runner import runner
from .candidate import Candidate

class Predictor(Actionable):
    """
    Last step of a pipeline -> can make prediction
    """
    
    @runner
    def run(self, candidate:Candidate, callback:callable=None) -> Candidate:
        """
        Run the step. In "Run" stage, predict does not "fit". Only add himself to pipeline

        Args:
            candidate (Candidate): Candidate informations 

        Returns:
            Candidate: transformed Candidate 
        """
        return candidate.add_to_pipeline(self)
