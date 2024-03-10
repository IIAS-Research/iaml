import numpy as np

from automed.step import Step
from .actionables import Actionable
from .decorators.runner import runner
from .splitter import random_splitter
from .candidate import Candidate
from .dataset import Dataset

class Predictor(Actionable):
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