"""
[STEP] Linear Discriminant Analysis
"""
import textwrap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from ....predictor import Predictor
from ....dataset import Dataset
from ....candidate import Candidate
from ....decorators.all import is_step

@is_step('predictor', 'tabular', 'classifier')
class ActLinearDiscriminantAnalysis(Predictor):
    """
    [STEP] Linear Discriminant Analysis
    """
    name = "Linear Discriminant Analysis"
    _description = textwrap.dedent('''\
        LinearDiscriminantAnalysis is a machine learning algorithm that finds
        a linear combination of features that maximizes the separation between
        classes for classification tasks.''')
    _description_long = textwrap.dedent('''\
        LinearDiscriminantAnalysis is a type of dimensionality reduction
        algorithm that finds a linear combination of features that maximizes the
        separation between classes for classification tasks.
        It works by calculating the within-class and between-class scatter matrices,
        and then finding the directions in the feature space that maximize the ratio of
        the between-class scatter to the within-class scatter.''')
    _usage = "Use when you need a fast linear classifier; simpler than ActCatBoost or ActExtraTreesClassifier. Applicable to tabular numeric data with binary or multiclass labels. Avoid when strong nonlinear patterns or categorical splits favor ActDecisionTreeClassifier."
    refs = [
        {
            'year': 1936,
            'name': 'The Use of Multiple Measurements in Taxonomic Problems',
            'authors': [
                'Sir Ronald Aylmer Fisher'
            ],
            'doi': 'https://doi.org/10.1111/j.1469-1809.1936.tb02137.x',
            'publisher': 'Annals of Eugenics Vol.7 page 179--188'
        }
    ]
    def __init__(self):
        self.configuration = {
            'tol': {
                'description': textwrap.dedent('''\
                    Absolute threshold for a singular value of X to be
                    considered significant, used to estimate the rank of X.'''),
                'default': 0.0001,
                'range': [1e-05, 0.1]
            }
        }
        self.model: LinearDiscriminantAnalysis = None

    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        self.model = LinearDiscriminantAnalysis(**self.passthrough_parameters())

        self.model.fit(dataset.X, dataset.y)

        return self

    def suitable(self, dataset: Dataset) -> bool:
        return dataset.type_of_target in ['binary', 'multiclass']

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.5 # neutral
