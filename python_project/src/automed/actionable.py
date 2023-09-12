from .step import *

@isStep('actionable')
class Actionable(Step):
    output = None
    configuration = {
        'ratio' : {
            'description': 'Description of the parameter\'s role',
            'default': 0.8 # Default value
        },
        'random_state': {
            'description': 'Description of the parameter\'s role',
            'default': 12
        }
    }