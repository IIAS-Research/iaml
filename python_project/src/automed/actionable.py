from step import *

@isStep('actionable')
class Actionable(Step):
    output_dataset = None
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
    