from .step import *

@isStep('actionable')
class Actionable(Step):
    output = None
    def __init__(self, *args, **kw):
        self.configurations = [{
            'ratio' : {
                'description': 'Description of the parameter\'s role',
                'default': 0.8 # Default value
            },
            'random_state': {
                'description': 'Description of the parameter\'s role',
                'default': 12
            }
        }]