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
    
    # Example :
        # configuration = {
        #     'ratio' : {
        #         'description': 'Description of the parameter\'s role',
        #         'default': 0.8 # Default value
        #     },
        #     'random_state': {
        #         'description': 'Description of the parameter\'s role',
        #         'default': 12
        #     }
        # }
    
    
    ################
    # Configurable #
    ################
    #
    # A actionable step can be configured by the user.
    # For example, we can configure learning rate of a machine learning step.
    #  
    # Each parameters have a name, a description and a default value. Default value can be fixed or computed based on dataset
    
    
    
    ############
    # Evaluate #
    ############
    #
    # Base on the dataset (or not) evaluate usefulness of this actionable.
    # Result is a value between 0 and 1. 0 stand for not useful
    #
    def evaluate(self, dataset):
        return 0  
    
    #######
    # RUN #
    #######
    #
    # Execute actionable on dataset with args (configurable and/or default_config). 
    #
    
    
    ##########
    # Result #
    ##########