"""
    Class decorator -> Must decorate each Step inheritance
"""
from ..step import Step

def is_step(*tags) -> callable:
    """
    is_step is needed to declare new Step.
    With the Step inheritance, it will setup everything to make it work smoothly
    Tags -> Your Step will be attached to these tags. 
        tags are use to easily include Step into Pipeline
    """
    def step_wrapper(cls) -> Step:
        """
        Declare the new step to Automed
        Add call to Step.__init__() so the Sub Step developer have one to care about this
        Returns:
            Step: Edited class
        """
        Step.available_steps[cls] = tags # Declare your Step to AutoMed
        
        # Help Python to find parent class
        __class__ = cls # pylint: disable=unused-variable
        
        
        initial_init = cls.__init__ # Keep the __init__ you have created
        def __init__(self, *args, **kw):
            if cls != Step:
                super().__init__(*args, **kw) # All parent constructor 
            
            initial_init(self, *args, **kw) # Run your __init__
            self.default_configuration() # Setup default configuration
            
        cls.__init__ = __init__ # Replace your init
            
        return cls
        
    return step_wrapper
