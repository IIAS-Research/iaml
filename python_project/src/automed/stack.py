class Stack:
    def __init__(self, step_class, configuration, step_id):
        self.step_class = step_class
        self.configuration = configuration
        self.step_id = step_id
        
    def __str__(self):
        return self.name
    
    def explain(self):
        return self.step_class.explain(self.configuration)