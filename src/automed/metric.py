class Metric:
    def explain(self):
        return "Here is an explaination of how this metrics work"
    
    # TODO Remove multilabel parameters and split into two metrics. One Metric = Only behavior   
    # Each class must implment its own method
    def compute(self, y, y_pred):
        raise NotImplementedError('Subclass must implement abstract method')
    
    def suitable(self, y):
        return False