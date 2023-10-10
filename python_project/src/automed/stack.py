class Stack:
    def __init__(self, name, description, citations, configuration, step_id):
        self.name = name
        self.description = description
        self.citations = citations
        self.configuration = configuration
        self.step_id = step_id
        
    def __str__(self):
        return self.name