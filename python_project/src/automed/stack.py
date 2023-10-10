class Stack:
    def __init__(self, name, description, citations, configuration):
        self.name = name
        self.description = description
        self.citations = citations
        self.configuration = configuration
        
    def __str__(self):
        return self.name