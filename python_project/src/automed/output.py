from dataclasses import dataclass

@dataclass
class Output:
    dataset = None
    metric = None
    model = None
    
    def save(self):
        pass