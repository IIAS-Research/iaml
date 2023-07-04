from .step import Step, Priority
from .dataset import Dataset

class Pipeline:    
            
    def __init__(self, available_steps: list[Step]):
        self.available_steps = available_steps
        self.first_step = None
        
        
    def run(self, dataset: Dataset) -> None:
        # Create first PipelineStep and run it
        self.first_step = PipelineStep(Step(), self.available_steps, dataset)
        return self.first_step.run()
    
    def outputs(self):
        return self.first_step.get_finals_outputs()
        
            
    

class PipelineStep:
    def __init__(self, step: Step, available_steps: list[Step], input: Dataset):
        self.step = step
        self.disabled_steps = []
        self.available_steps = available_steps
        self.outputs = None
        self.input = input
        self.next_steps = []
        
    def get_finals_outputs(self):
        outputs = []
        if self.next_steps:
            for next in self.next_steps:
                outputs = outputs + next.get_finals_outputs()
        else:
            return self.outputs
        return outputs
    
    def __disable_step(self, step: Step) -> bool:
        if step in self.available_steps:
            self.disabled_steps.append(step)
            self.available_steps.remove(step)
        else:
            return False
        
    def add_next(self, step: Step) -> list[Step]:
        self.next_steps.append(step)
        return self.next_steps
        
    def clean_step(self) -> None:
        self.set_steps([])
        
    def set_next(self, steps: list[Step]) -> list[Step]:
        self.next_steps = steps
        return self.next_steps
    
    def run(self) -> None:
        # print(self.step)
        self.outputs = self.step.run(self.input)
        
        # Must be a list
        if not(isinstance(self.outputs, list)):
            self.outputs = [self.outputs]
        
            
        # print(self.outputs.data)
        for output in self.outputs: # For each Output 
            priority = {}
            for step in self.available_steps:
                curr_prio = step.priorize(output)
                if curr_prio in priority.keys():
                    priority[curr_prio].append(step)
                else:
                    priority[curr_prio] = [step]
        
            if Priority.NEVER in priority.keys():
                for step in priority[Priority.NEVER]:
                    self.__disable_step(step)
                del priority[Priority.NEVER]
            
            if priority:
                for step in priority[min(priority.keys())]:
                    for variante in step.prepare(output):
                        if not step.reusable():
                            self.__disable_step(step)
                            
                        self.next_steps.append(PipelineStep(variante, self.available_steps, input=output))
            
            if self.next_steps:
                self.run_nexts()
            else:
                print('finito')
                # Ne rien return ici. C'est au niveau de pipeline qu'il faudra retrouver les outputs finaux
                # import random
                # rand = random.randint(0,1000)
                # print(rand)
                # output.data.to_csv('output_'+str(rand)+'.csv')
                
        # return self.outputs
                
                    
    def run_nexts(self):
        for next in self.next_steps:
            next.run()
