class Explanation:
    def __init__(self, step, processings: list[str]) -> None:
        self.step = step
        self.processings = processings

    @property
    def description(self) -> str:
        conf = { k: v['value'] for k, v in self.step.current_configuration.items() }

        return self.step.description.format(**conf)
    
    def add_processing(self, processing: str) -> None:
        self.processings.append(processing)
    
    def to_markdown(self) -> str:
        if len(self.processings) > 0:
            confs = '\n'.join([ f'| **{k}** | {v["description"]} | {v["value"]} |' for k, v in self.step.current_configuration.items() ])
            entries = '\n'.join([ f' - {p}' for p in self.processings ])

            return f'''
## {self.step.name}
**{self.description}**

{f"""
### Configuration
| Name | Description | Value |
| ---- | ----------- | ----- |
{confs}
""" if len(confs) > 0 else ""}

{f"""
### Processings
{entries}
""" if len(entries) > 0 else ""}
            '''
