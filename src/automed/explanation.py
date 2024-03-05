"""
Enables steps to "explain" their processings and prediction results.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .step import Step


class Explanation:
    """
    Enables steps to "explain" their processings and prediction results.
    """
    def __init__(self, step: 'Step', processings: list[str]) -> None:
        self.step = step
        self.processings = processings

    @property
    def description(self) -> str:
        """
        Formats the description of a step with its configuration.

        Returns:
            str: Formatted description.
        """
        conf = { k: v['value'] for k, v in self.step.current_configuration.items() }

        return self.step.description.format(**conf)
    
    def add_processing(self, processing: str) -> None:
        """
        Adds a processing description to the list of processings.

        Args:
            processings (str): Processing description.
        """
        self.processings.append(processing)
    
    def to_markdown(self, processings_limit: int = 100) -> str:
        """
        Renders the explanation as Markdown text.

        Returns:
            str: Markdown text.
        """
        if len(self.processings) > 0:
            confs = '\n'.join([
                f'| **{k}** | {v["description"]} | {v["value"]} |'
                for k, v in self.step.current_configuration.items()
            ])
            entries = '\n'.join([ f' - {p}' for p in self.processings[:processings_limit] ])

            processings_left = len(self.processings) - processings_limit
            print(processings_left)

            return f"""
## {self.step.name}
**{self.description}**

{f'''
### Configuration
| Name | Description | Value |
| ---- | ----------- | ----- |
{confs}
''' if len(confs) > 0 else ""}

{f'''
### Processings
{entries}
{f" - *and **{processings_left}** more processings...*" if processings_left > 0 else ""}
''' if len(entries) > 0 else ""}
            """
        
        return ''
