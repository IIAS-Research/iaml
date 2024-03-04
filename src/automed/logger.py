"""
Singleton used by Automed to generate nice logs
"""
import rich.console
import rich.progress
from .meta_singleton import MetaSingleton

class Logger(metaclass=MetaSingleton):
    """
    Singleton used by Automed to generate nice logs
    """
    def __init__(self, quiet:bool=False) -> None:
        """
        Args:
            quiet (bool, optional): If True -> Show only progress bar. Defaults to False.
        """
        self.console:rich.console = rich.console.Console(log_path=False)
        self.progress:rich.progress = rich.progress.Progress(console=self.console)
        self.quiet:bool = quiet
        
    def set_quiet(self, value:bool) -> None:
        """
        Set a new quiet value
        
        Args:
            value (bool): New quiet value
        """
        self.quiet = value

    def log(self, *text: list[str]) -> None:
        """
        Show text in console
        """
        if not self.quiet:
            self.console.log(*text)
