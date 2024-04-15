"""
Singleton used by Automed to generate nice logs
"""
import rich.console
import rich.progress
import multiprocess
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
        self.log_queue = multiprocess.Queue()
        
    def set_quiet(self, value:bool) -> None:
        """
        Set a new quiet value
        
        Args:
            value (bool): New quiet value
        """
        self.quiet = value

    def log(self, *text: list[str], force:bool=False) -> None:
        """
        Show text in console
        """
        if not(self.quiet) or force:
            if multiprocess.current_process().name == 'MainProcess':
                self.console.log(*text)
            else:
                self.log_queue.put([f"[{multiprocess.current_process().name}]", *text])

    def print_queue(self):
        """
        Print all texts from subProcess
        """
        if multiprocess.current_process().name == 'MainProcess':
            while not self.log_queue.empty():
                self.log(*self.log_queue.get(), force=True)
