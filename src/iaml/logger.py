"""
Singleton used by Automed to generate nice logs
"""
from enum import Enum
import rich.console
import rich.progress
import multiprocess
from multiprocess.queues import Empty

from .meta_singleton import MetaSingleton

class LogType(Enum):
    """
    Types of log messages
    """
    INFO = 1
    WARNING = 2
    ERROR = 3

class Logger(metaclass=MetaSingleton):
    """
    Singleton used by Automed to generate nice logs
    """
    def __init__(self, verbose:int = 1) -> None:
        """
        Args:
            verbose (int, optional):
                0 -> No print
                1 -> Progressbar only
                2 -> progressbar + infos
                3 -> progressbar + infos + warning
                4 -> progressbar + infos + warning + error
                -1 -> error and progressbar
                Default = 1
        """
        self.console:rich.console = rich.console.Console(log_path=False)
        self.progress:rich.progress = rich.progress.Progress(console=self.console)
        self.verbose:int = verbose

        self.log_queue = multiprocess.Queue()
        self.callback = None

    @property
    def verbose(self) -> int:
        """
            Logger's verbosity
                0 -> No print
                1 -> Progressbar only
                2 -> progressbar + infos
                3 -> progressbar + infos + warning
                4 -> progressbar + infos + warning + error
                -1 -> error and progressbar
        """
        return self.__verbose
    
    @verbose.setter
    def verbose(self, value:int) -> int:
        self.__verbose = max(min(value, 4), -1)

        return self.__verbose
    
    def __log(self, log_type, *text: list[str]) -> None:
        """
        Show text in console
        """
        if multiprocess.current_process().name == 'MainProcess':
            self.__print(*text)
        else:
            self.log_queue.put((log_type, f"[{multiprocess.current_process().name}]", *text))

    def __print(self, *text: list[str]) -> None:
        if self.callback is not None:
            self.callback(*text)
        else:
            self.console.log(*text)

    def info(self, *text: list[str]) -> None:
        """
        Show info text in console
        """
        if self.verbose > 1:
            self.__log(LogType.INFO, *text)
            
    def warning(self, *text: list[str]) -> None:
        """
        Show warning text in console
        """
        if self.verbose > 2:
            self.__log(LogType.WARNING, *text)

    def error(self, *text: list[str]) -> None:
        """
        Show info text in console
        """
        if self.verbose > 3 or self.verbose == -1:
            self.__log(LogType.ERROR, *text)

    def set_callback(self, callback: callable) -> None:
        """
        Sets a custom callback to this logger so that messages can be intercepted.

        Args:
            callback (callable, optional):
                Callback to send logs to.
        """
        self.callback = callback

    def print_queue(self):
        """
        Print all texts from subProcess
        """
        while not self.log_queue.empty():
            try:
                messages = self.log_queue.get(block=False)
                self.__print(*messages)
            except Empty:
                break
