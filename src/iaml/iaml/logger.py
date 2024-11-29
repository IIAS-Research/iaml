"""Singleton used by IAML to generate nice logs"""
from enum import Enum
from typing import List
import rich.console
import rich.progress
import multiprocess
from multiprocess.queues import Empty

from .meta_singleton import MetaSingleton


class LogType(Enum):
    """Types of log messages"""
    INFO = 1
    WARNING = 2
    ERROR = 3


class Logger(metaclass=MetaSingleton):
    """
    Singleton used by IAML to generate nice logs
    
    :param int, optional verbose: Logger verbosity. Default to 1.
    
        * 0 -> No print
        * 1 -> Progressbar only
        * 2 -> progressbar + infos
        * 3 -> progressbar + infos + warning
        * 4 -> progressbar + infos + warning + error
        * -1 -> error and progressbar
    """
    def __init__(self, verbose: int = 1) -> None:
        self.console:rich.console = rich.console.Console(log_path=False)
        """Console used by the logger"""

        self.progress:rich.progress = rich.progress.Progress(console=self.console)
        """Progress bar used by the logger"""

        self.verbose:int = verbose
        """Loagger verbosity"""

        self.log_queue = multiprocess.Queue()
        """Queue used by the logger to handle log from various process"""

    @property
    def verbose(self) -> int:
        """Get logger verbosity
        
        :return: Logger verbosity
        """
        return self.__verbose

    @verbose.setter
    def verbose(self, value: int) -> int:
        """Set logger verbosity
        
        :param int value: The new logger verbosity
        :return: The new logger verbosity
        """
        self.__verbose = max(min(value, 4), -1)
        self.console.quiet = self.__verbose == 0
        return self.__verbose

    def __log(self, log_type: LogType, *text: List[str]) -> None:
        """Show text in console
        
        :param LogType log_type: Logger type
        :param list[str] \\*text: Text to log in console
        """
        if multiprocess.current_process().name == 'MainProcess':
            self.console.log(*text)
        else:
            self.log_queue.put((log_type, f"[{multiprocess.current_process().name}]", *text))

    def info(self, *text: list[str]) -> None:
        """Show info text in console
        
        :param list[str] \\*text: Info Text to log in console
        """
        if self.verbose > 1:
            self.__log(LogType.INFO, *text)

    def warning(self, *text: List[str]) -> None:
        """Show warning text in console

        :param list[str] \\*text: Warning Text to log in console
        """
        if self.verbose > 2:
            self.__log(LogType.WARNING, *text)

    def error(self, *text: List[str]) -> None:
        """Show info text in console

        :param list[str] \\*text: Error Text to log in console
        """
        if self.verbose > 3 or self.verbose == -1:
            self.__log(LogType.ERROR, *text)

    def print_queue(self):
        """Print all texts from subProcess
        """
        if multiprocess.current_process().name == 'MainProcess':
            while not self.log_queue.empty():
                try:
                    self.__log(*self.log_queue.get(block=False))
                except Empty:
                    break
