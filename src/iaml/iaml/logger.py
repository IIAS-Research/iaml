"""
Singleton used by Automed to generate nice logs
"""
from enum import Enum
from typing import List
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
    def __init__(self, verbose: int = 1) -> None:
        """
        Parameters
        ----------
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
        
    @property
    def verbose(self) -> int:
        """
        Returns
        -------
        int
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
    def verbose(self, value: int) -> int:
        """
        Set logger verbosity
        
        Parameters
        ----------
        value : int
            The new logger verbosity
        
        Returns
        -------
        int
            The new logger verbosity
        """
        self.__verbose = max(min(value, 4), -1)
        
        self.console.quiet = self.__verbose == 0
        
        return self.__verbose
    
    def __log(self, log_type: LogType, *text: List[str]) -> None:
        """
        Show text in console
        
        Parameters
        ----------
        log_type : LogType
            Logger type
        text : List[str]
            Text to log
        """
        if multiprocess.current_process().name == 'MainProcess':
            self.console.log(*text)
        else:
            self.log_queue.put((log_type, f"[{multiprocess.current_process().name}]", *text))
                

    def info(self, *text: List[str]) -> None:
        """
        Show info text in console
        
        Parameters
        ----------
        text : List[str]
            Text to log in console
        """
        if self.verbose > 1:
            self.__log(LogType.INFO, *text)
            
    def warning(self, *text: List[str]) -> None:
        """
        Show warning text in console

        Parameters
        ----------
        text : List[str]
            Text to log in console
        """
        if self.verbose > 2:
            self.__log(LogType.WARNING, *text)
            
    def error(self, *text: List[str]) -> None:
        """
        Show info text in console

        Parameters
        ----------
        text : List[str]
            Text to log in console
        """
        if self.verbose > 3 or self.verbose == -1:
            self.__log(LogType.ERROR, *text)
    

    def print_queue(self):
        """
        Print all texts from subProcess
        """
        if multiprocess.current_process().name == 'MainProcess':
            while not self.log_queue.empty():
                try:
                    self.__log(*self.log_queue.get(block=False))
                except Empty:
                    break
