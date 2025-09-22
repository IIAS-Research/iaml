"""Singleton used by IAML to dispatch cores to process"""
import multiprocess
import multiprocess.managers
import psutil
from .meta_singleton import MetaSingleton
from .logger import Logger


class CoreDispatcher(metaclass=MetaSingleton):
    """Singleton used by IAML to dispatch cores to process
    
    :param tuple, optional \\*args: Additional parameters.
    :param dict, optional \\**kwargs: Additional parameters.
    """
    def __init__(self, *args, **kwargs) -> None:  #pylint: disable=unused-argument
        self.__all_cores: set[int] = set(range(psutil.cpu_count(logical=True)))
        """Set of all cores available"""

        self.manager: multiprocess.Manager = multiprocess.Manager()
        """Manager used to dispatch cores to process"""

        self.books: multiprocess.Manager.list = self.manager.list([])
        """Booked cores"""

    def reset_books(self) -> None:
        """Set all cpu cores as available"""
        while len(self.books) > 0:
            self.books.pop(0)

    @property
    def available_cores(self) -> set[int]:
        """All cores minus booked ones
        
        :return: Set of available cpu cores
        """
        avail = self.__all_cores
        for book in self.books:
            avail -= set(book['cores'])

        return avail

    def __book_cpu(self, pids: set[int], number: int) -> None:
        """Affiliate CPU cores to process

        :param set[int] pids: pids to set affinity with.
        :param int number: Number of cores to book.
        :raise RuntimeError: Not enough CPU cores available.
        """
        self.__free_cores()
        with self.manager.Lock():
            if len(self.available_cores) < number:
                raise RuntimeError('Not enough CPU cores available')

            to_book = list(self.available_cores)[:number]
            self.books.extend([{
                'pids': pids,
                'cores': to_book
                }])

        for pid in pids:
            process = psutil.Process(pid)
            process.cpu_affinity(to_book)

    def affiliate(self, pids: list[int], core_number: int = 1) -> None:
        """Run process with CPU affinity

        :param list[int] pids:  Process to affiliate with CPU cores.
        :param int, optional core_number: number of process to run 
            (each process run the same target). Defaults to 1.
        """
        test_process = psutil.Process(pids[0])
        if hasattr(test_process, 'cpu_affinity'):
            self.__book_cpu(pids, core_number)
        else:
            Logger().warning(
                'Your OS doesn\'t support CPU affinity. We are not able to control CPU cores access'
            )

    def __free_cores(self) -> None:
        """Free cores that are not used anymore"""
        with self.manager.Lock():
            new_books = []
            for book in self.books:
                if any(psutil.pid_exists(pid) and psutil.Process(pid).is_running()
                    for pid in book['pids']):
                    new_books.append(book)

            self.reset_books()
            self.books.extend(new_books)

CoreDispatcher() # Run it a first time to init shared objects
