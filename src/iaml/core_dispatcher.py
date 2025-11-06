"""Singleton used by IAML to dispatch cores to process"""
import threading
import warnings
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

        self.manager: multiprocess.Manager | None = None
        """Manager used to dispatch cores to process (None when not available)."""

        self._lock: threading.RLock | multiprocess.managers.SyncManager.RLockProxy
        """Lock protecting access to booked cores."""

        self.books: list | multiprocess.managers.ListProxy
        """Booked cores registry."""

        try:
            self.manager = multiprocess.Manager()
            self._lock = self.manager.RLock()
            self.books = self.manager.list([])
        except Exception as exc:  # pylint: disable=broad-except
            warnings.warn(f"CoreDispatcher fallback to local mode (manager start failed: {exc!r})")
            self.manager = None
            self._lock = threading.RLock()
            self.books = []

    def reset_books(self) -> None:
        """Set all cpu cores as available"""
        with self._lock:
            if hasattr(self.books, "clear"):
                self.books.clear()
            else:
                self.books[:] = []

    @property
    def available_cores(self) -> set[int]:
        """All cores minus booked ones
        
        :return: Set of available cpu cores
        """
        avail = set(self.__all_cores)
        with self._lock:
            for book in list(self.books):
                avail -= set(book['cores'])

        return avail

    def __book_cpu(self, pids: set[int], number: int) -> None:
        """Affiliate CPU cores to process

        :param set[int] pids: pids to set affinity with.
        :param int number: Number of cores to book.
        :raise RuntimeError: Not enough CPU cores available.
        """
        self.__free_cores()
        with self._lock:
            if len(self.available_cores) < number:
                raise RuntimeError('Not enough CPU cores available')

            to_book = list(self.available_cores)[:number]
            entry = {
                'pids': pids,
                'cores': to_book
            }
            self.books.extend([entry])

        for pid in pids:
            process = psutil.Process(pid)
            try:
                process.cpu_affinity(to_book)
            except (psutil.AccessDenied, AttributeError, NotImplementedError):
                Logger().warning(
                    'Unable to set CPU affinity for process '
                    f'{pid}. Continuing without affinity control.'
                )

    def affiliate(self, pids: list[int], core_number: int = 1) -> None:
        """Run process with CPU affinity

        :param list[int] pids:  Process to affiliate with CPU cores.
        :param int, optional core_number: number of process to run 
            (each process run the same target). Defaults to 1.
        """
        try:
            test_process = psutil.Process(pids[0])
        except psutil.Error:
            Logger().warning('Unable to inspect process for CPU affinity; skipping affinity control.')
            return

        if hasattr(test_process, 'cpu_affinity'):
            try:
                self.__book_cpu(pids, core_number)
            except RuntimeError as exc:
                Logger().warning(str(exc))
        else:
            Logger().warning(
                'Your OS doesn\'t support CPU affinity. We are not able to control CPU cores access'
            )

    def __free_cores(self) -> None:
        """Free cores that are not used anymore"""
        with self._lock:
            new_books = []
            for book in self.books:
                if any(psutil.pid_exists(pid) and psutil.Process(pid).is_running()
                    for pid in book['pids']):
                    new_books.append(book)

            self.reset_books()
            self.books.extend(new_books)

CoreDispatcher() # Run it a first time to init shared objects
