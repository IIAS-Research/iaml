"""
    Singleton used by IAML to dispatch cores to process
"""
import psutil
import time
import threading
from .meta_singleton import MetaSingleton
from .logger import Logger

class CoreDispatcher(metaclass=MetaSingleton):
    """
    Singleton used by IAML to dispatch cores to process
    """
    def __init__(self, *args, **kwargs) -> None:
        self.__all_cores = set(range(psutil.cpu_count(logical=True)))
        self.__books = []
        self.daemon = None
        
        

    def reset(self) -> None:
        """
        Set all cpu cores as available
        """
        self.__books = []
                
    @property
    def available_cores(self) -> set[int]:
        """
        All cores minus booked ones
        """
        avail = self.__all_cores
        for book in self.__books:
            avail -= book['cores']
            
        return avail
    
    def __run_daemon(self):
        if not self.daemon or not self.daemon.is_alive():
            self.daemon = threading.Thread(target=self.__keep_running)
            self.daemon.start()
    
    def __book_cpu(self, pids:set[int], number:int) -> None:
        """
        Affiliate CPU cores to process

        Args:
            pids (set[int]): pids to set affinity with
            number (int): Number of cores to book

        Raises:
            RuntimeError: Not enough CPU cores available
        """
        
        if len(self.available_cores) < number:
            raise RuntimeError('Not enough CPU cores available')
        
        to_book = list(self.available_cores)[:number]
        self.__books.append({
            'pids': pids,
            'cores': to_book
        })
        
        print("BOOK !", to_book)
        
        for pid in pids:
            process = psutil.Process(pid)
            process.cpu_affinity(to_book)
        
    def affiliate(self, 
            pids:list[int], 
            core_number=1) -> None:
        """
        Run process with CPU affinity

        Args:
            pids:list[int]: Process to affiliate with CPU cores. Defaults to 1.
            process_number (int, optional): number of process to run 
                (each process run the same target). Defaults to 1.

        """
        
        test_process = psutil.Process(pids[0])
        if hasattr(test_process, 'cpu_affinity'):
            self.__book_cpu(pids, core_number)
            self.__run_daemon()
        else:
            Logger().warning('Your OS doesn\'t support CPU afifnity. We are not able to control CPU cores access')
    
    def __keep_running(self):
        """
        Daemon THREAD process. Infinite loop to free cpu core when process are terminated
        """
        while True:
            if not self.__books:
                break

            new_books = []
            for book in self.__books:
                if any(psutil.Process(pid).is_running() for pid in book['pids']):
                    new_books.append(book)

            self.__books = new_books
            
            time.sleep(0.5)
    
        