"""
    TimedPoolExecutor will run *max_workers* new process and will send them actions
    to run.
    Compare to ProcessPoolExecutor, this one allow us to kill process quickly after timeout
"""

import random
import signal
import time
import traceback
import warnings
import threading
import multiprocess
import multiprocess.process

from .logger import Logger
from .worker_manager import WorkerManager


class TerminatedError(RuntimeError):
    """Custom RuntimeError
    Raised when we try to run a job in a stopped executor
    """


def process_daemon(
    to_run_queue:multiprocess.Queue,
    queue:multiprocess.Queue,
    error_queue:multiprocess.Queue,
    finally_queue:multiprocess.Queue):
    """
    Will be run by TimedPoolExecutor -> Daemon process able to handle actions

    Args:
        to_run_queue (multiprocess.Queue): List of action to run
        queue (multiprocess.Queue): Queue used to send result
        error_queue (multiprocess.Queue): Queue used to raise errors
        finally_queue (multiprocess.Queue): Queue used for every run (success or fail).
            Used to count number of ran actions
    """
    result = None
    
    time.sleep(random.random()) # Weird thing to un-sync the threads
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        while True:
            value = to_run_queue.get()
            
            method, args, kwargs, callback_id = value
            try:
                result = method(*args, **kwargs)
                queue.put((result, callback_id))
            except Exception:  # pylint: disable=broad-exception-caught
                error_queue.put((traceback.format_exc(), callback_id))
            finally:
                finally_queue.put(1)
            

class TimedPoolExecutor:  # pylint: disable=too-many-instance-attributes
    """
        TimedPoolExecutor will run *max_workers* new process and will send them actions
        to run.
        Compare to ProcessPoolExecutor, this one allow us to kill process quickly after timeout
    """
    def __init__(self,
                max_workers:int=None,
                callback:callable=None,
                sliding_stages:bool=True,
                debug:bool=False):
        self.max_workers = min(max_workers, multiprocess.cpu_count())
        self.debug = debug # If true, task will be done without using any process.  Easier to debug
        self.stop_flag:bool = False # Used to stop thread
        self.sliding_stages = sliding_stages
        
        # Daemon THREAD (& not Process) with a infinite loop to catch results of sub process
        self.daemon = None
        
        # Queue used to exchange data with sub process
        self.manager = multiprocess.Manager()
        self.to_run_queue = self.manager.Queue()
        self.error_queue = self.manager.Queue()
        self.result_queue = self.manager.Queue()
        self.finally_queue = self.manager.Queue()
        
        # Method to call after each run
        self.callbacks = [callback]
        
        # List of all result since last reset
        self.results = []
        
        # Count -> Help TimedPoolExecutor to know if everything is finished
        self.submit_count = 0
        self.finished_run = 0
        
        # List of sub process
        self.process: list[multiprocess.Process] = []
        
        # Create and start sub process (will only wait until first submit)
        for _ in range(max_workers):
            self.process.append(
                multiprocess.Process( # pylint: disable=not-callable
                    target=process_daemon,
                    args=[self.to_run_queue,
                        self.result_queue,
                        self.error_queue,
                        self.finally_queue
                    ]
                )
            )
            self.process[-1].start()
            
        self.__run_daemon() # Run the daemon THREAD

        signal.signal(signal.SIGINT, lambda *_: self.shutdown())
        signal.signal(signal.SIGTERM, lambda *_: self.shutdown())
        
    def __del__(self):
        """
        When delete -> TimedPoolExecutor kill all these daemons
        """
        self.shutdown()
    
    def shutdown(self) -> None:
        """
            Shutdown TimedPoolExecutor : Kill subprocess and thread
        """
        self.stop_flag = True # Main daemon thread will kill process
        self.daemon.join()

    def __collect_results(self) -> None:
        """
            Collect results from queues and run callback
        """
        while True:
            result, callback_id = self.result_queue.get()

            if isinstance(result, str) and result == 'stop':
                break

            if callback_id and callable(self.callbacks[callback_id]):
                self.callbacks[callback_id](result)

            Logger().info(str(result))
            self.results.append(result)

    def __collect_finally(self) -> None:
        while self.finally_queue.get() != "stop":
            self.finished_run += 1
        
    def __print_errors(self) -> None:
        """
            Collect and print error from error_queue
        """
        while True:
            error, callback_id = self.error_queue.get()
            
            if error == 'stop':
                break
            
            Logger().error("Error in a subprocess : ", error)
            self.callbacks[callback_id](None)
    
    def __keep_running(self) -> None:
        """
        Daemon THREAD process. Infinite loop to catch results & errors
        """
        while True:
            if self.stop_flag:
                self.error_queue.put(("stop", None)) # Gentilly ask process to stop
                self.result_queue.put(("stop", None)) # Gentilly ask process to stop
                self.finally_queue.put("stop") # Gentilly ask process to stop

                for process in self.process:
                    process.kill()

                # empty task queue
                while not self.to_run_queue.empty():
                    self.to_run_queue.get()

                self.manager.shutdown()
                WorkerManager().executor.shutdown()
                
                break

            Logger().print_queue()
            time.sleep(0.5)
    
    def __run_daemon(self) -> None:
        """
        Start the daemon THREAD
        """
        self.stop_flag = False
        if not self.daemon or not self.daemon.is_alive():
            self.daemon = threading.Thread(target=self.__keep_running)
            self.daemon.start()

            threading.Thread(target=self.__print_errors).start()
            threading.Thread(target=self.__collect_results).start()
            threading.Thread(target=self.__collect_finally).start()
                
        
    def submit(self, target:callable, *args, **kwargs) -> None:
        """
        Submit a new task to sub process

        Args:
            target (callable): Method to run
        """
        if self.stop_flag:
            raise TerminatedError("Job submission failed: Executor is currently \
                shutdown and cannot accept new tasks.")

        if self.debug:
            self.result_queue.put((target(*args, **kwargs), len(self.callbacks)-1))
        else:
            self.to_run_queue.put((target, args, kwargs, len(self.callbacks)-1))
            self.submit_count += 1
        
    def __finished(self) -> bool:
        """
        Are all the submitted tasks finished?
        """
        return self.finished_run >= self.submit_count
    
    def reset(self):
        """
        Reset all queues, callback, results, etc. 
        Allow to reuse this instance of TimedPoolExecutor without restarting subProcess
        """
        if not self.sliding_stages:
            self.callbacks = [self.callbacks[-1]]
            self.submit_count = 0
            self.finished_run = 0
            
        self.results = []
                
    def set_callback(self, callback:callable) -> None:
        """
        Set the method call to when a task finish

        Args:
            callback (callable): callback method
        """
        self.callbacks.append(callback)
        
    def join(self, timeout:int, reset:bool=True) -> list:
        """
        Wait until all the task are finished or timeout is reach
        If timeout is reach -> Remaining tasks will be kill without sending results

        Args:
            timeout (int): Maximum seconds to wait
            reset (bool, optional): Reset the instance after join(). Defaults to True.

        Returns:
            list: All finished task results
        """
        start_time = time.monotonic()
        def remain_time():
            return max(0, int(timeout - (time.monotonic() - start_time)))
        
        def slide():
            try:
                is_empty = self.to_run_queue.empty()
            except BrokenPipeError:
                is_empty = True

            return self.sliding_stages and (is_empty and \
                self.finished_run >= (self.submit_count - self.max_workers/2))
        
        while not self.__finished() and remain_time() and not slide():
            time.sleep(0.5)
        
        results = self.results # Save before reset!
        
        if reset:
            self.reset()
        
        return results
