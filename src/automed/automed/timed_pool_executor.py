"""
    TimedPoolExecutor will run *max_workers* new process and will send them actions
    to run.
    Compare to ProcessPoolExecutor, this one allow us to kill process quickly after timeout
"""

import threading
import warnings
import time
import random
import traceback
import multiprocess
from multiprocess.queues import Empty
from .logger import Logger

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
            try:
                value = to_run_queue.get(block=False, timeout=1)
                if isinstance(value, str) and value == "stop":
                    to_run_queue.put("stop")
                    break
                
                method, args, kwargs, callback_id = value
                try:
                    result = method(*args, **kwargs)
                    queue.put((result, callback_id))
                except Exception:  # pylint: disable=broad-exception-caught
                    Logger().log("error", force=True)
                    error_queue.put((traceback.format_exc(), callback_id))
                finally:
                    finally_queue.put(1)
                    
            except Empty:
                time.sleep(0.1)
            

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
        self.to_run_queue = multiprocess.Queue()
        self.error_queue = multiprocess.Queue()
        self.result_queue = multiprocess.Queue()
        self.finally_queue = multiprocess.Queue()
        
        # Method to call after each run
        self.callbacks = [callback]
        
        # List of all result since last reset
        self.results = []
        
        # Count -> Help TimedPoolExecutor to know if everything is finished
        self.submit_count = 0
        self.finished_run = 0
        
        # List of sub process
        self.process = []
        
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
        while not self.result_queue.empty():
            try:
                result, callback_id = self.result_queue.get(block=False, timeout=1)
                Logger().log(str(result))
                if callback_id and callable(self.callbacks[callback_id]):
                    self.callbacks[callback_id](result)
                self.results.append(result)
            except Empty:
                break
        
    def __print_errors(self) -> None:
        """
            Collect and print error from error_queue
        """
        while not self.error_queue.empty():
            try:
                error, callback_id = self.error_queue.get(block=False, timeout=1)
                Logger().log("ERROR IN PROCESS", error, force=True)
                self.callbacks[callback_id](None)
            except Empty:
                break
    
    def __keep_running(self) -> None:
        """
        Daemon THREAD process. Infinite loop to catch results & errors
        """
        while True:
            self.__print_errors()
            self.__collect_results()
            Logger().print_queue()
            time.sleep(0.1)
            
            if self.stop_flag:
                break
        
        # Kill process
        while not self.to_run_queue.empty():
            try:
                self.to_run_queue.get(block=False, timeout=1) # Empty task queue 
            except Empty:
                break
            
        self.to_run_queue.put("stop") # Gentilly ask process to stop
        time.sleep(0.5)
        
        for process in self.process:
            if process.is_alive(): # If process still alive, force stop
                process.kill()
                    
        time.sleep(1)
            
    
    def __run_daemon(self) -> None:
        """
        Start the daemon THREAD
        """
        self.stop_flag = False
        if not self.daemon or not self.daemon.is_alive():
            self.daemon = threading.Thread(target=self.__keep_running)
            self.daemon.start()
                
        
    def submit(self, target:callable, *args, **kwargs) -> None:
        """
        Submit a new task to sub process

        Args:
            target (callable): Method to run
        """
        if self.debug:
            self.result_queue.put((target(*args, **kwargs), len(self.callbacks)-1))
        else:
            self.to_run_queue.put((target, args, kwargs, len(self.callbacks)-1))
            self.submit_count += 1
        
    def __finished(self) -> bool:
        """
        Do all the submit task are finished ?
        """
        while not self.finally_queue.empty():
            try:
                self.finally_queue.get(block=False, timeout=1)
                self.finished_run += 1
            except Empty:
                break
            
        return self.finished_run >= self.submit_count
    
    def reset(self):
        """
        Reset all queues, callback, results, etc. 
        Allow to reuse this instance of TimedPoolExecutor without restarting subProcess
        """
        if not self.sliding_stages:
            for queue in \
                [self.error_queue, self.finally_queue, self.to_run_queue, self.result_queue]:
                while not queue.empty():
                    try:
                        queue.get(block=False, timeout=0.05)
                    except Empty:
                        pass
                    
            self.callbacks = [self.callbacks[-1]]
            self.submit_count = 0
            self.finished_run = 0
            
        self.results = []
                
    def set_callback(self, callback:callable) -> None:
        """
        Set the method to when a task finish

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
            return self.sliding_stages and (self.to_run_queue.empty() and \
                self.finished_run >= (self.submit_count - self.max_workers/2) \
                and self.results)
        
        while not self.__finished() and remain_time() and not slide():
            time.sleep(0.3)
        
            
        self.__collect_results()
        self.__print_errors()
        
        results = self.results # Save before reset !
        
        if reset:
            self.reset()
        
        return results
