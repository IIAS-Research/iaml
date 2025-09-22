"""TimedPoolExecutor will run *max_workers* new process and will send them actions
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
import multiprocess.managers
import multiprocess.process

from .logger import Logger
from .core_dispatcher import CoreDispatcher


class TerminatedError(RuntimeError):
    """Custom RuntimeError
    Raised when we try to run a job in a stopped executor
    """


def process_daemon(
    to_run_queue: multiprocess.Queue,
    queue: multiprocess.Queue,
    error_queue: multiprocess.Queue,
    finally_queue: multiprocess.Queue) -> None:
    """Will be run by TimedPoolExecutor -> Daemon process able to handle actions

    :param multiprocess.Queue to_run_queue: List of action to run
    :param multiprocess.Queue queue: Queue used to send result
    :param multiprocess.Queue error_queue: Queue used to raise errors
    :param multiprocess.Queue finally_queue: Queue used for every run (success or fail).
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
    """TimedPoolExecutor will run *max_workers* new process and will send them actions
    to run.
    Compare to ProcessPoolExecutor, this one allow us to kill process quickly after timeout
    
    :param int, optional max_workers: Maximum number of parallel workers
    :param callable, optional callback: Function call when the worker is done
    :param bool, optional sliding_stages: Wait for all workers to end, or not
    :param bool, optional debug: Are we in debug mode ?
    """
    def __init__(
        self,
        max_workers: int = None,
        callback: callable = None,
        sliding_stages: bool = True,
        debug: bool = False) -> None:
        """Initialize a TimedPoolExecutor
        """
        self.max_workers: int = min(max_workers, multiprocess.cpu_count())
        """Maximum number of workers allowed to work in parallel"""

        self.debug: bool = debug
        """If true, task will be done without using any process. Easier to debug"""

        self.stop_flag: bool = False
        """Used to stop thread"""

        self.sliding_stages: bool = sliding_stages
        """If True, don't wait for all workers to end, leaving empty cpu cores"""

        # Daemon THREAD (& not Process)
        self.main_daemon: threading.Thread = None
        """Main runnng thread with a infinite loop to catch results of sub process"""

        self.daemons_collectors: list[threading.Thread] = None
        """List of running daemons"""

        self.manager: multiprocess.Manager = multiprocess.Manager()
        """Manager object handling multiprocess Queues"""

        self.to_run_queue: multiprocess.Manager.Queue = self.manager.Queue()
        """Queue used to exchange data with sub process"""

        self.error_queue: multiprocess.Manager.Queue  = self.manager.Queue()
        """Queue used to exchange data with sub process"""

        self.result_queue: multiprocess.Manager.Queue  = self.manager.Queue()
        """Queue used to exchange data with sub process"""

        self.finally_queue: multiprocess.Manager.Queue  = self.manager.Queue()
        """Queue used to exchange data with sub process"""

        self.callbacks: list[callable] = [callback]
        """Method to call after each run"""

        self.results: list = []
        """List of all result since last reset"""

        self.submit_count: int = 0
        """Count -> Help TimedPoolExecutor to know if everything is finished"""

        self.finished_run: int = 0
        """Count -> Help TimedPoolExecutor to know if everything is finished"""

        self.process: list[multiprocess.Process] = []
        """List of sub processes"""

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

        CoreDispatcher().affiliate(
            [process.pid for process in self.process],
            core_number=self.max_workers)

        self.__run_daemon() # Run the daemon THREAD

        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, lambda *_: self.shutdown())
            signal.signal(signal.SIGTERM, lambda *_: self.shutdown())

    def __del__(self):
        """When delete -> TimedPoolExecutor kill all these daemons
        """
        self.shutdown()

    def shutdown(self) -> None:
        """Shutdown TimedPoolExecutor : Kill subprocess and thread
        """
        self.stop_flag = True # Main daemon thread will kill process
        self.main_daemon.join()

    def __collect_results(self) -> None:
        """Collect results from queues and run callback
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
        """Collect and print error from error_queue"""
        while True:
            error, callback_id = self.error_queue.get()

            if error == 'stop':
                break

            Logger().error("Error in a subprocess : ", error)
            self.callbacks[callback_id](None)

    def __keep_running(self) -> None:
        """Daemon THREAD process. Infinite loop to catch results & errors"""
        while True:
            if self.stop_flag:
                self.error_queue.put(("stop", None)) # Gentilly ask thread to stop
                self.result_queue.put(("stop", None)) # Gentilly ask thread to stop
                self.finally_queue.put("stop") # Gentilly ask thread to stop

                for process in self.process:
                    process.kill()

                # empty task queue
                while not self.to_run_queue.empty():
                    self.to_run_queue.get()

                self.manager.shutdown()

                break

            Logger().print_queue()
            time.sleep(0.5)

    def __run_daemon(self) -> None:
        """Start the daemon THREAD"""
        self.stop_flag = False
        if not self.main_daemon or not self.main_daemon.is_alive():
            self.main_daemon = threading.Thread(target=self.__keep_running)
            self.main_daemon.start()

            self.__run_collectors()

    def __run_collectors(self) -> None:
        """Start collector daemons"""
        self.daemons_collectors = [
            threading.Thread(target=self.__print_errors),
            threading.Thread(target=self.__collect_results),
            threading.Thread(target=self.__collect_finally)]

        for collector in self.daemons_collectors:
            collector.start()

    def submit(self, target: callable, *args, **kwargs) -> None:
        """Submit a new task to sub process

        :param callable target: Method to run
        :param Tuple, optional args: parameters passed to the callable
        :param Dict, optional kwargs: parameters passed to the callable
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
        """Are all the submitted tasks finished?
        
        :return: True if all tasks are finished
        """
        return self.finished_run >= self.submit_count

    def reset(self):
        """Reset all queues, callback, results, etc. 
        Allow to reuse this instance of TimedPoolExecutor without restarting subProcess
        """
        if not self.sliding_stages:
            self.callbacks = [self.callbacks[-1]]
            self.submit_count = 0
            self.finished_run = 0

        self.results = []

    def __join_collectors(self):
        """Join collector thread.
        Stop and start thread, used when we want to sync with thread to collect all data 
        """

        # Stop and join collector
        self.error_queue.put(("stop", None))
        self.result_queue.put(("stop", None))
        self.finally_queue.put("stop")

        for collector in self.daemons_collectors:
            collector.join()

        # Restart collectors
        self.__run_collectors()

    def set_callback(self, callback: callable) -> None:
        """Set the method call to when a task finish

        :param callable callback: callback method
        """
        self.callbacks.append(callback)

    def join(self, timeout: int, reset: bool = True) -> list:
        """Wait until all the task are finished or timeout is reach
        If timeout is reach -> Remaining tasks will be kill without sending results

        :param int timeout: Maximum seconds to wait
        :param bool, optional reset: Reset the instance after join(). Defaults to True.

        :return: All finished task results
        """
        start_time = time.monotonic()
        def remain_time():
            return max(0, int(timeout - (time.monotonic() - start_time)))

        def slide():
            try:
                is_empty = self.to_run_queue.empty()
            except BrokenPipeError:
                is_empty = True

            return self.sliding_stages \
                and (
                    is_empty # submit queue is empty
                    and (
                        self.submit_count - self.finished_run <= self.max_workers/2
                        # At least half of the worker is free
                        )
                    and self.results # We have got at least one result
                )

        while not self.__finished() and remain_time() and not slide():
            time.sleep(0.5)

        # Join collector thread, just to be sure we have collected all data
        self.__join_collectors()

        results = self.results # Save before reset!

        if reset:
            self.reset()

        return results
