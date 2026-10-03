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
import queue
import pickle
import multiprocess
import multiprocess.managers
import multiprocess.process

from .logger import Logger
from .core_dispatcher import CoreDispatcher
from .shared_cache import start_cache_manager
from .cache import Cache


class TerminatedError(RuntimeError):
    """Custom RuntimeError
    Raised when we try to run a job in a stopped executor
    """


def process_daemon(
    to_run_queue: multiprocess.Queue,
    queue: multiprocess.Queue,
    error_queue: multiprocess.Queue,
    finally_queue: multiprocess.Queue,
    shared_cache) -> None:
    """Will be run by TimedPoolExecutor -> Daemon process able to handle actions

    :param multiprocess.Queue to_run_queue: List of action to run
    :param multiprocess.Queue queue: Queue used to send result
    :param multiprocess.Queue error_queue: Queue used to raise errors
    :param multiprocess.Queue finally_queue: Queue used for every run (success or fail).
        Used to count number of ran actions
    """
    result = None
    
    Cache().configure(shared_cache)

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
    :param bool, optional sliding_stages: Preserve unfinished tasks between stages.
    :param bool, optional debug: Are we in debug mode ?
    :param float, optional deadline: Absolute global deadline from time.monotonic().
    """
    def __init__(
        self,
        max_workers: int = None,
        callback: callable = None,
        sliding_stages: bool = True,
        debug: bool = False,
        deadline: float | None = None) -> None:
        """Initialize a TimedPoolExecutor
        """
        self.max_workers: int = min(max_workers, multiprocess.cpu_count())
        """Maximum number of workers allowed to work in parallel"""

        self.debug: bool = debug
        """If true, task will be done without using any process. Easier to debug"""
        self._global_deadline = None if deadline == float('inf') else deadline
        self._global_timed_out = False
        self._state_lock = threading.RLock()
        self._workers_lock = threading.RLock()
        self._collector_lock = threading.RLock()
        self._wakeup = threading.Event()
        self._collectors_stopped = False
        self._queues_closed = False

        self.stop_flag: bool = False
        """Used to stop thread"""

        self.sliding_stages: bool = sliding_stages
        """If True, stage boundaries leave running and queued tasks intact."""

        # Daemon THREAD (& not Process)
        self.main_daemon: threading.Thread = None
        """Main runnng thread with a infinite loop to catch results of sub process"""

        self.daemons_collectors: list[threading.Thread] = None
        """List of running daemons"""

        self._mp_capable: bool = True
        """Flag indicating whether multiprocessing primitives are available."""
        self._mp_fallback: bool = False
        """Flag indicating whether we already fell back to sequential execution."""

        try:
            self.manager: multiprocess.Manager = multiprocess.Manager()
        except Exception as exc:  # pylint: disable=broad-except
            warnings.warn(f"TimedPoolExecutor fallback to sequential mode (manager start failed: {exc!r})")
            self.manager = None
            self._mp_capable = False

        if self._mp_capable:
            self.to_run_queue = self.manager.Queue()
            self.error_queue = self.manager.Queue()
            self.result_queue = self.manager.Queue()
            self.finally_queue = self.manager.Queue()
        else:
            self.to_run_queue = queue.Queue()
            self.error_queue = queue.Queue()
            self.result_queue = queue.Queue()
            self.finally_queue = queue.Queue()
        # Queues used to exchange data with subprocesses

        self.cache_manager: multiprocess.managers.BaseManager | None = None
        """Keep a strong reference to the shared cache manager process"""

        self.shared_cache = None
        """Proxy object used by workers to talk to the shared cache"""

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
        
        if not self._mp_capable:
            self.debug = True
            self.max_workers = 1
            self.cache_manager = None
            self.shared_cache = None
            Cache().configure(None)
        else:
            try:
                self.cache_manager, self.shared_cache = start_cache_manager(max_cache_size=500)
                Cache().configure(self.shared_cache)
            except OSError as exc:
                warnings.warn(f"Shared cache disabled (start_cache_manager failed: {exc!r})")
                self.cache_manager = None
                self.shared_cache = None
                Cache().configure(None)

            # Create and start sub process (will only wait until first submit)
            for _ in range(max_workers):
                self.process.append(
                    multiprocess.Process( # pylint: disable=not-callable
                        target=process_daemon,
                        args=[self.to_run_queue,
                            self.result_queue,
                            self.error_queue,
                            self.finally_queue,
                            self.shared_cache
                        ]
                    )
                )
                self.process[-1].start()

            CoreDispatcher().affiliate(
                [process.pid for process in self.process],
                core_number=self.max_workers)

        self.__run_daemon() # Run the daemon THREAD

        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, self._handle_signal)
            signal.signal(signal.SIGTERM, self._handle_signal)

    def __del__(self):
        """When delete -> TimedPoolExecutor kill all these daemons
        """
        self.shutdown()

    def _handle_signal(self, *_):
        """Request shutdown without waiting for locks held by the interrupted thread."""
        self.stop_flag = True

    def shutdown(self) -> None:
        """Shutdown TimedPoolExecutor : Kill subprocess and thread
        """
        self.stop_flag = True # Main daemon thread will kill process
        if hasattr(self, '_wakeup'):
            self._wakeup.set()
        if self.main_daemon:
            current = threading.current_thread()
            if (self.main_daemon is not current
                    and current not in (self.daemons_collectors or ())):
                self.main_daemon.join()

    def _remaining_global_budget(self) -> float:
        """Remaining search time, independent of the current stage."""
        if self._global_timed_out:
            return 0.0
        if self._global_deadline is None:
            return float('inf')
        return max(0.0, self._global_deadline - time.monotonic())

    @property
    def pending_count(self) -> int:
        """Number of submitted tasks not yet completed or cancelled."""
        with self._state_lock:
            return max(0, self.submit_count - self.finished_run)

    def _notify_callback(self, result, callback_id) -> None:
        """A failing user callback must not interrupt result collection."""
        with self._state_lock:
            callback = (self.callbacks[callback_id]
                        if callback_id is not None and callback_id < len(self.callbacks)
                        else None)
        if callable(callback):
            try:
                callback(result)
            except Exception:  # pylint: disable=broad-exception-caught
                Logger().error('Error in an executor callback: ', traceback.format_exc())

    def __collect_results(self) -> None:
        """Collect results from queues and run callback
        """
        while True:
            result, callback_id = self.result_queue.get()

            if callback_id is None and isinstance(result, str) and result == 'stop':
                break

            self._notify_callback(result, callback_id)

            Logger().info(str(result))
            with self._state_lock:
                self.results.append(result)

    def __collect_finally(self) -> None:
        while True:
            item = self.finally_queue.get()
            if item == "stop":
                break
            with self._state_lock:
                self.finished_run += 1

    def __print_errors(self) -> None:
        """Collect and print error from error_queue"""
        while True:
            item = self.error_queue.get()

            if item is None:
                continue

            error, callback_id = item

            if error == 'stop':
                break

            Logger().error("Error in a subprocess : ", error)
            self._notify_callback(None, callback_id)

    def __keep_running(self) -> None:
        """Daemon THREAD process. Infinite loop to catch results & errors"""
        while True:
            if self.stop_flag:
                self._terminate_workers()
                self._drain_queue(self.to_run_queue)
                with self._collector_lock:
                    self.__join_collectors(restart=False)
                    if self.manager is not None:
                        self.manager.shutdown()
                    if self.cache_manager is not None:
                        self.cache_manager.shutdown()
                        self.cache_manager = None
                    self._queues_closed = True

                self.shared_cache = None
                Cache().configure(None)

                break

            remaining = self._remaining_global_budget()
            if remaining <= 0 and not self._global_timed_out:
                self._global_timed_out = True
                self._terminate_workers()
                self._drain_queue(self.to_run_queue)

            Logger().print_queue()
            self._wakeup.wait(min(0.5, remaining) if remaining > 0 else 0.5)
            self._wakeup.clear()

    def __run_daemon(self) -> None:
        """Start the daemon THREAD"""
        self.stop_flag = False
        if not self.main_daemon or not self.main_daemon.is_alive():
            self.main_daemon = threading.Thread(target=self.__keep_running)
            self.main_daemon.start()

            self.__run_collectors()

    def __run_collectors(self) -> None:
        """Start collector daemons"""
        self._collectors_stopped = False
        self.daemons_collectors = [
            threading.Thread(target=self.__print_errors),
            threading.Thread(target=self.__collect_results),
            threading.Thread(target=self.__collect_finally)]

        for collector in self.daemons_collectors:
            collector.start()

    def _drain_queue(self, target_queue: queue.Queue | multiprocess.managers.BaseProxy) -> None:
        """Clear queued tasks without blocking."""
        while True:
            try:
                target_queue.get_nowait()
            except Exception:
                break

    def _terminate_workers(self) -> None:
        """Stop all worker processes immediately."""
        with self._workers_lock:
            for process in self.process:
                try:
                    if process.is_alive():
                        process.kill()
                except Exception:
                    try:
                        process.terminate()
                    except Exception:
                        pass
                try:
                    process.join(timeout=0.2)
                except Exception:
                    pass
            self.process = []

    def _restart_workers(self) -> None:
        """Restart worker processes after a timeout cancellation."""
        with self._workers_lock:
            if (not self._mp_capable or self.stop_flag
                    or self._global_timed_out or self._remaining_global_budget() <= 0):
                return

            for _ in range(self.max_workers):
                self.process.append(
                    multiprocess.Process( # pylint: disable=not-callable
                        target=process_daemon,
                        args=[self.to_run_queue,
                            self.result_queue,
                            self.error_queue,
                            self.finally_queue,
                            self.shared_cache
                        ]
                    )
                )
                self.process[-1].start()

            CoreDispatcher().affiliate(
                [process.pid for process in self.process],
                core_number=self.max_workers)

    def submit(self, target: callable, *args, deadline: float | None = None, **kwargs) -> bool:
        """Submit a task, waiting for space when a deadline is specified.

        :param callable target: Method to run
        :param Tuple, optional args: parameters passed to the callable
        :param float, optional deadline: Absolute time from ``time.monotonic()``.
        :param Dict, optional kwargs: parameters passed to the callable
        :return: False if the deadline expires before submission, otherwise True.
        """
        while True:
            if self.stop_flag:
                raise TerminatedError("Job submission failed: Executor is currently \
                    shutdown and cannot accept new tasks.")

            remaining = min(self._remaining_global_budget(),
                            float("inf") if deadline is None else deadline - time.monotonic())
            if remaining <= 0:
                return False

            # Keep at most one waiting task per worker in addition to those running.
            # Large candidate objects otherwise make both submission and cancellation
            # spend most of the training budget serializing an unbounded backlog.
            if deadline is None or self.pending_count < self.max_workers * 2:
                break
            time.sleep(min(0.05, remaining))

        with self._state_lock:
            callback_id = len(self.callbacks) - 1

        if self.debug:
            if self._global_deadline is not None:
                raise RuntimeError('A finite global budget requires multiprocessing; '
                                   'sequential fallback cannot interrupt evaluations.')
            result = target(*args, **kwargs)
            self._notify_callback(result, callback_id)
            Logger().info(str(result))
            with self._state_lock:
                self.results.append(result)
                self.submit_count += 1
                self.finished_run += 1
            return True

        try:
            self.to_run_queue.put((target, args, kwargs, callback_id))
            with self._state_lock:
                self.submit_count += 1
        except pickle.PicklingError as exc:
            if (self._remaining_global_budget() <= 0
                    or deadline is not None and time.monotonic() >= deadline):
                return False
            if not self._mp_fallback:
                warnings.warn(
                    f"TimedPoolExecutor fallback to sequential mode (pickle failed: {exc!r})"
                )
                self._mp_fallback = True
            self.debug = True
            if self._global_deadline is not None:
                raise RuntimeError('A finite global budget requires serializable worker tasks; '
                                   'sequential fallback cannot interrupt evaluations.') from exc
            result = target(*args, **kwargs)
            self._notify_callback(result, callback_id)
            Logger().info(str(result))
            with self._state_lock:
                self.results.append(result)
                self.submit_count += 1
                self.finished_run += 1

        return True

    def __finished(self) -> bool:
        """Are all the submitted tasks finished?
        
        :return: True if all tasks are finished
        """
        return self.pending_count == 0

    def reset(self):
        """Reset all queues, callback, results, etc. 
        Allow to reuse this instance of TimedPoolExecutor without restarting subProcess
        """
        with self._state_lock:
            if not self.sliding_stages and self.pending_count == 0:
                self.callbacks = [self.callbacks[-1]]
                self.submit_count = 0
                self.finished_run = 0
            self.results = []

    def __join_collectors(self, restart=True):
        """Join collector thread.
        Stop and start thread, used when we want to sync with thread to collect all data 
        """

        with self._collector_lock:
            if self._queues_closed or self._collectors_stopped:
                return
            # A worker publishes its result/error before its completion marker.
            # Flushing all three queues also collects callbacks that lag behind
            # the completion counter.
            self.error_queue.put(("stop", None))
            self.result_queue.put(("stop", None))
            self.finally_queue.put("stop")

            for collector in self.daemons_collectors:
                collector.join()
            self._collectors_stopped = True

            if restart and not self.stop_flag:
                self.__run_collectors()

    def set_callback(self, callback: callable) -> None:
        """Set the method call to when a task finish

        :param callable callback: callback method
        """
        with self._state_lock:
            self.callbacks.append(callback)

    def join(self, timeout: float | None, reset: bool = True,
             *, cancel_pending: bool | None = None) -> list:
        """Collect finished tasks while preserving sliding stages by default.

        :param float timeout: Maximum seconds to wait. None waits without a timeout.
        :param bool, optional reset: Reset the instance after join(). Defaults to True.
        :param bool, optional cancel_pending: Cancel running and queued tasks on
            timeout. Defaults to False for sliding stages and True otherwise.
            True waits for all tasks rather than leaving a stage early. The
            global deadline always cancels, regardless of this option.

        :return: All finished task results
        """
        cancel_on_timeout = not self.sliding_stages if cancel_pending is None else cancel_pending
        start_time = time.monotonic()
        def remain_time():
            remaining = (float("inf") if timeout is None
                         else max(0.0, timeout - (time.monotonic() - start_time)))
            return min(remaining, self._remaining_global_budget())

        def slide():
            try:
                is_empty = self.to_run_queue.empty()
            except BrokenPipeError:
                is_empty = True

            return self.sliding_stages and not cancel_on_timeout \
                and (
                    is_empty # submit queue is empty
                    and (
                        self.pending_count <= self.max_workers/2
                        # At least half of the worker is free
                        )
                    and self.results # We have got at least one result
                )

        while not self.__finished():
            remaining = remain_time()
            if remaining <= 0 or self.stop_flag or slide():
                break
            time.sleep(min(0.05, remaining))

        timed_out = not self.__finished() and remain_time() == 0
        global_expired = self._global_timed_out or self._remaining_global_budget() <= 0
        cancelled = global_expired or self.stop_flag or (timed_out and cancel_on_timeout)
        if cancelled:
            if global_expired:
                self._global_timed_out = True
            if self._mp_capable and self.process:
                self._terminate_workers()
            self._drain_queue(self.to_run_queue)

        # Join collector thread, just to be sure we have collected all data
        self.__join_collectors()

        if cancelled:
            with self._state_lock:
                self.submit_count = self.finished_run
            if self._mp_capable:
                self._restart_workers()

        # Collector threads can already receive the next stage's completions.
        # Transfer the buffer atomically so each result belongs to one return.
        with self._state_lock:
            results = list(self.results)
            if reset:
                self.reset()

        return results
