"""
This implementation adds a level of abstraction to futures, allowing us to
manage the worker queue with more control, and keep multithreading code
within the steps as simple as possible.
"""
import math
import os
import traceback

from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any
from .logger import Logger
from .meta_singleton import MetaSingleton
from .step import Step


class WorkerFuture(Future):
    """
    Thread Step
    """
    def __init__(self, step: Step, task: callable, *args, **kwargs) -> None:
        super().__init__()

        self.step = step
        self.task = task
        self.args = args
        self.kwargs = kwargs
    

    def run(self) -> Any:
        """
        Execute the thread
        """
        try:
            return self.task(*self.args, **self.kwargs)
        except:  # pylint: disable=bare-except
            Logger().log(f'[red]A worker for [b]{self.step.__class__.__name__}[/b] \
                has crashed.[/red]')
            Logger().log(traceback.format_exc())

            return []


class WorkerManager(metaclass=MetaSingleton):
    """
    This implementation adds a level of abstraction to futures, allowing us to
    manage the worker queue with more control, and keep multithreading code
    within the steps as simple as possible.

    This worker manager acts as a smarter thread pool executor, which is able
    to enqueue resource-intensive tasks in order to limit the amount of active
    threads while still allowing jobs to complete when they depend on "child"
    jobs, preventing deadlocks.

    Reference: https://docs.python.org/3/library/concurrent.futures.html#threadpoolexecutor.
    """
    def __init__(self, max_workers: int = None) -> None:
        self.active_workers_count = 0
        self.executor = ThreadPoolExecutor(max_workers=math.inf)
        self.max_workers = max_workers or os.cpu_count()
        self.running_futures: list[WorkerFuture] = []
        self.running_parents: list[list[int]] = []
        self.queue: list[WorkerFuture] = []


    def __done_callback(self, base_future: Future, worker_future: WorkerFuture) -> None:
        worker_future.set_result(base_future.result())

        self.active_workers_count -= 1
        self.running_futures.remove(worker_future)

        self.run_next(worker_future.step)
    

    def is_sibling_running(self, step: Step) -> bool:
        """
        Checks whether a sibling of `step` is currently running.
        """
        return step.parents_steps in [ f.step.parents_steps for f in self.running_futures ]
    

    def run(self, future: WorkerFuture) -> None:
        """
        Immediately starts `future`.
        """
        # parents are supposed to wait for their children to complete before
        # doing anything else, so they are not an "active" worker as long as
        # they have children: "release" a worker
        if future.step.parents_steps not in self.running_parents:
            self.running_parents.append(future.step.parents_steps)

        self.active_workers_count += 1
        self.running_futures.append(future)

        f = self.executor.submit(future.run)
        f.add_done_callback(lambda _f: self.__done_callback(_f, future))
    

    def run_next(self, current_step: Step) -> None:
        """
        Finds a sibling of `current_step`, and starts it. This should be used
        when a future is done running. If there is no sibling, start whatever
        is next in the queue.
        """
        next_sibling_in_queue = next(( f for f in self.queue \
            if current_step.parents_steps == f.step.parents_steps ), None)

        if next_sibling_in_queue is not None:
            self.queue.remove(next_sibling_in_queue)
            self.run(next_sibling_in_queue)
        else:
            if current_step.parents_steps in self.running_parents:
                # all the children have completed: "unrelease" a worker
                self.running_parents.remove(current_step.parents_steps)

            if (self.active_workers_count - len(self.running_parents) + 1) < self.max_workers \
                and len(self.queue) > 0:
                self.run(self.queue.pop())
    

    def submit(self, step: Step, task: callable, *args, **kwargs) -> WorkerFuture:
        """
        Submits a job to the worker manager, and returns a wrapped future which
        can be used to wait for the job to end, even if it's not started yet
        (as opposed to jobs submitted to a `ThreadPoolExecutor`).
        """
        future = WorkerFuture(step, task, *args, **kwargs)

        if (self.active_workers_count - len(self.running_parents) + 1) < self.max_workers \
            or not self.is_sibling_running(step):
            self.run(future)
        else:
            self.queue.append(future)

        return future
