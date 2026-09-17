"""Deadline and queue bounds, without starting processes or training models."""

import pickle
import queue
import time
import unittest
from unittest.mock import Mock, patch

from iaml.timed_pool_executor import TerminatedError, TimedPoolExecutor


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []
        self.on_sleep = None

    def monotonic(self):
        return self.now

    def sleep(self, duration):
        self.sleeps.append(duration)
        self.now += duration
        if self.on_sleep is not None:
            self.on_sleep()


class TestTimedPoolExecutor(unittest.TestCase):
    def setUp(self):
        # Exercise the real submit/join/reset methods with local queues and a
        # controlled clock. Process startup and collectors are tested separately.
        self.executor = object.__new__(TimedPoolExecutor)
        self.executor.max_workers = 1
        self.executor.debug = False
        self.executor.stop_flag = False
        self.executor.sliding_stages = False
        self.executor.main_daemon = None
        self.executor.callbacks = [None]
        self.executor.results = []
        self.executor.submit_count = 0
        self.executor.finished_run = 0
        self.executor.process = []
        self.executor._mp_capable = False
        self.executor._mp_fallback = False
        self.executor.to_run_queue = queue.Queue()
        self.collectors = Mock()
        self.executor._TimedPoolExecutor__join_collectors = self.collectors

        self.clock = FakeClock()
        for name in ("monotonic", "sleep"):
            patcher = patch(
                f"iaml.timed_pool_executor.time.{name}", getattr(self.clock, name)
            )
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_expired_deadline_does_not_submit_or_execute(self):
        for debug in (False, True):
            with self.subTest(debug=debug):
                self.executor.debug = debug
                target = Mock()
                self.assertFalse(self.executor.submit(target, deadline=0.0))
                target.assert_not_called()
                self.assertTrue(self.executor.to_run_queue.empty())
                self.assertEqual(self.executor.submit_count, 0)

    def test_debug_execution_keeps_results_and_callback(self):
        self.executor.debug = True
        callback = Mock()
        self.executor.set_callback(callback)
        target = Mock(return_value="result")

        self.assertTrue(self.executor.submit(target, 7, value=9, deadline=0.1))

        target.assert_called_once_with(7, value=9)
        callback.assert_called_once_with("result")
        self.assertEqual(self.executor.results, ["result"])
        self.assertEqual(self.executor.submit_count, 1)
        self.assertEqual(self.executor.finished_run, 1)

    def test_full_backlog_stops_at_deadline(self):
        self.assertTrue(self.executor.submit(abs, -1))
        self.assertTrue(self.executor.submit(abs, -2))

        self.assertFalse(self.executor.submit(abs, -3, deadline=0.12))

        self.assertAlmostEqual(self.clock.now, 0.12)
        self.assertEqual(self.executor.to_run_queue.qsize(), 2)
        self.assertEqual(self.executor.submit_count, 2)
        self.assertTrue(all(duration <= 0.05 for duration in self.clock.sleeps))

    def test_submission_waits_for_a_completion_before_deadline(self):
        self.executor.submit(abs, -1)
        self.executor.submit(abs, -2)

        def finish_one():
            self.executor.to_run_queue.get_nowait()
            self.executor.finished_run += 1

        self.clock.on_sleep = finish_one
        self.assertTrue(self.executor.submit(abs, -3, deadline=0.12))

        self.assertEqual(len(self.clock.sleeps), 1)
        self.assertEqual(self.executor.submit_count, 3)
        self.assertEqual(self.executor.finished_run, 1)
        self.assertEqual(self.executor.to_run_queue.qsize(), 2)

    def test_submission_without_deadline_keeps_nonblocking_behavior(self):
        for value in range(3):
            self.assertTrue(self.executor.submit(abs, -value))

        self.assertEqual(self.executor.to_run_queue.qsize(), 3)
        self.assertEqual(self.clock.sleeps, [])

    def test_completion_at_deadline_does_not_allow_a_new_submission(self):
        self.executor.submit(abs, -1)
        self.executor.submit(abs, -2)
        self.clock.on_sleep = lambda: setattr(self.executor, "finished_run", 1)

        self.assertFalse(self.executor.submit(abs, -3, deadline=0.05))

        self.assertEqual(self.executor.submit_count, 2)

    def test_shutdown_interrupts_waiting_submission(self):
        self.executor.submit(abs, -1)
        self.executor.submit(abs, -2)
        self.clock.on_sleep = lambda: setattr(self.executor, "stop_flag", True)

        with self.assertRaises(TerminatedError):
            self.executor.submit(abs, -3, deadline=0.12)

        self.assertEqual(self.executor.submit_count, 2)

    def test_expired_serialization_does_not_start_sequential_fallback(self):
        def failed_put(_):
            self.clock.now = 0.1
            raise pickle.PicklingError("cannot serialize")

        self.executor.to_run_queue.put = failed_put
        target = Mock()

        self.assertFalse(self.executor.submit(target, deadline=0.1))

        target.assert_not_called()
        self.assertEqual(self.executor.submit_count, 0)

    def test_join_waits_for_fractional_timeout_and_clears_backlog(self):
        self.executor.submit(abs, -1)
        self.executor.submit(abs, -2)

        self.assertEqual(self.executor.join(timeout=0.12), [])

        self.assertAlmostEqual(self.clock.now, 0.12)
        self.assertEqual(len(self.clock.sleeps), 3)
        self.assertTrue(self.executor.to_run_queue.empty())
        self.assertEqual(self.executor.submit_count, 0)
        self.assertEqual(self.executor.finished_run, 0)
        self.collectors.assert_called_once_with()

    def test_join_collects_completion_before_fractional_timeout(self):
        self.executor.submit(abs, -1)

        def finish_one():
            self.executor.to_run_queue.get_nowait()
            self.executor.results.append(1)
            self.executor.finished_run += 1

        self.clock.on_sleep = finish_one

        self.assertEqual(self.executor.join(timeout=0.12), [1])
        self.assertAlmostEqual(self.clock.now, 0.05)
        self.assertEqual(self.executor.results, [])

    def test_join_without_time_limit_accepts_none_and_infinity(self):
        def finish_one():
            self.executor.to_run_queue.get_nowait()
            self.executor.results.append(1)
            self.executor.finished_run += 1

        self.clock.on_sleep = finish_one
        for timeout in (None, float("inf")):
            with self.subTest(timeout=timeout):
                self.executor.submit(abs, -1)
                self.assertEqual(self.executor.join(timeout=timeout), [1])

        self.assertEqual(self.clock.sleeps, [0.05, 0.05])

    def test_timeout_preserves_results_and_allows_the_next_sliding_stage(self):
        self.executor.sliding_stages = True
        self.executor.submit(abs, -1)
        self.executor.submit(abs, -2)
        # One result is ready; the other task still occupies the pending queue.
        self.executor.to_run_queue.get_nowait()
        self.executor.results.append(1)
        self.executor.finished_run = 1

        self.assertEqual(self.executor.join(timeout=0.1), [1])

        self.assertEqual(self.executor.submit_count, 1)
        self.assertEqual(self.executor.finished_run, 1)
        self.assertTrue(self.executor.to_run_queue.empty())
        self.assertEqual(self.executor.results, [])
        self.assertTrue(self.executor.submit(abs, -3, deadline=0.2))
        self.assertEqual(self.executor.submit_count, 2)
        self.assertEqual(self.executor.to_run_queue.qsize(), 1)


def delayed_result(delay, value):
    """Wait without using CPU, then return a small result from a real worker."""
    time.sleep(delay)
    return value


def failed_task():
    raise ValueError("expected worker failure")


class TestTimedPoolExecutorProcesses(unittest.TestCase):
    """Check cancellation and reuse through real queues and collector threads."""

    def setUp(self):
        self.executor = TimedPoolExecutor(max_workers=1, sliding_stages=False)
        self.addCleanup(self.executor.shutdown)
        if not self.executor._mp_capable or self.executor.debug:
            self.skipTest("Multiprocessing IPC is unavailable")
        # Worker startup includes a random delay: finish it before timing joins.
        self.executor.submit(abs, -1, deadline=time.monotonic() + 5)
        self.assertEqual(self.executor.join(5), [1])

    def test_fractional_timeout_keeps_completed_result_and_pool_is_reusable(self):
        completed = []
        self.executor.set_callback(completed.append)
        self.executor.submit(delayed_result, 0.02, "completed")
        self.executor.submit(delayed_result, 10, "cancelled")
        self.executor.submit(delayed_result, 0, "queued")

        started = time.monotonic()
        self.assertEqual(self.executor.join(0.3), ["completed"])
        self.assertLess(time.monotonic() - started, 2)
        self.assertTrue(self.executor.to_run_queue.empty())
        self.assertEqual(completed, ["completed"])

        self.executor.submit(abs, -7, deadline=time.monotonic() + 5)
        self.assertEqual(self.executor.join(5), [7])
        self.assertEqual(completed, ["completed", 7])

    def test_bounded_submission_reaches_deadline_with_real_worker(self):
        deadline = time.monotonic() + 0.2
        self.assertTrue(self.executor.submit(delayed_result, 10, "running", deadline=deadline))
        self.assertTrue(self.executor.submit(delayed_result, 10, "queued", deadline=deadline))
        self.assertFalse(self.executor.submit(abs, -3, deadline=deadline))
        self.assertEqual(self.executor.submit_count, 2)
        self.assertEqual(self.executor.join(0), [])

    def test_failed_task_releases_space_and_results_are_collected(self):
        deadline = time.monotonic() + 5
        self.executor.submit(failed_task, deadline=deadline)
        self.executor.submit(abs, -2, deadline=deadline)
        self.assertTrue(self.executor.submit(abs, -3, deadline=deadline))
        self.assertEqual(self.executor.join(5), [2, 3])


if __name__ == "__main__":
    unittest.main()
