"""
Thread with return value on join
"""
from threading import Thread

class ThreadWithReturnValue(Thread):
    """
    Thread with return value on join
    """
    def __init__(self, *args, group=None, target=None, name=None, **kwargs):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self) -> None:
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args) -> any:
        Thread.join(self, *args)
        return self._return
