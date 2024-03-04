"""
Thread-safe implementation of Singleton
"""
from threading import Lock

# Inspired by : https://refactoring.guru/fr/design-patterns/singleton/python/example#example-1
class MetaSingleton(type):
    """
    Thread-safe implementation of Singleton
    """
    _instances = {}
    _lock: Lock = Lock() # Use to synchronize threads during first access

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]
