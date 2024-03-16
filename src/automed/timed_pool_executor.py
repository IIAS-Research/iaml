import multiprocess
import threading
import time
import math
import traceback

def target_wrapper(method:callable, queue:multiprocess.Queue, error_queue:multiprocess.Queue, callback_queue:multiprocess.Queue, *args, **kwargs):
    result = None
    try:
        result = method(*args, **kwargs)
        queue.put(result)
    except Exception:
        error_queue.put(traceback.format_exc())
    finally:
        callback_queue.put(result)

class TimedPoolExecutor:
    def __init__(self, max_workers:int=None, callback:callable=None):
        self.max_workers = max_workers or math.inf
        self.started:list = []
        self.to_run:list = []
        self.stop_flag:bool = False
        self.daemon = None
        self.error_queue = multiprocess.Queue()
        self.callback_queue = multiprocess.Queue()
        self.callback = callback
    
    @property   
    def finished(self):
        return self.__count_running() == 0 and self.__count_to_run() == 0
    
    def __count_to_run(self):
        return len(self.to_run)
    
    def __run_callback(self):
        while not self.callback_queue.empty():
            self.callback(self.callback_queue.get())
        
    def __count_running(self):
        return sum(map(lambda process : process[0].is_alive(), self.started))
    
    def __print_errors(self):
        while not self.error_queue.empty():
            print("ERROR IN PROCESS", self.error_queue.get())
    
    def __keep_running(self):
        while True:
            
            self.__print_errors()
            self.__run_callback()
                
            if not self.__run_next():
                if self.finished:
                    break
                else:
                    time.sleep(1)
            if self.stop_flag:
                break
            
    def __run_next(self) -> bool:
        if self.__count_running() < self.max_workers:
            if self.to_run:
                target, args, kwargs = self.to_run[0]
                del self.to_run[0]
                
                queue = multiprocess.Queue()
                process = multiprocess.Process(target=target_wrapper, args=[target, queue, self.error_queue, self.callback_queue, *args], kwargs=kwargs)
                process.start()
                
                self.started.append((process, queue))
                return True
        return False
    
    def __run_daemon(self):
        self.stop_flag = False
        if not self.daemon or not self.daemon.is_alive():
            self.daemon = threading.Thread(target=self.__keep_running)
            self.daemon.start()
            
            # Wait until daemon have started is first process
            while self.to_run and not self.__count_running():          
                time.sleep(0.1)
                
        
    def submit(self, target, *args, **kwargs):
        self.to_run.append((target, args, kwargs))
        # # Fix a strange behavior -> if daemon start the with process, Process raise an Exception
        # if not self.daemon:
        #     self.__run_next()
            
        self.__run_daemon()
            
    def join(self, timeout):
        start_time = time.time()
        def remain_time():
            return max(0, int(timeout - (time.time() - start_time)))
        
        results = []
        for process, queue in self.started:
            self.__print_errors()
            
            if remain_time():
                self.__run_next()
            
            process.join(remain_time())
            
            if process.is_alive():
                process.terminate()
                process.join()
            else:
                while not queue.empty():
                    results.append(queue.get())
                    
        self.stop_flag = True
        self.__print_errors()
        self.__run_callback()
        
        return results