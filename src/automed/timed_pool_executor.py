import multiprocess
import threading
import time
import math
import traceback

def process_daemon(to_run_queue:multiprocess.Queue, queue:multiprocess.Queue, error_queue:multiprocess.Queue, finally_queue:multiprocess.Queue):
    result = None
    while True:
        if not to_run_queue.empty():
            value = to_run_queue.get()
            if isinstance(value, str) and value == "stop":
                to_run_queue.put("stop")
                break
            
            method, args, kwargs = value
            
            try:
                result = method(*args, **kwargs)
                queue.put(result)
            except Exception:
                error_queue.put(traceback.format_exc())
            finally:
                finally_queue.put(1)
        else:
            time.sleep(0.1)
        

class TimedPoolExecutor:
    def __init__(self, max_workers:int=None, callback:callable=None):
        self.max_workers = min(max_workers, multiprocess.cpu_count())
        self.stop_flag:bool = False
        self.daemon = None
        self.to_run_queue = multiprocess.Queue()
        self.error_queue = multiprocess.Queue()
        self.result_queue = multiprocess.Queue()
        self.finally_queue = multiprocess.Queue()
        self.callback = callback
        self.results = []
        self.submit_count = 0
        self.finished_run = 0
        
        self.process = []
        for _ in range(max_workers):
            self.process.append(
                multiprocess.Process(
                    target=process_daemon,
                    args=[self.to_run_queue, self.result_queue, self.error_queue, self.finally_queue]
                )
            )
            self.process[-1].start()
            
        self.__run_daemon()
    
    def shutdown(self):
        while not self.to_run_queue.empty():
            self.to_run_queue.get()
        self.to_run_queue.put("stop")
        self.stop_flag = True
        
        # Force stop if needed
        for process in self.process:
            if process.is_alive():
                process.terminate()

    def __collect_results(self):
        while not self.result_queue.empty():
            result = self.result_queue.get()
            if self.callback:
                self.callback(result)
            self.results.append(result)
        
    def __print_errors(self):
        while not self.error_queue.empty():
            print("ERROR IN PROCESS", self.error_queue.get())
    
    def __keep_running(self):
        while True:
            self.__print_errors()
            self.__collect_results()
            time.sleep(0.1)
            
            if self.stop_flag:
                break
            
    
    def __run_daemon(self):
        self.stop_flag = False
        if not self.daemon or not self.daemon.is_alive():
            self.daemon = threading.Thread(target=self.__keep_running)
            self.daemon.start()
                
        
    def submit(self, target, *args, **kwargs):
        self.to_run_queue.put((target, args, kwargs))
        self.submit_count += 1
        
    def __finished(self) -> bool:
        while not self.finally_queue.empty():
            self.finally_queue.get()
            self.finished_run += 1
            
        return self.finished_run >= self.submit_count
    
    def reset(self):
        for queue in [self.error_queue, self.finally_queue, self.to_run_queue, self.result_queue]:
            while not queue.empty():
                queue.get()
                
        self.callback = None
        self.results = []
        self.submit_count = 0
        self.finished_run = 0
                
    def set_callback(self, callback:callable):
        self.callback = callback
        
    def join(self, timeout, reset:bool=True):
        start_time = time.time()
        def remain_time():
            return max(0, int(timeout - (time.time() - start_time)))
        
        while not self.__finished() and remain_time():
            time.sleep(0.5)
        
        self.__collect_results()
        self.__print_errors()
        
        results = self.results # Save before reset !
        
        if reset:
            self.reset()
        
        return results