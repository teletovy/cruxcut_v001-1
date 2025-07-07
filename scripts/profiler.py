import cProfile
import pstats
import io
import time
import psutil
import os
import logging
from functools import wraps
from memory_profiler import profile as memory_profile
from typing import Callable, Any, Optional
import tracemalloc
import threading

class Profiler:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.process = psutil.Process(os.getpid())
        self.start_time = None
        self.start_memory = None
        self._tracemalloc_running = False
        self._profiling_lock = threading.Lock()
        self._active_profiling = False
        
    def start(self):
        """Start profiling"""
        if not self.enabled:
            return
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss
        if not self._tracemalloc_running:
            tracemalloc.start()
            self._tracemalloc_running = True
        
    def stop(self) -> dict:
        """Stop profiling and return metrics"""
        if not self.enabled:
            return {}
            
        end_time = time.time()
        end_memory = self.process.memory_info().rss
        
        if self._tracemalloc_running:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            self._tracemalloc_running = False
        else:
            current, peak = 0, 0
        
        metrics = {
            'execution_time': end_time - self.start_time,
            'memory_used': (end_memory - self.start_memory) / 1024 / 1024,  # MB
            'current_memory': current / 1024 / 1024,  # MB
            'peak_memory': peak / 1024 / 1024  # MB
        }
        
        return metrics
    
    def profile_function(self, func: Callable) -> Callable:
        """Decorator to profile a specific function"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Prevent nested profiling
            if self._active_profiling:
                return func(*args, **kwargs)
                
            with self._profiling_lock:
                if not self._active_profiling:
                    self._active_profiling = True
                    try:
                        # CPU profiling
                        pr = cProfile.Profile()
                        pr.enable()
                        
                        # Execute function
                        start_time = time.time()
                        result = func(*args, **kwargs)
                        end_time = time.time()
                        
                        pr.disable()
                        s = io.StringIO()
                        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
                        ps.print_stats(20)  # Print top 20 time-consuming operations
                        
                        # Get current memory usage
                        memory_info = self.process.memory_info()
                        
                        # Log results
                        logging.info(f"\n{'='*50}\nProfiling results for {func.__name__}:")
                        logging.info(f"Execution time: {end_time - start_time:.2f} seconds")
                        logging.info(f"Current memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
                        logging.info(f"\nDetailed CPU profiling:\n{s.getvalue()}")
                        logging.info('='*50)
                        
                        return result
                    finally:
                        self._active_profiling = False
                else:
                    return func(*args, **kwargs)
                    
        return wrapper
    
    @staticmethod
    def memory_profile(func: Callable) -> Callable:
        """Decorator specifically for memory profiling"""
        return memory_profile(func)
    
    def log_memory_usage(self, message: str = ""):
        """Log current memory usage"""
        if not self.enabled:
            return
            
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        logging.info(f"{message} - Current memory usage: {current_memory:.2f} MB") 