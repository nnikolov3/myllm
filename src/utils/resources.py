from dataclasses import dataclass
import logging
import psutil
import torch
import gc
import asyncio
from threading import Lock
from typing import Dict, Optional, Tuple
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class SystemResources:
    """Container for system resource information."""
    cpu_count: int
    total_memory: int
    available_memory: int
    gpu_available: bool
    gpu_count: int
    gpu_info: Dict[int, Dict] = None
    timestamp: str = None

    def __post_init__(self):
        self.timestamp = datetime.now().isoformat()
        if self.gpu_available and not self.gpu_info:
            self.gpu_info = {
                i: {
                    'name': torch.cuda.get_device_name(i),
                    'total_memory': torch.cuda.get_device_properties(i).total_memory,
                    'memory_allocated': torch.cuda.memory_allocated(i),
                    'memory_cached': torch.cuda.memory_reserved(i)
                }
                for i in range(self.gpu_count)
            }

class ResourceMonitor:
    """Monitors system resource usage."""
    
    def __init__(self, sampling_interval: float = 1.0):
        self.sampling_interval = sampling_interval
        self._monitoring = False
        self._lock = Lock()
        self.history = []
        
    async def start_monitoring(self):
        """Start resource monitoring."""
        self._monitoring = True
        while self._monitoring:
            try:
                resources = self._get_current_resources()
                with self._lock:
                    self.history.append(resources)
                    if len(self.history) > 100:  # Keep last 100 samples
                        self.history.pop(0)
                await asyncio.sleep(self.sampling_interval)
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                break
        return True
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self._monitoring = False
    
    def _get_current_resources(self) -> SystemResources:
        """Get current system resource state."""
        vm = psutil.virtual_memory()
        return SystemResources(
            cpu_count=psutil.cpu_count(),
            total_memory=vm.total,
            available_memory=vm.available,
            gpu_available=torch.cuda.is_available(),
            gpu_count=torch.cuda.device_count() if torch.cuda.is_available() else 0
        )
    
    def get_resource_usage(self) -> Dict:
        """Get current resource usage statistics."""
        with self._lock:
            if not self.history:
                return {}
            
            latest = self.history[-1]
            return {
                'cpu_percent': psutil.cpu_percent(interval=None),
                'memory_percent': (latest.total_memory - latest.available_memory) / latest.total_memory * 100,
                'gpu_utilization': self._get_gpu_utilization() if latest.gpu_available else None
            }
    
    def _get_gpu_utilization(self) -> Dict[int, float]:
        """Get GPU utilization percentages."""
        return {
            i: torch.cuda.utilization(i)
            for i in range(torch.cuda.device_count())
        }

class ResourceManager:
    """Manages system resources and optimization."""
    
    def __init__(self):
        self._lock = Lock()
        self.monitor = ResourceMonitor()
        self.resources = self._initialize_resources()
        self._log_system_info()
        
    def _initialize_resources(self) -> SystemResources:
        """Initialize and return system resources information."""
        try:
            return SystemResources(
                cpu_count=psutil.cpu_count(),
                total_memory=psutil.virtual_memory().total,
                available_memory=psutil.virtual_memory().available,
                gpu_available=torch.cuda.is_available(),
                gpu_count=torch.cuda.device_count() if torch.cuda.is_available() else 0
            )
        except Exception as e:
            logger.error(f"Error initializing resources: {e}")
            # Provide fallback values if there's an error
            return SystemResources(
                cpu_count=1,
                total_memory=0,
                available_memory=0,
                gpu_available=False,
                gpu_count=0
            )

    def _log_system_info(self) -> None:
        """Log system resource information."""
        logger.info("System Resources:")
        logger.info(f"CPU Count: {self.resources.cpu_count}")
        logger.info(f"Memory: {self.resources.total_memory / (1024**3):.2f} GB")
        logger.info(f"GPU Available: {self.resources.gpu_available}")
        
        if self.resources.gpu_available:
            logger.info(f"GPU Count: {self.resources.gpu_count}")
            for i in range(self.resources.gpu_count):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
    async def start_monitoring(self):
        """Start resource monitoring."""
        try:
            # Delegate to the monitor's start_monitoring
            monitoring_task = asyncio.create_task(self.monitor.start_monitoring())
            logger.info("Resource monitoring started")
            return monitoring_task
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            raise

    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitor.stop_monitoring()

    def optimize_batch_size(self, item_size_bytes: int = 1024 * 1024) -> int:
        """Calculate optimal batch size based on available memory."""
        with self._lock:
            if self.resources.gpu_available:
                # Use 80% of available GPU memory
                available_memory = int(
                    torch.cuda.get_device_properties(0).total_memory * 0.8
                )
            else:
                # Use 80% of available system memory
                available_memory = int(
                    psutil.virtual_memory().available * 0.8
                )
            return max(1, available_memory // item_size_bytes)

    def get_optimal_thread_count(self) -> int:
        """Calculate optimal number of worker threads."""
        return max(1, self.resources.cpu_count - 1)

    async def clear_gpu_memory(self):
        """Clear GPU memory cache."""
        if self.resources.gpu_available:
            with self._lock:
                torch.cuda.empty_cache()
                gc.collect()
                await asyncio.sleep(0)

    def get_device_allocation(self, model_size_bytes: int) -> Tuple[str, Optional[int]]:
        """Determine optimal device for model allocation."""
        with self._lock:
            if not self.resources.gpu_available:
                return "cpu", None

            # Get GPU memory info
            gpu_memory = []
            for i in range(self.resources.gpu_count):
                free_memory = (
                    torch.cuda.get_device_properties(i).total_memory -
                    torch.cuda.memory_allocated(i)
                )
                gpu_memory.append((i, free_memory))

            # Sort by available memory
            gpu_memory.sort(key=lambda x: x[1], reverse=True)

            # Check if largest available GPU has enough memory
            if gpu_memory[0][1] >= model_size_bytes * 1.2:  # 20% buffer
                return "cuda", gpu_memory[0][0]
            return "cpu", None

    def monitor_memory_usage(self) -> Dict:
        """Get current memory usage statistics."""
        vm = psutil.virtual_memory()
        memory_stats = {
            'total': vm.total,
            'available': vm.available,
            'percent': vm.percent,
            'used': vm.used,
            'free': vm.free
        }

        if self.resources.gpu_available:
            memory_stats['gpu'] = {}
            for i in range(self.resources.gpu_count):
                memory_stats['gpu'][i] = {
                    'total': torch.cuda.get_device_properties(i).total_memory,
                    'allocated': torch.cuda.memory_allocated(i),
                    'cached': torch.cuda.memory_reserved(i)
                }

        return memory_stats

    def get_resource_summary(self) -> Dict:
        """Get summary of current resource state."""
        usage = self.monitor.get_resource_usage()
        memory = self.monitor_memory_usage()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu': {
                'count': self.resources.cpu_count,
                'usage_percent': usage.get('cpu_percent')
            },
            'memory': {
                'total_gb': memory['total'] / (1024**3),
                'available_gb': memory['available'] / (1024**3),
                'usage_percent': memory['percent']
            },
            'gpu': {
                'available': self.resources.gpu_available,
                'count': self.resources.gpu_count,
                'utilization': usage.get('gpu_utilization')
            } if self.resources.gpu_available else None
        }

    async def cleanup(self):
        """Cleanup resources before shutdown."""
        try:
            self.stop_monitoring()
            await asyncio.sleep(0)  # Allow any pending tasks to complete
            logger.info("Resource manager cleanup completed")
        except Exception as e:
            logger.error(f"Error during resource cleanup: {e}")