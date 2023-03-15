import psutil
import torch
import time

from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate


class MaxGPUAllocationExceeded(Exception):
    def __init__(self, allocated, max_allowed):
        message = f"GPU memory allocation ({allocated} MB) exceeded the " \
                  f"maximum allowed amount which is {max_allowed} MB."
        super(MaxGPUAllocationExceeded, self).__init__(message)


class GPUMemoryChecker(SupervisedPlugin):
    """
    This plugin checks the maximum amount of GPU memory allocated after each
    experience.
    """
    def __init__(self,
                 max_allowed: int = 5000,
                 device: torch.DeviceObjType = torch.device("cuda:0")
    ):
        """

        :param max_allowed: Maximum GPU memory allowed in MB.
        :param device: Device for which memory allocation should be checked.
        """

        super().__init__()
        self.max_allowed = max_allowed
        self.gpu_allocated = 0
        self.device = device

    def after_training_exp(self, strategy: SupervisedTemplate, *args, **kwargs):
        gpu_allocated = torch.cuda.max_memory_allocated(device=self.device)
        gpu_allocated = gpu_allocated // 1000000
        self.gpu_allocated = gpu_allocated
        print(f"MAX GPU MEMORY ALLOCATED: {gpu_allocated} MB")

        if gpu_allocated > self.max_allowed:
            raise MaxGPUAllocationExceeded(gpu_allocated, self.max_allowed)


class MaxRAMAllocationExceeded(Exception):
    def __init__(self, allocated, max_allowed):
        message = f"RAM allocation ({allocated} MB) exceeded the " \
                  f"maximum allowed amount which is {max_allowed} MB."
        super(MaxRAMAllocationExceeded, self).__init__(message)


class RAMChecker(SupervisedPlugin):
    """
    This plugin checks the maximum amount of RAM used after each experience.
    """
    def __init__(self, max_allowed: int = 5000):
        """
        :param max_allowed: Maximum GPU memory allowed in MB.
        """

        super().__init__()
        self.max_allowed = max_allowed
        self.ram_allocated = 0

    def after_training_exp(self, strategy: SupervisedTemplate, *args, **kwargs):
        ram_allocated = psutil.Process().memory_info().rss
        ram_allocated = ram_allocated // 1000000
        self.ram_allocated = ram_allocated
        print(f"MAX RAM ALLOCATED: {ram_allocated} MB")

        if ram_allocated > self.max_allowed:
            raise MaxGPUAllocationExceeded(ram_allocated, self.max_allowed)


class TimeExceeded(Exception):
    def __init__(self, allocated, max_allowed):
        message = f"Time ({allocated} minutes) exceeded the " \
                  f"maximum allowed amount which is {max_allowed} minutes."
        super(TimeExceeded, self).__init__(message)


class TimeChecker(SupervisedPlugin):
    """
    This plugin checks the amount of time spent after each experience.
    """
    def __init__(self, max_allowed: int = 5000):
        """
        :param max_allowed: Maximum amount of time allowed in minutes.
        """

        super().__init__()
        self.max_allowed = max_allowed
        self.start = time.time()
        self.time_spent = 0

    def after_training_exp(self, strategy: SupervisedTemplate, *args, **kwargs):
        time_spent = time.time() - self.start
        time_spent = time_spent // 60
        self.time_spent = time_spent
        print(f"TIME SPENT: {time_spent} MINUTES")

        if time_spent > self.max_allowed:
            raise TimeExceeded(time_spent, self.max_allowed)


__all__ = ["GPUMemoryChecker", "RAMChecker", "TimeChecker"]
