"""Resource-aware scheduling that yields to the host application."""

from __future__ import annotations

import time
from dataclasses import dataclass


class ResourceBusyError(RuntimeError):
    """Raised when running VIO would exceed the configured host budget."""


@dataclass(frozen=True)
class ResourceSnapshot:
    cpu_percent: float | None
    ram_percent: float | None
    gpu_percent: float | None
    vram_percent: float | None

    def as_dict(self) -> dict[str, float | None]:
        return self.__dict__.copy()


class ResourceGovernor:
    def __init__(self, config):
        self.config = config
        self.last_decision = "balanced"
        self.last_snapshot = ResourceSnapshot(None, None, None, None)

    def snapshot(self) -> ResourceSnapshot:
        cpu = ram = gpu = vram = None
        try:
            import psutil
            cpu, ram = psutil.cpu_percent(interval=None), psutil.virtual_memory().percent
        except ImportError:
            pass
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu = float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram = round(memory.used * 100 / memory.total, 1)
        except (ImportError, Exception):
            pass
        self.last_snapshot = ResourceSnapshot(cpu, ram, gpu, vram)
        return self.last_snapshot

    def select_profile(self, requested: str) -> str:
        metrics = self.snapshot()
        overloaded = any(value is not None and value >= self.config.resource_hard_limit
                         for value in (metrics.cpu_percent, metrics.ram_percent, metrics.gpu_percent, metrics.vram_percent))
        pressured = any(value is not None and value >= self.config.resource_soft_limit
                        for value in (metrics.cpu_percent, metrics.ram_percent, metrics.gpu_percent, metrics.vram_percent))
        if overloaded:
            deadline = time.monotonic() + self.config.resource_wait_seconds
            while time.monotonic() < deadline:
                time.sleep(0.25)
                metrics = self.snapshot()
                overloaded = any(value is not None and value >= self.config.resource_hard_limit
                                 for value in (metrics.cpu_percent, metrics.ram_percent, metrics.gpu_percent, metrics.vram_percent))
                if not overloaded:
                    break
            if overloaded:
                self.last_decision = "yielding-to-host"
                raise ResourceBusyError("Host resource budget is busy; retry later")
        levels = ["eco", "balanced", "quality"]
        selected = requested if requested in levels else "balanced"
        if pressured and selected != "eco":
            selected = levels[levels.index(selected) - 1]
        self.last_decision = selected
        return selected

    def status(self, requested: str, active: str | None) -> dict:
        return {"requested_profile": requested, "active_profile": active,
                "decision": self.last_decision, "limits": {"soft": self.config.resource_soft_limit,
                "hard": self.config.resource_hard_limit, "wait_seconds": self.config.resource_wait_seconds},
                "resources": self.snapshot().as_dict()}
