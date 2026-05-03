#!/usr/bin/env python3
"""
GPU energy monitor — continuous NVML sampling at 10 Hz.

Same methodology as the parent ``vlm_energy_signatures_multilingual`` v3
script, kept self-contained so this package can be used stand-alone.

Primary metric: total Wh over the context-managed window.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


class GPUEnergySampler:
    """Context manager; use as::

        with GPUEnergySampler() as s:
            # ... GPU work ...
        print(s.total_wh(), s.stats())
    """

    def __init__(self, interval_sec: float = 0.1, gpu_index: int = 0) -> None:
        self.interval       = interval_sec
        self.gpu_index      = gpu_index
        self._thread: Optional[threading.Thread] = None
        self._stop          = threading.Event()
        self.samples_watts: List[float] = []
        self.start_time:    Optional[float] = None
        self.end_time:      Optional[float] = None
        self._enabled       = NVML_AVAILABLE
        self.handle         = None

    def __enter__(self) -> "GPUEnergySampler":
        if not self._enabled:
            self.start_time = time.time()
            return self
        try:
            pynvml.nvmlInit()
            self.handle        = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            self.samples_watts = []
            self.start_time    = time.time()
            self._stop.clear()
            self._thread = threading.Thread(target=self._sample_loop, daemon=True)
            self._thread.start()
        except Exception as e:
            print(f"⚠️  NVML init failed: {e}")
            self._enabled = False
        return self

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            try:
                mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
                self.samples_watts.append(mw / 1000.0)
            except Exception:
                self.samples_watts.append(float("nan"))
            time.sleep(self.interval)

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.end_time = time.time()
        if not self._enabled:
            return False
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
        return False

    # ---- accessors --------------------------------------------------------

    def duration_sec(self) -> float:
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time is not None else time.time()
        return float(end - self.start_time)

    def total_wh(self) -> float:
        if not self._enabled or not self.samples_watts:
            return float("nan")
        valid = [s for s in self.samples_watts if math.isfinite(s)]
        if not valid:
            return float("nan")
        return float(np.mean(valid)) * (self.duration_sec() / 3600.0)

    def stats(self) -> Dict[str, Any]:
        if not self._enabled or not self.samples_watts:
            return {"enabled": False, "duration_sec": self.duration_sec()}
        arr = np.array(self.samples_watts, dtype=float)
        valid = arr[np.isfinite(arr)]
        if valid.size == 0:
            return {"enabled": True, "error": "no valid samples"}
        return {
            "enabled":       True,
            "samples_valid": int(valid.size),
            "duration_sec":  self.duration_sec(),
            "avg_watts":     float(np.mean(valid)),
            "max_watts":     float(np.max(valid)),
            "min_watts":     float(np.min(valid)),
            "total_wh":      self.total_wh(),
        }
