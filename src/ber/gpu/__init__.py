"""GPU-only compute for the final phase: every function takes and returns CUDA tensors.

Ids are int64 indices from ``ber.io.ids`` (sorted-string order, so "smaller id" tie-breaks
match the pandas implementations they replace). Sorting is stable; lexicographic sorts are
successive stable ``torch.sort`` passes; every tie has an explicit rule.

``StageTimer`` records wall, process-CPU, I/O and GPU (CUDA-event) time per stage so the
ledger can flag a stage whose CPU time exceeds 10% of wall outside permitted I/O phases.
"""
from __future__ import annotations

import time


def blocking_sync() -> None:
    """``cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync)`` on the (single) visible device.

    Must run before torch / cupy / xgboost create the primary CUDA context. With the default
    spin-wait a host thread burns one core while it waits for the GPU, which makes GPU-bound
    stages look CPU-bound to the ``cpu_violation`` guard (measured: 6.1 s CPU for a 6.1 s pure-GPU
    matmul loop; 0.3 s with blocking sync, wall +6%).
    """
    import ctypes
    import glob
    import os

    import nvidia
    libs = sorted(glob.glob(os.path.join(list(nvidia.__path__)[0], "cu*", "lib", "libcudart.so*")))
    if not libs:
        raise RuntimeError("libcudart not found for cudaSetDeviceFlags")
    rc = ctypes.CDLL(libs[-1]).cudaSetDeviceFlags(4)
    if rc != 0:
        raise RuntimeError(f"cudaSetDeviceFlags(BlockingSync) failed: {rc}")


class StageTimer:
    """Wall / CPU / GPU time of a stage; ``io()`` and ``tok()`` add permitted host phases."""

    def __init__(self, name: str):
        import torch
        self.name = name
        self.t0 = time.time()
        self.c0 = time.process_time()
        self.io_s = 0.0
        self.io_cpu = 0.0
        self.tok_s = 0.0
        self.gpu_ms = 0.0
        self._torch = torch
        self._open = None
        self.tok_cpu = 0.0
        from ber.gpu.xgb import DRIVER
        self.drv0 = (DRIVER["wall"], DRIVER["cpu"])

    def io(self, seconds: float, cpu: float | None = None) -> None:
        self.io_s += seconds
        self.io_cpu += seconds if cpu is None else cpu

    def tok(self, seconds: float, cpu: float | None = None) -> None:
        self.tok_s += seconds
        self.tok_cpu += seconds if cpu is None else cpu

    def host(self, kind: str = "io"):
        """Context manager timing a permitted host phase ("io" or "tok") in wall AND process CPU."""
        outer = self

        class _H:
            def __enter__(self):
                self.w, self.c = time.time(), time.process_time()

            def __exit__(self, *exc):
                (outer.io if kind == "io" else outer.tok)(time.time() - self.w, time.process_time() - self.c)
        return _H()

    class _Gpu:
        def __init__(self, outer):
            self.o = outer

        def __enter__(self):
            t = self.o._torch
            self.a = t.cuda.Event(enable_timing=True)
            self.a.record()
            self.o._open = self.a
            return self

        def __exit__(self, *exc):
            b = self.o._torch.cuda.Event(enable_timing=True)
            b.record()
            b.synchronize()
            self.o.gpu_ms += self.a.elapsed_time(b)
            self.o._open = None

    def gpu(self):
        return StageTimer._Gpu(self)

    def summary(self) -> dict:
        """Wall / CPU / GPU seconds so far. ``gpu_time_s``: CUDA-event time of the ``gpu()`` sections
        (an open section counts up to now)."""
        import torch
        gpu_ms = self.gpu_ms
        if self._open is not None:
            b = torch.cuda.Event(enable_timing=True)
            b.record()
            b.synchronize()
            gpu_ms += self._open.elapsed_time(b)
        from ber.gpu.xgb import DRIVER
        wall = time.time() - self.t0
        cpu = time.process_time() - self.c0
        drv_c = DRIVER["cpu"] - self.drv0[1]
        # CPU outside the permitted phases (I/O incl. multi-threaded parquet decode, tokenisation)
        # and outside the XGBoost kernel-launch driver, against the wall time of the same remainder
        cpu_ex = max(0.0, cpu - self.io_cpu - self.tok_cpu - drv_c)
        wall_ex = max(1e-9, wall - self.io_s - self.tok_s)
        return {"wall_s": round(wall, 1), "cpu_time_s": round(cpu, 1), "gpu_time_s": round(gpu_ms / 1000, 1),
                "io_s": round(self.io_s, 1), "io_cpu_s": round(self.io_cpu, 1), "tok_s": round(self.tok_s, 1),
                "xgb_driver_cpu_s": round(drv_c, 1), "cpu_other_s": round(cpu_ex, 1),
                "cpu_violation": bool(cpu_ex > 0.10 * wall_ex),
                "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
