"""Compute caps for the shared server.

Every parallel call in the pipeline takes its worker count from here, and heavy
runs wrap themselves in :class:`RamMonitor` so peak memory is logged and the
``max_ram_gb`` cap is enforced.

Config (``configs/default.yaml``)::

    compute:
      n_jobs: 32
      max_ram_gb: 200
      use_gpu: true
      cpu_fallback: false    # GPU-first policy: never fall back to CPU silently
"""
from __future__ import annotations

import multiprocessing as mp
import os
import threading
import time
from functools import lru_cache

import psutil

from ber.io_utils import load_config

_THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


class RamCapExceeded(RuntimeError):
    """Raised when the process tree's resident memory exceeds ``max_ram_gb``."""


@lru_cache(maxsize=1)
def _cfg() -> dict:
    c = load_config().get("compute") or {}
    return {
        "n_jobs": int(c.get("n_jobs", 8)),
        "max_ram_gb": float(c.get("max_ram_gb", 64)),
        "use_gpu": bool(c.get("use_gpu", False)),
        "cpu_fallback": bool(c.get("cpu_fallback", False)),
        "gpu_ids": [int(g) for g in c.get("gpu_ids", [c.get("gpu_device", 0)])],
        "max_vram_gb": float(c.get("max_vram_gb", 40)),
        # P2: pick the least-loaded GPU per stage instead of a fixed id, and keep this
        # much VRAM free on it for the other users of this shared server.
        "gpu_auto": bool(c.get("gpu_auto", False)),
        "gpu_reserve_gb": float(c.get("gpu_reserve_gb", 8)),
    }


def n_jobs() -> int:
    """Worker count: config cap, optionally LOWERED by env BER_N_JOBS (never raised above the cap)."""
    cap = _cfg()["n_jobs"]
    env = os.environ.get("BER_N_JOBS")
    if env:
        cap = min(cap, max(1, int(env)))
    return max(1, min(cap, os.cpu_count() or 1))


def max_ram_gb() -> float:
    return _cfg()["max_ram_gb"]


def use_gpu() -> bool:
    return _cfg()["use_gpu"]


def cpu_fallback() -> bool:
    """Whether a GPU stage may run on CPU when CUDA is unavailable (policy: False; stop and report)."""
    return _cfg()["cpu_fallback"]


def gpu_ids() -> list[int]:
    return _cfg()["gpu_ids"]


def gpu_device() -> int:
    """The primary GPU (first of gpu_ids)."""
    return _cfg()["gpu_ids"][0]


def max_vram_gb() -> float:
    return _cfg()["max_vram_gb"]


def gpu_auto() -> bool:
    """Whether each stage picks its own GPU at start-up instead of using ``gpu_ids``."""
    return _cfg()["gpu_auto"]


def gpu_reserve_gb() -> float:
    """Headroom left free on the chosen GPU (P2 policy: stay below free - reserve)."""
    return _cfg()["gpu_reserve_gb"]


class NoGpuAvailable(RuntimeError):
    """Raised when no GPU has enough free memory. GPU-first policy: never fall back to CPU."""


def gpu_status() -> list[dict]:
    """[{index, util_pct, free_gb, total_gb}] from nvidia-smi, most free memory first."""
    import subprocess

    q = "index,utilization.gpu,memory.used,memory.total"
    out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    rows = []
    for line in out.strip().splitlines():
        i, util, used, total = (int(float(x)) for x in line.split(","))
        rows.append({"index": i, "util_pct": util, "free_gb": (total - used) / 1024,
                     "total_gb": total / 1024})
    return sorted(rows, key=lambda r: (-r["free_gb"], r["util_pct"]))


def pick_gpu(min_free_gb: float = 15.0) -> dict:
    """The GPU with the most free memory. Raises :class:`NoGpuAvailable` if all are too busy.

    Never returns a CPU device: the compute policy forbids a silent fallback.
    """
    if not use_gpu():
        raise NoGpuAvailable("GPU use is disabled in config (compute.use_gpu: false)")
    rows = gpu_status()
    only = os.environ.get("BER_GPU")
    if only:  # steer a job off a device a protected run is using (still subject to the free check)
        allowed = {int(x) for x in only.split(",")}
        rows = [r for r in rows if r["index"] in allowed] or rows[:0]
        if not rows:
            raise NoGpuAvailable(f"BER_GPU={only} names no existing GPU")
    best = rows[0]
    if best["free_gb"] < min_free_gb:
        raise NoGpuAvailable(
            f"no GPU has {min_free_gb:.0f} GB free (best: GPU {best['index']} with "
            f"{best['free_gb']:.1f} GB). Stopping rather than falling back to CPU.")
    return best


def gpu_budget_gb(gpu: dict | None = None) -> float:
    """VRAM this process may use on ``gpu``: min(max_vram_gb, free - reserve)."""
    gpu = gpu or pick_gpu()
    return max(1.0, min(max_vram_gb(), gpu["free_gb"] - gpu_reserve_gb()))


def limit_threads(n: int | None = None) -> int:
    """Cap BLAS/OpenMP thread pools via env vars (effective for libraries loaded later)."""
    n = n or n_jobs()
    for k in _THREAD_ENV:
        os.environ[k] = str(n)
    return n


def pool(processes: int | None = None):
    """A multiprocessing pool sized by the cap. Workers get single-threaded BLAS."""
    return mp.get_context("fork").Pool(processes=processes or n_jobs(), initializer=limit_threads, initargs=(1,))


def tree_rss_gb(pid: int | None = None) -> float:
    """Memory of a process and all its children, in GB.

    Uses PSS (proportional set size) where available: forked workers share the
    parent's pages copy-on-write, and summing plain RSS would count those shared
    pages once per worker. Falls back to RSS if PSS is unavailable.
    """
    try:
        p = psutil.Process(pid or os.getpid())
        procs = [p] + p.children(recursive=True)
    except psutil.NoSuchProcess:
        return 0.0
    total = 0
    for q in procs:
        try:
            try:
                total += q.memory_full_info().pss
            except (psutil.AccessDenied, AttributeError):
                total += q.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return total / 1e9


class RamMonitor:
    """Context manager that samples process-tree RSS in a background thread.

    ``peak_gb`` holds the maximum observed. Call :meth:`check` at safe points to
    raise :class:`RamCapExceeded` if the cap was crossed; the cap is also checked
    on exit.
    """

    def __init__(self, label: str = "", interval: float = 1.0, cap_gb: float | None = None):
        self.label = label
        self.interval = interval
        self.cap_gb = cap_gb if cap_gb is not None else max_ram_gb()
        self.peak_gb = 0.0
        self.exceeded = False
        self.t0 = 0.0
        self.elapsed = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.is_set():
            gb = tree_rss_gb()
            self.peak_gb = max(self.peak_gb, gb)
            if gb > self.cap_gb:
                self.exceeded = True
            self._stop.wait(self.interval)

    def __enter__(self) -> "RamMonitor":
        self.t0 = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def check(self) -> None:
        if self.exceeded:
            raise RamCapExceeded(f"{self.label}: memory (PSS) peaked at {self.peak_gb:.1f} GB > cap {self.cap_gb:.0f} GB")

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()
        self.peak_gb = max(self.peak_gb, tree_rss_gb())
        self.elapsed = time.time() - self.t0
        print(f"[compute] {self.label}: {self.elapsed:.1f}s, peak PSS {self.peak_gb:.1f} GB (cap {self.cap_gb:.0f} GB)")
        if exc[0] is None:
            self.check()
