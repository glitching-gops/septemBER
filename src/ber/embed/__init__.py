"""Dense encodings (multilingual-E5) on GPU, cached per split x source x field."""
from __future__ import annotations

import os

from ber import compute

MODEL_SUBDIR = "models/multilingual-e5-small"
MODEL_ID = "intfloat/multilingual-e5-small"
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
MODEL_LICENSE = "MIT"
MODEL_PARAMS = "118M"
_GPU_DEVICE = None


def gpu_setup():
    """Expose one GPU via CUDA_VISIBLE_DEVICES and cap this process's VRAM.

    With ``compute.gpu_auto`` (P2 default) the GPU with the most free memory is chosen at
    call time and the cap is ``min(max_vram_gb, free - gpu_reserve_gb)``; otherwise the
    configured ``gpu_ids`` are used with a flat ``max_vram_gb`` cap (P1 behaviour).
    Must run before torch initialises CUDA. Returns a torch.device ("cuda:0").
    """
    global _GPU_DEVICE
    if _GPU_DEVICE is not None:
        return _GPU_DEVICE
    if not compute.use_gpu():
        raise RuntimeError("GPU use is disabled in config (compute.use_gpu: false)")
    budget = compute.max_vram_gb()
    # ``compute`` reads default.yaml only, so a config's ``gpu_auto: true`` never reached here
    # in P2 (every stage used the static gpu_ids). BER_GPU (e.g. "0,1", or "0" to steer away
    # from a protected run) switches on the least-loaded pick among the listed devices.
    if compute.gpu_auto() or os.environ.get("BER_GPU"):
        gpu = compute.pick_gpu()  # raises NoGpuAvailable rather than falling back to CPU
        budget = compute.gpu_budget_gb(gpu)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu["index"])
        print(f"[gpu] using GPU {gpu['index']} ({gpu['free_gb']:.1f} GB free, "
              f"{gpu['util_pct']}% util); budget {budget:.1f} GB", flush=True)
    else:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", ",".join(str(g) for g in compute.gpu_ids()))
    if os.environ.get("BER_BLOCKING_SYNC", "0") == "1":
        from ber.gpu import blocking_sync
        blocking_sync()  # before the CUDA context exists: host threads sleep on GPU syncs instead of spinning
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available (check driver / PyTorch build)")
    dev = torch.device("cuda:0")
    total = torch.cuda.get_device_properties(dev).total_memory / 1e9
    torch.cuda.set_per_process_memory_fraction(min(1.0, budget / total), dev)
    _GPU_DEVICE = dev
    return dev
