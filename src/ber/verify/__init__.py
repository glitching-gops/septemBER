"""Pair verifier: a cross-encoder / reranker that re-reads the stage-1 borderline pairs.

The verifier sees both records as text and outputs P(same business). Its score becomes a
feature of a retrained stage-1 XGBoost rather than replacing it (D7), so every model here
only has to be informative, not calibrated to the decision threshold.

Every model is MIT or Apache-2.0 and at most 8B parameters; exact counts were read from the
safetensors metadata and are recorded in ``rerankers.MODELS``. Nothing here calls a network
service: weights are downloaded once into HF_HOME and loaded with HF_HUB_OFFLINE=1.
"""


def set_deterministic(seed: int = 42) -> None:
    """Deterministic GPU inference: fixed seeds, cuDNN/cuBLAS determinism, no autotuning.

    Must run before the first CUDA call (CUBLAS_WORKSPACE_CONFIG is read at cuBLAS init).
    ``warn_only`` keeps ops without a deterministic kernel running (they are reported, not
    silently swapped for a CPU path).
    """
    import os
    import random

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def set_deterministic(seed: int = 42) -> None:
    """Deterministic GPU inference: fixed seeds, cuDNN/cuBLAS determinism, no TF32/autotuning.

    Call before the first CUDA call (CUBLAS_WORKSPACE_CONFIG is read at cuBLAS init).
    ``warn_only``: an op without a deterministic kernel is reported, never swapped for CPU.
    Measured on the verifier: bit-identical reruns, and faster than the default path.
    """
    import os
    import random

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def score_dir(cfg: dict, key: str, fold: int):
    """Where fold ``fold``'s scores live: ``verifier.scores`` = "det" -> the deterministic rescore."""
    from pathlib import Path

    d = Path(cfg["abs"]["cache_dir"]).parent / "verifier" / key / f"fold{fold}"
    sub = (cfg.get("verifier") or {}).get("scores", "")
    return d / sub if sub else d
