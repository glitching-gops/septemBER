"""multilingual-E5 encoder (fp16, mean pooling, L2-normalised) and the embedding cache.

Symmetric task: the "query: " prefix is used on BOTH sides. Input text keeps the
original script (no transliteration) so the multilingual model can use it.
"""
from __future__ import annotations

import time
import unicodedata
from pathlib import Path

import numpy as np

from ber.embed import MODEL_SUBDIR, gpu_setup

MAX_LEN = {"name": 64, "addr": 96}


class E5Encoder:
    def __init__(self, model_dir: str | Path, batch: int = 1024):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = gpu_setup()
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(model_dir, dtype=torch.float16).to(self.device).eval()
        self.batch = batch
        self.dim = self.model.config.hidden_size

    def encode(self, texts: list[str], max_len: int = 64, prefix: str = "query: ") -> np.ndarray:
        torch = self.torch
        out = np.empty((len(texts), self.dim), dtype=np.float16)
        order = np.argsort([len(t or "") for t in texts], kind="stable")  # length bucketing
        with torch.inference_mode():
            for s in range(0, len(texts), self.batch):
                idx = order[s:s + self.batch]
                batch = [prefix + unicodedata.normalize("NFKC", texts[i] or "") for i in idx]
                enc = self.tok(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(self.device)
                h = self.model(**enc).last_hidden_state
                m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
                e = (h * m).sum(1) / m.sum(1).clamp(min=1)
                out[idx] = torch.nn.functional.normalize(e.float(), dim=-1).half().cpu().numpy()
        return out

    def peak_vram_gb(self) -> float:
        return self.torch.cuda.max_memory_allocated(self.device) / 1e9


def emb_paths(cfg: dict, split: str, source: str, field: str) -> tuple[Path, Path]:
    d = cfg["abs"]["cache_dir"] / "emb"
    return d / f"{split}_{source}_{field}.npy", d / f"{split}_{source}_{field}_ids.npy"


def emb_valid(cfg: dict, split: str, source: str, field: str, ids: np.ndarray) -> bool:
    """True when the cached matrix exists, has one row per id and the id index matches."""
    e, i = emb_paths(cfg, split, source, field)
    if not (e.exists() and i.exists()):
        return False
    try:
        E = np.load(e, mmap_mode="r")
        I = np.load(i, allow_pickle=True)
    except Exception:
        return False
    return E.shape[0] == len(ids) and len(I) == len(ids) and bool((I == ids).all())


def load_emb(cfg: dict, split: str, source: str, field: str) -> np.ndarray:
    return np.load(emb_paths(cfg, split, source, field)[0], mmap_mode="r")


def save_emb(cfg: dict, split: str, source: str, field: str, E: np.ndarray, ids: np.ndarray) -> None:
    e, i = emb_paths(cfg, split, source, field)
    e.parent.mkdir(parents=True, exist_ok=True)
    tmp = e.with_suffix(".tmp.npy")
    np.save(tmp, E)
    np.save(i, np.asarray(ids, dtype=object), allow_pickle=True)
    tmp.replace(e)  # atomic: a crash never leaves a half-written matrix under the final name


def benchmark(enc: E5Encoder, texts: list[str], max_len: int) -> float:
    enc.encode(texts[:2000], max_len)
    t = time.time()
    enc.encode(texts, max_len)
    return len(texts) / (time.time() - t)
