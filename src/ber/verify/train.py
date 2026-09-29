"""Verifier fine-tuning: training-pair construction and the training loops (GPU only).

Training pairs per S1 (D5): every labelled positive (retrieved or not), the top ``n_hard``
negatives by stage-1 E1b probability, and ``n_rand`` random other candidate negatives;
pairs touching a non-Latin name or an empty address are repeated ``oversample`` times.

Two recipes:

* ``qwen_yesno`` + LoRA (r 32, alpha 64, dropout 0.05, all linear layers): BCE on the
  yes-minus-no logit at the last position, so fine-tuning sharpens exactly the quantity
  the zero-shot readout already uses.
* ``cross`` full fine-tune (fp32 master weights, bf16 autocast): BCE on the head logit.

Both: AdamW, cosine schedule with 3% warm-up, one epoch, gradient accumulation to the
effective batch, length-bucketed micro-batches under a token budget, checkpoint every
``ckpt_every`` optimiser steps.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd


def build_training_pairs(oof: pd.DataFrame, gt: dict, s1_ids, nonlatin: set, empty_addr: set,
                         n_hard: int = 8, n_rand: int = 2, oversample: int = 2,
                         seed: int = 42) -> pd.DataFrame:
    """(s1_id, m_id, y) training rows for the given S1 (their candidates live in ``oof``)."""
    rng = np.random.default_rng(seed)
    ids = set(s1_ids)
    c = oof[oof["s1_id"].isin(ids)]
    neg = c[c["y"] == 0].sort_values(["s1_id", "p"], ascending=[True, False])
    neg = neg.assign(r=neg.groupby("s1_id").cumcount())
    hard = neg[neg["r"] < n_hard]
    rest = neg[neg["r"] >= n_hard]
    rest = rest.assign(u=rng.random(len(rest))).sort_values(["s1_id", "u"])
    rand = rest[rest.groupby("s1_id").cumcount() < n_rand]
    pos = pd.DataFrame([(s, m) for s in ids for m in gt.get(s, ())], columns=["s1_id", "m_id"])
    rows = pd.concat([pos.assign(y=1), hard[["s1_id", "m_id"]].assign(y=0),
                      rand[["s1_id", "m_id"]].assign(y=0)], ignore_index=True)
    hot = (rows["s1_id"].isin(nonlatin) | rows["m_id"].isin(nonlatin) |
           rows["s1_id"].isin(empty_addr) | rows["m_id"].isin(empty_addr))
    rows = pd.concat([rows] + [rows[hot]] * (oversample - 1), ignore_index=True)
    return rows.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _micro_batches(lengths: np.ndarray, token_budget: int, max_pairs: int, seed: int):
    """Length-bucketed micro-batches: sort within random chunks, cut by padded-token budget."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(lengths))
    batches = []
    chunk = 8192
    for s in range(0, len(order), chunk):
        idx = order[s:s + chunk]
        idx = idx[np.argsort(lengths[idx], kind="stable")]
        i = 0
        while i < len(idx):
            j = i + 1
            while j < len(idx) and j - i < max_pairs and lengths[idx[j]] * (j + 1 - i) <= token_budget:
                j += 1
            batches.append(idx[i:j])
            i = j
    rng.shuffle(batches)
    return batches


def train(verifier, left: list[str], right: list[str], y: np.ndarray, out_dir: Path, *,
          lora: bool, lr: float, eff_batch: int, token_budget: int = 16_000, max_pairs: int = 128,
          warmup: float = 0.03, ckpt_every: int = 2000, seed: int = 42, max_minutes: float = 1e9,
          lora_r: int = 32, lora_alpha: int = 64, lora_dropout: float = 0.05, log_every: int = 50,
          resumable: bool = False) -> dict:
    """Fine-tune ``verifier.model`` in place (one epoch). Returns training stats.

    ``resumable``: every ``ckpt_every`` optimiser steps the full training state (weights,
    AdamW moments, schedule, position in the seeded batch order) is written to
    ``out_dir/ckpt-latest/state.pt`` (atomically, previous one replaced); a rerun with the same
    inputs picks up from it and replays the identical remaining batch sequence.
    """
    import torch

    torch.manual_seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    v = verifier
    if lora:
        from peft import LoraConfig, get_peft_model

        v.model = get_peft_model(v.model, LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules="all-linear", task_type="CAUSAL_LM"))
        for n, p in v.model.named_parameters():  # LoRA weights in fp32, base stays bf16
            if p.requires_grad:
                p.data = p.data.float()
    else:
        v.model = v.model.float()
    v.model.train()
    params = [p for p in v.model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01, fused=True)

    seqs = v.encode(left, right)
    lengths = np.array([len(s) for s in seqs])
    batches = _micro_batches(lengths, token_budget, max_pairs, seed)
    total_steps = max(1, math.ceil(len(seqs) / eff_batch))
    warm = max(1, int(warmup * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total_steps - warm))))
    yt = torch.tensor(y, dtype=torch.float32)
    lossf = torch.nn.BCEWithLogitsLoss(reduction="sum")
    t0, seen, step, acc, run_loss, run_n = time.time(), 0, 0, 0, 0.0, 0
    stopped_early = False
    start_bi = 0
    state_p = out_dir / "ckpt-latest" / "state.pt"
    if resumable and state_p.exists():
        st = torch.load(state_p, map_location="cpu", weights_only=False)
        assert st["n_pairs"] == len(seqs) and st["n_batches"] == len(batches), "resume: inputs changed"
        v.model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        step, seen, start_bi = st["step"], st["seen"], st["next_bi"]
        torch.set_rng_state(st["rng_cpu"])
        torch.cuda.set_rng_state(st["rng_cuda"])
        print(f"    resumed from step {step} ({seen:,} pairs)", flush=True)
        del st
    t_resume_seen = seen
    opt.zero_grad(set_to_none=True)
    for bi, idx in enumerate(batches):
        if bi < start_bi:
            continue
        ids, att = v.collate([seqs[k] for k in idx])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit, _ = v.logits(ids, att)
        loss = lossf(logit.float(), yt[idx].to(logit.device)) / eff_batch
        loss.backward()
        acc += len(idx)
        seen += len(idx)
        run_loss += float(loss) * eff_batch
        run_n += len(idx)
        last = bi == len(batches) - 1
        if acc >= eff_batch or last:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            acc = 0
            step += 1
            if step % log_every == 0 or last:
                el = time.time() - t0
                rate = (seen - t_resume_seen) / el
                print(f"    step {step}/{total_steps} pairs {seen:,}/{len(seqs):,} loss {run_loss / run_n:.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} {rate:,.0f} pairs/s "
                      f"eta {(len(seqs) - seen) / rate / 60:.1f} min", flush=True)
                run_loss, run_n = 0.0, 0
            if step % ckpt_every == 0 and not last:
                if resumable:
                    tmp = out_dir / "ckpt-latest.tmp"
                    tmp.mkdir(parents=True, exist_ok=True)
                    torch.save({"model": v.model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                                "step": step, "seen": seen, "next_bi": bi + 1, "n_pairs": len(seqs),
                                "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state(),
                                "n_batches": len(batches)}, tmp / "state.pt")
                    import shutil
                    shutil.rmtree(out_dir / "ckpt-latest", ignore_errors=True)
                    tmp.rename(out_dir / "ckpt-latest")
                else:
                    save(v, out_dir / f"ckpt-{step}", lora)
            if (time.time() - t0) / 60 > max_minutes:
                print(f"    time cap {max_minutes:.0f} min reached at step {step}", flush=True)
                stopped_early = True
                break
    save(v, out_dir / "final", lora)
    v.model.eval()
    if lora:
        v.model = v.model.merge_and_unload()
    v.model = v.model.to(torch.bfloat16)
    return {"pairs": int(seen), "pairs_total": len(seqs), "steps": step, "trainable_params": int(n_train),
            "minutes": (time.time() - t0) / 60, "pairs_per_s": (seen - t_resume_seen) / (time.time() - t0),
            "stopped_early": stopped_early}


def save(v, path: Path, lora: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    v.model.save_pretrained(path)
    v.tok.save_pretrained(path)


def build_training_pairs_r2(cand: pd.DataFrame, gt: dict, s1_ids, nonlatin: set, empty_addr: set, *,
                            n_hard: int = 8, n_rand: int = 2, oversample: int = 2, fn_dup: int = 2,
                            seed: int = 42) -> pd.DataFrame:
    """Round-2 training rows (D2) for the given S1.

    ``cand``: that fold's candidate pairs with ``y``, ``key`` (combined E1b / verifier score)
    and ``fp`` / ``fn`` flags (the sub3 OOF decision). Negatives are ranked current false
    positives first, then by ``key``; the top ``n_hard`` plus ``n_rand`` random others per S1.
    Every labelled positive enters (retrieved or not); current false negatives enter
    ``fn_dup`` times. Non-Latin / empty-address rows are repeated ``oversample`` times.
    """
    rng = np.random.default_rng(seed)
    ids = set(s1_ids)
    c = cand[cand["s1_id"].isin(ids)]
    neg = c[c["y"] == 0].sort_values(["s1_id", "fp", "key"], ascending=[True, False, False])
    neg = neg.assign(r=neg.groupby("s1_id").cumcount())
    hard = neg[neg["r"] < n_hard]
    rest = neg[neg["r"] >= n_hard]
    rest = rest.assign(u=rng.random(len(rest))).sort_values(["s1_id", "u"])
    rand = rest[rest.groupby("s1_id").cumcount() < n_rand]
    pos = pd.DataFrame([(s, m) for s in sorted(ids) for m in sorted(gt.get(s, ()))], columns=["s1_id", "m_id"])
    fn = c.loc[(c["y"] == 1) & c["fn"].astype(bool), ["s1_id", "m_id"]]
    rows = pd.concat([pos.assign(y=1)] + [fn.assign(y=1)] * max(0, fn_dup - 1)
                     + [hard[["s1_id", "m_id"]].assign(y=0), rand[["s1_id", "m_id"]].assign(y=0)], ignore_index=True)
    hot = (rows["s1_id"].isin(nonlatin) | rows["m_id"].isin(nonlatin) |
           rows["s1_id"].isin(empty_addr) | rows["m_id"].isin(empty_addr))
    rows = pd.concat([rows] + [rows[hot]] * (oversample - 1), ignore_index=True)
    return rows.sample(frac=1.0, random_state=seed).reset_index(drop=True)
