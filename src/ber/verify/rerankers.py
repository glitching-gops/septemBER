"""Verifier models: one uniform ``score(left, right) -> {"logit", "p"[, "p_contra"]}`` API.

Three readouts:

* ``qwen_yesno``: Qwen3-Reranker. A causal LM asked a yes/no question with the model
  card's exact system prefix and assistant suffix; the score is the yes-vs-no logit at the
  last position (P(yes) = softmax over the two). Left padding, as on the model card.
* ``cross``: encoder cross-encoders (gte, bge). One logit from the classification head.
* ``nli``: openjev (Qwen3.5 sequence classifier, labels contradiction/entailment/neutral).
  P(entailment) is the score; P(contradiction) is returned too. Right padding: the head
  pools the last non-pad token and Qwen3.5's linear-attention layers must not see padding
  before real tokens.

All inference is bf16 with SDPA, no packing, length-bucketed batches under a token budget.
"""
from __future__ import annotations

import time

import numpy as np

INSTRUCTION = (
    "Decide whether Record B refers to the same real-world business (same legal entity at the "
    "same location) as Record A. Names may be abbreviated, misspelled, transliterated or in "
    "another script; a similar name at a different address is usually not a match.")
QWEN_PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the "
               "Query and the Instruct provided. Note that the answer can only be \"yes\" or "
               "\"no\".<|im_end|>\n<|im_start|>user\n")
QWEN_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

# Exact parameter counts from the safetensors headers (checked 2026-09-26); licences from
# the model cards. Revisions pin the snapshot that was downloaded and screened.
MODELS = {
    "qwen3-rr-0.6b": dict(hf="Qwen/Qwen3-Reranker-0.6B", kind="qwen_yesno", style="plain",
                          licence="Apache-2.0", params=595_776_512,
                          revision="e61197ed45024b0ed8a2d74b80b4d909f1255473"),
    "bge-rr-v2-m3": dict(hf="BAAI/bge-reranker-v2-m3", kind="cross", style="ditto",
                         licence="Apache-2.0", params=567_755_777,
                         revision="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"),
    "gte-mr-base": dict(hf="Alibaba-NLP/gte-multilingual-reranker-base", kind="cross", style="ditto",
                        licence="Apache-2.0", params=305_959_681, remote_code=True,
                        revision="8215cf04918ba6f7b6a62bb44238ce2953d8831c"),
    "qwen3-rr-4b": dict(hf="Qwen/Qwen3-Reranker-4B", kind="qwen_yesno", style="plain",
                        licence="Apache-2.0", params=4_021_784_576, revision=None),
    "openjev-2b-nli": dict(hf="AlexWortega/openjev", subfolder="qwen3.5-2b-nli-v5", kind="nli",
                           style="plain", licence="MIT", params=2_213_247_808,
                           revision="058a6c24911b46d908fbe23541390f8af3df3e4d"),
}


def _shim_extended_mask(model) -> None:
    """Make gte's pinned remote code (written for transformers 4.39) run under transformers 5.

    It calls ``get_extended_attention_mask``, which transformers 5 removed from
    ``PreTrainedModel``: re-add the transformers-4 behaviour (additive 0 / -inf bias over key
    positions) on the modules that lack it.
    """
    import torch

    def ext(self, attention_mask, input_shape, *args, **kwargs):
        m = attention_mask[:, None, None, :].to(self.dtype)
        return (1.0 - m) * torch.finfo(self.dtype).min

    for mod in model.modules():
        if hasattr(mod, "config") and not hasattr(mod, "get_extended_attention_mask"):
            type(mod).get_extended_attention_mask = ext
    # transformers 5 builds the model on the meta device, so the remote code's NON-persistent
    # buffers (rotary inv_freq / cos / sin caches, position ids) are never materialised from
    # the checkpoint and hold garbage. Recompute them exactly as the module's __init__ does.
    dev = next(model.parameters()).device
    for mod in model.modules():
        if hasattr(mod, "_set_cos_sin_cache") and hasattr(mod, "inv_freq"):
            mod.inv_freq = 1.0 / (mod.base ** (torch.arange(0, mod.dim, 2, device=dev).float() / mod.dim))
            n = int(mod.max_position_embeddings * getattr(mod, "scaling_factor", 1.0))
            mod._set_cos_sin_cache(n, dev, torch.float32)
        elif "position_ids" in dict(mod.named_buffers(recurse=False)):
            mod.position_ids = torch.arange(mod.position_ids.numel(), device=dev)


class Verifier:
    """Load one model from ``MODELS`` onto ``device`` (bf16) and score text pairs."""

    def __init__(self, key: str, device, max_len: int = 128, model=None, tokenizer=None):
        import torch
        from transformers import (AutoModelForCausalLM, AutoModelForSequenceClassification,
                                  AutoTokenizer)

        self.key, self.spec, self.device, self.max_len = key, MODELS[key], device, max_len
        s = self.spec
        kw = dict(revision=s.get("revision"), trust_remote_code=bool(s.get("remote_code")))
        if s.get("subfolder"):
            kw["subfolder"] = s["subfolder"]
        self.kind = s["kind"]
        self.tok = tokenizer or AutoTokenizer.from_pretrained(s["hf"], **kw)
        if model is not None:
            self.model = model
        elif self.kind == "qwen_yesno":
            self.model = AutoModelForCausalLM.from_pretrained(
                s["hf"], dtype=torch.bfloat16, attn_implementation="sdpa", **kw)
        else:
            extra = {} if s.get("remote_code") else {"attn_implementation": "sdpa"}
            self.model = AutoModelForSequenceClassification.from_pretrained(
                s["hf"], dtype=torch.bfloat16, **extra, **kw)
        self.model.to(device).eval()
        if s.get("remote_code"):
            _shim_extended_mask(self.model)
        if self.kind == "qwen_yesno":
            self.tok.padding_side = "left"
            self.yes = self.tok.convert_tokens_to_ids("yes")
            self.no = self.tok.convert_tokens_to_ids("no")
            self.pre = self.tok.encode(QWEN_PREFIX, add_special_tokens=False)
            self.suf = self.tok.encode(QWEN_SUFFIX, add_special_tokens=False)
            self.instr = self.tok.encode(f"<Instruct>: {INSTRUCTION}\n<Query>: ", add_special_tokens=False)
        else:
            self.tok.padding_side = "right"
        if self.kind == "nli":
            cfg = self.model.config
            self.template = getattr(cfg, "nli_template", None) or "Premise: {premise}\nHypothesis: {hypothesis}"
            lab = {v.lower(): int(k) for k, v in cfg.id2label.items()}
            self.i_ent, self.i_con = lab["entailment"], lab["contradiction"]
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

    # ---- tokenisation -------------------------------------------------------------------
    def encode(self, left: list[str], right: list[str]) -> list[list[int]]:
        """Token ids per pair (unpadded). ``max_len`` bounds the two records' tokens."""
        half = self.max_len // 2
        if self.kind == "qwen_yesno":
            a = self.tok(left, add_special_tokens=False, truncation=True, max_length=half)["input_ids"]
            b = self.tok([f"\n<Document>: {x}" for x in right], add_special_tokens=False,
                         truncation=True, max_length=half)["input_ids"]
            return [self.pre + self.instr + x + y + self.suf for x, y in zip(a, b)]
        if self.kind == "nli":
            texts = [self.template.format(premise=f"Record A: {x}",
                                          hypothesis=f"Record B ({y}) is the same business as Record A.")
                     for x, y in zip(left, right)]
            return self.tok(texts, truncation=True, max_length=self.max_len + 32)["input_ids"]
        return self.tok(left, right, truncation=True, max_length=self.max_len)["input_ids"]

    def collate(self, seqs: list[list[int]]):
        import torch

        L = max(len(s) for s in seqs)
        pad = self.tok.pad_token_id
        ids = torch.full((len(seqs), L), pad, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, s in enumerate(seqs):
            if self.tok.padding_side == "left":
                ids[i, L - len(s):] = torch.tensor(s)
                att[i, L - len(s):] = 1
            else:
                ids[i, :len(s)] = torch.tensor(s)
                att[i, :len(s)] = 1
        return ids.to(self.device), att.to(self.device)

    # ---- forward ------------------------------------------------------------------------
    def logits(self, ids, att):
        """Raw pair logit (yes-no for qwen, head logit for cross, ent-con for nli) + extras."""
        import torch

        if self.kind == "qwen_yesno":
            # only the last position is read: never materialise [batch, seq, 151k] logits
            lg = self.model(input_ids=ids, attention_mask=att, logits_to_keep=1, use_cache=False).logits
            last = lg[:, -1, :]
            return (last[:, self.yes] - last[:, self.no]).float(), None
        lg = self.model(input_ids=ids, attention_mask=att).logits
        if self.kind == "nli":
            pr = torch.softmax(lg.float(), -1)
            return (lg[:, self.i_ent] - lg[:, self.i_con]).float(), pr
        return lg[:, 0].float(), None

    def score_prefix_cached(self, left: list[str], right: list[str], token_budget: int = 24_000,
                            log_every: int = 0) -> dict:
        """Qwen yes/no scoring with the shared prefix (system prompt + instruction) encoded ONCE.

        Every pair starts with the same ~100 tokens; their KV cache is computed once and
        repeated across each batch, and only the pair-specific tail runs. Tails are
        right-padded (positions continue from the prefix, as in the full sequence), the last
        real token's hidden state is gathered and projected onto (W_yes - W_no) only.
        Equivalent to :meth:`score` up to bf16 rounding (checked on real pairs before use).
        """
        import copy

        import torch
        from transformers import DynamicCache

        assert self.kind == "qwen_yesno"
        half = self.max_len // 2
        a = self.tok(left, add_special_tokens=False, truncation=True, max_length=half)["input_ids"]
        b = self.tok([f"\n<Document>: {x}" for x in right], add_special_tokens=False,
                     truncation=True, max_length=half)["input_ids"]
        tails = [x + y + self.suf for x, y in zip(a, b)]
        prefix = torch.tensor([self.pre + self.instr], device=self.device)
        base = self.model.model if hasattr(self.model, "model") else self.model
        w = self.model.get_output_embeddings().weight
        wdiff = (w[self.yes] - w[self.no]).float()
        with torch.inference_mode():
            pc = DynamicCache()
            base(input_ids=prefix, past_key_values=pc, use_cache=True)
        P = prefix.shape[1]
        order = np.argsort([len(s) for s in tails], kind="stable")
        logit = np.zeros(len(tails), np.float32)
        t0, done, i = time.time(), 0, 0
        with torch.inference_mode():
            while i < len(order):
                j = i + 1
                while j < len(order) and (len(tails[order[j]]) + P // 4) * (j + 1 - i) <= token_budget:
                    j += 1
                idx = order[i:j]
                B = len(idx)
                L = max(len(tails[k]) for k in idx)
                ids = torch.full((B, L), self.tok.pad_token_id, dtype=torch.long)
                att = torch.zeros((B, P + L), dtype=torch.long)
                att[:, :P] = 1
                last = torch.zeros(B, dtype=torch.long)
                for r, k in enumerate(idx):
                    n = len(tails[k])
                    ids[r, :n] = torch.tensor(tails[k])
                    att[r, P:P + n] = 1
                    last[r] = n - 1
                cache = copy.deepcopy(pc)
                cache.batch_repeat_interleave(B)
                h = base(input_ids=ids.to(self.device), attention_mask=att.to(self.device),
                         past_key_values=cache, use_cache=True).last_hidden_state
                hl = h[torch.arange(B, device=self.device), last.to(self.device)]
                logit[idx] = (hl.float() @ wdiff).cpu().numpy()
                done += B
                i = j
                if log_every and done // log_every != (done - B) // log_every:
                    r = done / (time.time() - t0)
                    print(f"    [{self.key}] {done:,}/{len(order):,} pairs, {r:,.0f}/s, "
                          f"eta {(len(order) - done) / r / 60:.1f} min", flush=True)
        return {"logit": logit, "p": 1.0 / (1.0 + np.exp(-logit))}

    def score(self, left: list[str], right: list[str], token_budget: int = 24_000,
              log_every: int = 0, batch_size: int | None = None) -> dict:
        """Score pairs; returns numpy arrays in input order.

        ``batch_size`` (deterministic mode): FIXED-size batches over the stable length sort, so
        each pair's batch neighbours - hence its padded length and kernel shapes - depend only
        on the input list. Without it, batches are cut by ``token_budget`` (also a function of
        the input only, but of varying size).
        """
        import torch

        if self.kind == "qwen_yesno" and getattr(self, "prefix_cache", False) and not self.model.training:
            return self.score_prefix_cached(left, right, token_budget, log_every)
        _t, _c = time.time(), time.process_time()
        seqs = self.encode(left, right)
        self.last_tok_s = time.time() - _t  # tokenisation phase (host, timed separately)
        self.last_tok_cpu = time.process_time() - _c
        # stable sort by (length, input position): the batch plan is a pure function of the input
        order = np.argsort([len(s) for s in seqs], kind="stable")
        logit = np.zeros(len(seqs), np.float32)
        p_ent = np.zeros(len(seqs), np.float32)
        p_con = np.zeros(len(seqs), np.float32)
        t0, done, i = time.time(), 0, 0
        with torch.inference_mode():
            while i < len(order):
                if batch_size:
                    j = min(len(order), i + batch_size)
                else:
                    # sorted ascending, so the padded batch costs len(last member) x batch size
                    j = i + 1
                    while j < len(order) and len(seqs[order[j]]) * (j + 1 - i) <= token_budget:
                        j += 1
                idx = order[i:j]
                ids, att = self.collate([seqs[k] for k in idx])
                lg, pr = self.logits(ids, att)
                logit[idx] = lg.cpu().numpy()
                if pr is not None:
                    p_ent[idx] = pr[:, self.i_ent].cpu().numpy()
                    p_con[idx] = pr[:, self.i_con].cpu().numpy()
                done += len(idx)
                i = j
                if log_every and done // log_every != (done - len(idx)) // log_every:
                    r = done / (time.time() - t0)
                    print(f"    [{self.key}] {done:,}/{len(order):,} pairs, {r:,.0f}/s, "
                          f"eta {(len(order) - done) / r / 60:.1f} min", flush=True)
        out = {"logit": logit}
        out["p"] = p_ent if self.kind == "nli" else 1.0 / (1.0 + np.exp(-logit))
        if self.kind == "nli":
            out["p_contra"] = p_con
        return out
