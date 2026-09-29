"""B4: exact-key blocking with a block-size cap.

Keys (computed per target partition, so "rare" means rare among the targets):
    hn_tok   house_no + the rarest non-numeric address token (df > 0 in targets)
    name_tok the rarest name_core token of length >= 3 (df > 0 in targets)
Blocks whose target count exceeds ``cap`` are skipped entirely (too generic).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _explode_tokens(texts: pd.Series, min_len: int, numeric: bool = False) -> pd.DataFrame:
    toks = texts.str.split()
    ex = toks.explode().dropna()
    ex = ex[ex.str.len() >= min_len]
    if not numeric:
        ex = ex[~ex.str.isdigit()]
    return pd.DataFrame({"row": ex.index.to_numpy(), "tok": ex.to_numpy()}).drop_duplicates()


def _rarest(tok_df: pd.DataFrame, df_counts: pd.Series) -> pd.Series:
    """row -> rarest token (lowest df > 0; ties broken alphabetically)."""
    t = tok_df.assign(df=tok_df["tok"].map(df_counts).fillna(0).astype(np.int64))
    t = t[t["df"] > 0].sort_values(["row", "df", "tok"])
    return t.drop_duplicates("row").set_index("row")["tok"]


def key_pairs(
    q_addr: pd.Series, q_house: pd.Series, q_name: pd.Series,
    t_addr: pd.Series, t_house: pd.Series, t_name: pd.Series,
    cap: int = 500,
) -> pd.DataFrame:
    """All (q, t) pairs sharing a key; columns q, t, b4_hn, b4_name, b4_size (min block size)."""
    q_addr, q_house, q_name = (s.reset_index(drop=True) for s in (q_addr, q_house, q_name))
    t_addr, t_house, t_name = (s.reset_index(drop=True) for s in (t_addr, t_house, t_name))
    out = []

    # --- house number + rarest address token
    t_at = _explode_tokens(t_addr, 3)
    adf = t_at["tok"].value_counts()
    t_rare = _rarest(t_at, adf)
    q_rare = _rarest(_explode_tokens(q_addr, 3), adf)
    tk = pd.DataFrame({"t": t_rare.index, "key": t_house.iloc[t_rare.index].to_numpy() + "|" + t_rare.to_numpy()})
    tk = tk[~tk["key"].str.startswith("|")]
    qk = pd.DataFrame({"q": q_rare.index, "key": q_house.iloc[q_rare.index].to_numpy() + "|" + q_rare.to_numpy()})
    qk = qk[~qk["key"].str.startswith("|")]
    size = tk["key"].value_counts()
    tk = tk[tk["key"].map(size) <= cap]
    m = qk.merge(tk, on="key")
    out.append(pd.DataFrame({"q": m["q"], "t": m["t"], "b4_hn": True, "b4_name": False,
                             "b4_size": m["key"].map(size).to_numpy()}))

    # --- rarest name token
    t_nt = _explode_tokens(t_name, 3)
    ndf = t_nt["tok"].value_counts()
    t_rare = _rarest(t_nt, ndf)
    q_rare = _rarest(_explode_tokens(q_name, 3), ndf)
    # a query matches every target CONTAINING its rarest token (not only targets whose
    # own rarest token is the same), limited to blocks of at most `cap` targets
    small = ndf[ndf <= cap]
    tt = t_nt[t_nt["tok"].isin(small.index)]
    qk = pd.DataFrame({"q": q_rare.index, "tok": q_rare.to_numpy()})
    m = qk.merge(tt.rename(columns={"row": "t"}), on="tok")
    out.append(pd.DataFrame({"q": m["q"], "t": m["t"], "b4_hn": False, "b4_name": True,
                             "b4_size": m["tok"].map(ndf).to_numpy()}))
    del t_rare

    res = pd.concat(out, ignore_index=True)
    if res.empty:
        return pd.DataFrame(columns=["q", "t", "b4_hn", "b4_name", "b4_size"])
    res = res.groupby(["q", "t"], as_index=False).agg(
        b4_hn=("b4_hn", "max"), b4_name=("b4_name", "max"), b4_size=("b4_size", "min")
    )
    return res
