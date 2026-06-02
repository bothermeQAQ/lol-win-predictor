"""
Team-symmetric feature construction for the DSC148 LoL winner classifier.

Design goals (from the project spec):
  * The model must NOT be able to memorize slot positions. So within one side we
    use an order-invariant BAG (multi-hot): "blue has champion X" — never "champion
    X is in slot 3".
  * Blue and red occupy DISTINCT column blocks, so the genuine side prior
    (red wins ~54%) and side-specific champion strength are preserved.

Two feature regimes:
  A  draft-only           : champion picks + bans + summoner spells, per side.
  B  draft + objectives   : A  +  the four first-objective flags, each encoded
                            symmetrically as (blue-got-it, red-got-it).

Vocabularies are FIT ON TRAIN ONLY and reused for the test set (any champion /
spell unseen in train simply maps to no column), so there is no test->train leak.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import scipy.sparse as sp

from data_utils import (
    BLUE_CHAMP_COLS, RED_CHAMP_COLS, SLOTS, OBJECTIVE_COLS, NO_BAN_TOKENS, REGIONS,
)

BLUE_BAN_COLS = [f"t1_ban{i}" for i in SLOTS]
RED_BAN_COLS = [f"t2_ban{i}" for i in SLOTS]
BLUE_SPELL_COLS = [f"t1_champ{i}_spell{s}" for i in SLOTS for s in (1, 2)]
RED_SPELL_COLS = [f"t2_champ{i}_spell{s}" for i in SLOTS for s in (1, 2)]

_DROP_CHAMP = {"", "nan", "None"}
_DROP_SPELL = {"", "nan", "0", "-1"}


def _vals(df, cols) -> np.ndarray:
    """(n, k) string matrix for the requested columns.

    The trailing numpy ``.astype(str)`` normalizes the object array that pandas
    3.0 StringDtype produces for missing cells (a real ``float('nan')``) into the
    literal string ``"nan"``, which every drop-set below already excludes.
    """
    return df[cols].astype(str).values.astype(str)


def fit_vocabs(df) -> dict:
    """Build champion / ban / summoner-spell vocabularies from a (train) frame."""
    champ = sorted(set(_vals(df, BLUE_CHAMP_COLS + RED_CHAMP_COLS).ravel()) - _DROP_CHAMP)
    ban = sorted(set(_vals(df, BLUE_BAN_COLS + RED_BAN_COLS).ravel()) - NO_BAN_TOKENS)
    spell = sorted(set(_vals(df, BLUE_SPELL_COLS + RED_SPELL_COLS).ravel()) - _DROP_SPELL,
                   key=lambda x: (len(x), x))
    return {"champ": champ, "ban": ban, "spell": spell}


def feature_names(vocabs: dict, regime: str, add_region: bool = False,
                  include_spells: bool = True) -> list:
    """Ordered feature-name list defining the column layout for a regime.

    ``include_spells=False`` drops the summoner-spell block (used by the
    checkpoint-5 demo, whose UI does not collect spells — keeping the encoder
    exactly aligned with the inputs the demo provides). Default True preserves
    the checkpoint 2-4 layout.
    """
    names = []
    for side in ("blue", "red"):
        names += [f"{side}_pick={c}" for c in vocabs["champ"]]
    for side in ("blue", "red"):
        names += [f"{side}_ban={c}" for c in vocabs["ban"]]
    if include_spells:
        for side in ("blue", "red"):
            names += [f"{side}_spell={s}" for s in vocabs["spell"]]
    if regime == "B":
        for obj in OBJECTIVE_COLS:
            names += [f"{obj}_blue", f"{obj}_red"]
    elif regime != "A":
        raise ValueError(f"regime must be 'A' or 'B', got {regime!r}")
    if add_region:
        names += [f"region={r}" for r in REGIONS]
    return names


def _multi_hot_block(df, side_cols, vocab, prefix, name_to_col, tri):
    """Append (row_idx, col_idx) arrays for one multi-hot block (vectorized).

    `tri` is a 2-tuple of lists (row-array list, col-array list). Tokens with no
    column (out-of-vocab / drop tokens) map to NaN and are filtered out.
    """
    vals = _vals(df, side_cols)                       # (n, k) unicode
    n, k = vals.shape
    tok2col = {tok: name_to_col[f"{prefix}={tok}"] for tok in vocab}
    col_arr = pd.Series(vals.ravel()).map(tok2col).to_numpy()   # float w/ NaN where no col
    row_arr = np.repeat(np.arange(n), k)
    mask = ~pd.isna(col_arr)
    tri[0].append(row_arr[mask])
    tri[1].append(col_arr[mask].astype(np.int64))


def transform(df, vocabs: dict, regime: str, add_region: bool = False,
              include_spells: bool = True):
    """Return (X csr float32, names) for the given regime over `df` (vectorized).

    ``include_spells=False`` omits the summoner-spell multi-hot blocks (demo).
    """
    names = feature_names(vocabs, regime, add_region=add_region, include_spells=include_spells)
    name_to_col = {nm: i for i, nm in enumerate(names)}
    n = len(df)
    tri = ([], [])                                    # (row-arrays, col-arrays)

    # ---- champion picks / bans / spells (order-invariant bag, per side) ----
    _multi_hot_block(df, BLUE_CHAMP_COLS, vocabs["champ"], "blue_pick", name_to_col, tri)
    _multi_hot_block(df, RED_CHAMP_COLS, vocabs["champ"], "red_pick", name_to_col, tri)
    _multi_hot_block(df, BLUE_BAN_COLS, vocabs["ban"], "blue_ban", name_to_col, tri)
    _multi_hot_block(df, RED_BAN_COLS, vocabs["ban"], "red_ban", name_to_col, tri)
    if include_spells:
        _multi_hot_block(df, BLUE_SPELL_COLS, vocabs["spell"], "blue_spell", name_to_col, tri)
        _multi_hot_block(df, RED_SPELL_COLS, vocabs["spell"], "red_spell", name_to_col, tri)

    # ---- early objectives (regime B only), symmetric (blue/red) ----
    if regime == "B":
        for obj in OBJECTIVE_COLS:
            ov = df[obj].astype(float).to_numpy()
            br = np.where(ov == 1.0)[0]               # 1 == blue/team1 first
            rr = np.where(ov == 2.0)[0]               # 2 == red/team2 first
            tri[0].append(br); tri[1].append(np.full(br.shape, name_to_col[f"{obj}_blue"], np.int64))
            tri[0].append(rr); tri[1].append(np.full(rr.shape, name_to_col[f"{obj}_red"], np.int64))

    # ---- region one-hot (optional; pooled models only) ----
    if add_region:
        rv = df["region"].astype(str).to_numpy()
        for r in REGIONS:
            idxs = np.where(rv == r)[0]
            tri[0].append(idxs)
            tri[1].append(np.full(idxs.shape, name_to_col[f"region={r}"], np.int64))

    rows = np.concatenate(tri[0]) if tri[0] else np.empty(0, np.int64)
    cols = np.concatenate(tri[1]) if tri[1] else np.empty(0, np.int64)
    data = np.ones(rows.shape[0], dtype=np.float32)
    X = sp.coo_matrix((data, (rows, cols)), shape=(n, len(names)), dtype=np.float32).tocsr()
    X.data[:] = 1.0                                   # binarize (collapse repeated spell tokens)
    return X, names
