"""
Checkpoint 3 — proposed models vs the checkpoint-2 baselines (pooled).

For each regime (A = draft-only, B = draft + the 4 first-objective flags):
  * baselines (re-run for one unified table): LogisticRegression, BernoulliNB
        -> one-hot/multi-hot champion features, NO region (as in ckpt 2)
  * proposed, WITH region one-hot:
        - LightGBM       : gradient boosting on the same one-hot/multi-hot features
        - ChampPoolNet   : learned champion embeddings, mean-pooled per side,
                           concat [blue/red pick pools, blue/red ban pools,
                           spell multi-hot, region one-hot, objective flags (B)]

The embedding net and the one-hot models differ ONLY in how *champions*
(picks + bans) are represented (pooled embeddings vs multi-hot) — spells,
region and objectives are identical in both — so the embedding-vs-one-hot
contrast stays clean and ablatable (ckpt 4).

Run:  PYTHONPATH=src python3 src/proposed.py
"""
from __future__ import annotations
import os
# --- macOS OpenMP guard: LightGBM and PyTorch each bundle libomp; loading both
# --- in one process can deadlock. Force single-thread + allow lib coexistence.
# --- MUST run before importing lightgbm / torch.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import json
import random
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import lightgbm as lgb
import torch
import torch.nn as nn

from data_utils import load_matches, SEED, REGIONS, OBJECTIVE_COLS
import features as F

EMB_DIM = 32
HIDDEN = 64
MAX_EPOCHS = 40
PATIENCE = 6
BATCH = 512
LR = 1e-3
WD = 1e-5
DEVICE = "cpu"
torch.set_num_threads(1)   # single-thread (deadlock-safe alongside LightGBM)


def set_seed(s: int = SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def metrics(name, regime, yte, proba, nfeat):
    pred = (proba >= 0.5).astype(int)
    return {"model": name, "regime": regime, "nfeat": nfeat,
            "acc": accuracy_score(yte, pred),
            "f1_blue": f1_score(yte, pred, pos_label=1, zero_division=0),
            "f1_macro": f1_score(yte, pred, average="macro", zero_division=0),
            "auc": roc_auc_score(yte, proba)}


# --------------------------------------------------------------------------- net inputs
def build_champ_index(df_train) -> dict:
    """Unified champion vocab over picks AND bans (shared embedding table). 0 = pad."""
    names = set(F._vals(df_train, F.BLUE_CHAMP_COLS + F.RED_CHAMP_COLS).ravel()) - F._DROP_CHAMP
    names |= set(F._vals(df_train, F.BLUE_BAN_COLS + F.RED_BAN_COLS).ravel()) - set(
        ["None", "-1", "", "nan"])
    return {c: i + 1 for i, c in enumerate(sorted(names))}  # idx 0 reserved for pad/unknown


def _idx_block(df, cols, champ2idx):
    """Champion-name -> embedding index (1..V); unknown/drop tokens -> 0 pad (vectorized)."""
    vals = F._vals(df, cols)
    flat = pd.Series(vals.ravel()).map(champ2idx).to_numpy()   # NaN for non-champions
    return np.nan_to_num(flat, nan=0.0).astype(np.int64).reshape(vals.shape)


def encode_indices(df, champ2idx):
    bp = _idx_block(df, F.BLUE_CHAMP_COLS, champ2idx)
    rp = _idx_block(df, F.RED_CHAMP_COLS, champ2idx)
    bb = _idx_block(df, F.BLUE_BAN_COLS, champ2idx)
    rb = _idx_block(df, F.RED_BAN_COLS, champ2idx)
    return bp, rp, bb, rb


def build_sideinfo(df, vocabs, regime):
    """Dense [spell multi-hot (2x|spell|) | region one-hot (3) | objectives (8 if B)]."""
    n = len(df)
    spell = vocabs["spell"]; sidx = {s: i for i, s in enumerate(spell)}
    ns = len(spell)
    blocks = []
    for cols in (F.BLUE_SPELL_COLS, F.RED_SPELL_COLS):
        vals = F._vals(df, cols)
        m = np.zeros((n, ns), dtype=np.float32)
        colj = pd.Series(vals.ravel()).map(sidx).to_numpy()
        rr = np.repeat(np.arange(n), vals.shape[1])
        mask = ~pd.isna(colj)
        m[rr[mask], colj[mask].astype(int)] = 1.0
        blocks.append(m)
    reg = np.zeros((n, len(REGIONS)), dtype=np.float32)
    rv = df["region"].astype(str).to_numpy()
    for j, r in enumerate(REGIONS):
        reg[rv == r, j] = 1.0
    blocks.append(reg)
    if regime == "B":
        obj = np.zeros((n, 2 * len(OBJECTIVE_COLS)), dtype=np.float32)
        for k, o in enumerate(OBJECTIVE_COLS):
            ov = df[o].astype(float).to_numpy()
            obj[:, 2 * k] = (ov == 1.0)   # blue took it
            obj[:, 2 * k + 1] = (ov == 2.0)  # red took it
        blocks.append(obj)
    return np.concatenate(blocks, axis=1)


# --------------------------------------------------------------------------- the net
class ChampPoolNet(nn.Module):
    def __init__(self, vocab_size, side_dim, emb_dim=EMB_DIM, hidden=HIDDEN, use_embeddings=True):
        super().__init__()
        self.use_embeddings = use_embeddings
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        in_dim = 4 * emb_dim + side_dim           # blue/red pick + blue/red ban pools + side info
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden // 2, 1),
        )

    def _pool(self, idx):
        e = self.emb(idx)                          # (B, S, D)
        mask = (idx > 0).float().unsqueeze(-1)     # (B, S, 1)
        s = (e * mask).sum(1)
        c = mask.sum(1).clamp(min=1.0)
        return s / c

    def forward(self, bp, rp, bb, rb, side):
        z = torch.cat([self._pool(bp), self._pool(rp), self._pool(bb), self._pool(rb), side], dim=1)
        return self.mlp(z).squeeze(1)


def train_net(tr, va, te, vocab_size, side_dim, regime):
    set_seed(SEED)
    (bp_tr, rp_tr, bb_tr, rb_tr, s_tr, y_tr) = tr
    (bp_va, rp_va, bb_va, rb_va, s_va, y_va) = va
    (bp_te, rp_te, bb_te, rb_te, s_te) = te

    def T(a): return torch.as_tensor(a)
    model = ChampPoolNet(vocab_size, side_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    lossf = nn.BCEWithLogitsLoss()
    n = len(y_tr); idx = np.arange(n)
    best_auc, best_state, bad = -1.0, None, 0

    for ep in range(MAX_EPOCHS):
        model.train()
        rng = np.random.RandomState(SEED + ep); rng.shuffle(idx)
        for b in range(0, n, BATCH):
            j = idx[b:b + BATCH]
            opt.zero_grad()
            out = model(T(bp_tr[j]), T(rp_tr[j]), T(bb_tr[j]), T(rb_tr[j]), T(s_tr[j]))
            loss = lossf(out, T(y_tr[j]).float())
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            va_logit = model(T(bp_va), T(rp_va), T(bb_va), T(rb_va), T(s_va))
            va_auc = roc_auc_score(y_va, torch.sigmoid(va_logit).numpy())
        if va_auc > best_auc + 1e-4:
            best_auc, best_state, bad = va_auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    print(f"    [net {regime}] stopped at epoch {ep + 1}, best val AUC={best_auc:.4f}", flush=True)
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        proba = torch.sigmoid(model(T(bp_te), T(rp_te), T(bb_te), T(rb_te), T(s_te))).numpy()
    return proba


# --------------------------------------------------------------------------- main
def main():
    set_seed(SEED)
    df = load_matches()
    df = df[df["winner"].isin([1, 2])].copy()
    y = df["blue_win"].values

    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.20,
                                      random_state=SEED, stratify=y)
    df_tr, df_te = df.iloc[idx_tr], df.iloc[idx_te]
    ytr, yte = y[idx_tr], y[idx_te]
    # inner split for early stopping (proposed models only)
    fit_i, val_i = train_test_split(np.arange(len(df_tr)), test_size=0.10,
                                    random_state=SEED, stratify=ytr)
    df_fit, df_val = df_tr.iloc[fit_i], df_tr.iloc[val_i]
    yfit, yval = ytr[fit_i], ytr[val_i]

    vocabs = F.fit_vocabs(df_tr)
    champ2idx = build_champ_index(df_tr)
    vsize = len(champ2idx) + 1

    rows = []
    maj_acc = max(y.mean(), 1 - y.mean())
    rows.append({"model": "Majority (predict red)", "regime": "—", "nfeat": 0,
                 "acc": maj_acc, "f1_blue": 0.0,
                 "f1_macro": f1_score(yte, np.zeros_like(yte), average="macro", zero_division=0),
                 "auc": 0.5})

    print("=" * 96)
    print("CHECKPOINT 3 — proposed models (LightGBM + ChampPoolNet) vs baselines (pooled)")
    print("=" * 96)
    print(f"labeled matches={len(y)}  train={len(idx_tr)} (fit={len(fit_i)} val={len(val_i)})  test={len(idx_te)}")
    print(f"champ-embed vocab={vsize - 1} (+pad)  emb_dim={EMB_DIM}  hidden={HIDDEN}  seed={SEED}")
    print("baselines: NO region (as ckpt 2).  proposed (LightGBM, ChampPoolNet): WITH region one-hot.\n")

    for regime in ("A", "B"):
        t0 = time.time()
        rlabel = f"{regime} ({'draft-only' if regime == 'A' else 'draft+objectives'})"
        print(f"[regime {regime}] building features + training baselines/LightGBM/net...", flush=True)

        # ---- one-hot/multi-hot matrices ----
        Xtr_nr, _ = F.transform(df_tr, vocabs, regime, add_region=False)   # baselines (no region)
        Xte_nr, _ = F.transform(df_te, vocabs, regime, add_region=False)
        Xfit_r, _ = F.transform(df_fit, vocabs, regime, add_region=True)   # GBM (region)
        Xval_r, _ = F.transform(df_val, vocabs, regime, add_region=True)
        Xte_r, names_r = F.transform(df_te, vocabs, regime, add_region=True)

        # ---- baselines (re-run, identical to ckpt 2) ----
        for nm, clf in [("LogisticRegression", LogisticRegression(C=1.0, solver="liblinear",
                                                                  max_iter=1000, random_state=SEED)),
                        ("BernoulliNB", BernoulliNB())]:
            clf.fit(Xtr_nr, ytr)
            rows.append(metrics(f"{nm} (baseline)", rlabel, yte,
                                clf.predict_proba(Xte_nr)[:, 1], Xtr_nr.shape[1]))

        # ---- LightGBM (proposed, region) ----
        gbm = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.03, num_leaves=31,
                                 min_child_samples=50, subsample=0.8, subsample_freq=1,
                                 colsample_bytree=0.6, reg_lambda=2.0, random_state=SEED,
                                 n_jobs=1, verbose=-1)
        gbm.fit(Xfit_r, yfit, eval_set=[(Xval_r, yval)], eval_metric="auc",
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        rows.append(metrics("LightGBM (+region)", rlabel, yte,
                            gbm.predict_proba(Xte_r)[:, 1], Xte_r.shape[1]))

        # ---- ChampPoolNet (proposed, region, embeddings) ----
        side_fit = build_sideinfo(df_fit, vocabs, regime)
        side_val = build_sideinfo(df_val, vocabs, regime)
        side_te = build_sideinfo(df_te, vocabs, regime)
        bp_f, rp_f, bb_f, rb_f = encode_indices(df_fit, champ2idx)
        bp_v, rp_v, bb_v, rb_v = encode_indices(df_val, champ2idx)
        bp_t, rp_t, bb_t, rb_t = encode_indices(df_te, champ2idx)
        proba = train_net(
            (bp_f, rp_f, bb_f, rb_f, side_fit, yfit),
            (bp_v, rp_v, bb_v, rb_v, side_val, yval),
            (bp_t, rp_t, bb_t, rb_t, side_te),
            vsize, side_fit.shape[1], regime)
        nfeat_net = 4 * EMB_DIM + side_fit.shape[1]
        rows.append(metrics("ChampPoolNet (+region, emb)", rlabel, yte, proba, nfeat_net))
        print(f"[regime {regime}] done in {time.time() - t0:.1f}s", flush=True)

    # ---- unified table ----
    out = []
    out.append(f"{'regime':<20}{'model':<30}{'nfeat':<8}{'acc':<9}{'F1_blue':<10}{'F1_macro':<10}{'AUC':<8}")
    out.append("-" * 95)
    order = {"—": 0, "A (draft-only)": 1, "B (draft+objectives)": 2}
    rows_sorted = sorted(rows, key=lambda r: (order.get(r["regime"], 9),))
    last = None
    for r in rows_sorted:
        reg = r["regime"] if r["regime"] != last else ""
        last = r["regime"]
        d = "" if r["regime"] == "—" else f"  ({r['acc'] - maj_acc:+.4f} vs floor)"
        out.append(f"{reg:<20}{r['model']:<30}{r['nfeat']:<8}"
                   f"{r['acc']:.4f}   {r['f1_blue']:.4f}    {r['f1_macro']:.4f}    {r['auc']:.4f}{d}")
    out.append("-" * 95)
    out.append(f"Floor = always-predict-red = {maj_acc:.4f} acc. Baselines: no region. "
               f"LightGBM & ChampPoolNet: + region one-hot.")
    out.append("Lit. context: combined pre-game + early in-game win models report ~72-75% acc; "
               "regime B here lands in that band.")
    table = "\n".join(out)
    print("\n" + table, flush=True)
    with open(os.path.join("models", "ckpt3_table.txt"), "w") as fh:
        fh.write(table + "\n")
    with open(os.path.join("models", "ckpt3_rows.json"), "w") as fh:
        json.dump(rows_sorted, fh, indent=2)
    print("\n[written] models/ckpt3_table.txt  models/ckpt3_rows.json", flush=True)


if __name__ == "__main__":
    main()
