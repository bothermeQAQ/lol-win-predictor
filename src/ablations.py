"""
Checkpoint 4 — ablations, significance, cross-region transfer, patch robustness,
hyper-parameter sensitivity, and case studies.

Everything reuses the SAME stratified 80/20 split (seed 42), the SAME inner
fit/val split for early stopping, and vocab/champ-index FIT ON TRAIN ONLY — so
every contrast below is "feature-matched + reproducible".

Sections (each writes its own models/ckpt4_*.txt and is echoed to the combined
models/ckpt4_summary.txt as it completes, so partial progress is never lost):

  1  Ablation grid           : LogReg/NB/LightGBM/ChampPoolNet(emb)/OneHotMLP,
                               regimes A & B, each WITH and WITHOUT region.
                               (OneHotMLP = same MLP as ChampPoolNet but champions
                                as multi-hot instead of pooled embeddings -> the
                                clean embedding-vs-one-hot swap.)
  2  Significance            : 1000x bootstrap 95% CI for every model's test AUC;
                               paired-bootstrap AUC differences (+ two-sided p) for
                               the headline contrasts. Regime A judged on AUC.
                               Explicit region-vs-nonlinearity decomposition on A.
  3  Embedding vs one-hot    : pulled out as its own matched table + paired test.
  4  Leakage CONTROL (labeled): LightGBM on regime-B features + leaky end-of-game
                               columns (kill counts, firstInhibitor/Baron, duration)
                               -> shows how high leakage inflates the numbers.
                               NOT a real result; for the writeup contrast only.
  5  Cross-region 3x3        : strongest draft-only model (regime-A LightGBM, NO
                               region) trained per single region, tested on all
                               three. acc/F1/AUC matrix. Regime-B matrix as control.
  6  Patch robustness        : per-patch (16.9/16.10/16.11) test metrics for the
                               pooled regime-B LightGBM; plus does adding a patch
                               one-hot to the pooled model move AUC?
  7  Hyper-parameter sweep   : regime-B LightGBM 1-D sweeps over learning_rate,
                               num_leaves, n_estimators.
  8  Case studies            : confident-correct / confident-wrong (upset) /
                               split-objective regime-B matches, decoded.

Run:  PYTHONPATH=src python3 src/ablations.py
"""
from __future__ import annotations
import os
# --- macOS OpenMP guard (MUST precede lightgbm / torch imports; proposed.py
# --- re-asserts the same setdefaults on import).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import json
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import lightgbm as lgb
import torch
import torch.nn as nn

from data_utils import (
    load_matches, SEED, REGIONS, OBJECTIVE_COLS, LEAKY_KILL_COLS,
)
import features as F
import proposed as P   # reuse ChampPoolNet, train_net, encode_indices, build_champ_index, metrics

MODELS_DIR = "models"
DEVICE = "cpu"
N_BOOT = 1000


# ===========================================================================
# small shared helpers
# ===========================================================================
def side_info(df, vocabs, regime, add_region):
    """Dense non-champion side features: [spell multi-hot | region? | objectives?].

    Identical to proposed.build_sideinfo but with an explicit add_region flag so
    the ChampPoolNet region ablation is feature-matched against the GBM grid.
    """
    n = len(df)
    spell = vocabs["spell"]; sidx = {s: i for i, s in enumerate(spell)}; ns = len(spell)
    blocks = []
    for cols in (F.BLUE_SPELL_COLS, F.RED_SPELL_COLS):
        vals = F._vals(df, cols)
        m = np.zeros((n, ns), dtype=np.float32)
        colj = pd.Series(vals.ravel()).map(sidx).to_numpy()
        rr = np.repeat(np.arange(n), vals.shape[1])
        mask = ~pd.isna(colj)
        m[rr[mask], colj[mask].astype(int)] = 1.0
        blocks.append(m)
    if add_region:
        reg = np.zeros((n, len(REGIONS)), dtype=np.float32)
        rv = df["region"].astype(str).to_numpy()
        for j, r in enumerate(REGIONS):
            reg[rv == r, j] = 1.0
        blocks.append(reg)
    if regime == "B":
        obj = np.zeros((n, 2 * len(OBJECTIVE_COLS)), dtype=np.float32)
        for k, o in enumerate(OBJECTIVE_COLS):
            ov = df[o].astype(float).to_numpy()
            obj[:, 2 * k] = (ov == 1.0)
            obj[:, 2 * k + 1] = (ov == 2.0)
        blocks.append(obj)
    return np.concatenate(blocks, axis=1)


class OneHotMLP(nn.Module):
    """Same MLP head as ChampPoolNet, but consumes the raw multi-hot feature
    matrix (champions one-hot) instead of pooled champion embeddings. This is the
    feature-matched one-hot counterpart for the embedding ablation."""
    def __init__(self, in_dim, hidden=P.HIDDEN):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.mlp(x).squeeze(1)


def train_onehot_mlp(Xfit, yfit, Xval, yval, Xte, regime, tag):
    """Train OneHotMLP on dense multi-hot features (early stop on val AUC)."""
    P.set_seed(SEED)
    Xf = Xfit.toarray().astype(np.float32)
    Xv = Xval.toarray().astype(np.float32)
    Xt = Xte.toarray().astype(np.float32)
    model = OneHotMLP(Xf.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=P.LR, weight_decay=P.WD)
    lossf = nn.BCEWithLogitsLoss()
    n = len(yfit); idx = np.arange(n)
    best, best_state, bad = -1.0, None, 0

    def T(a): return torch.as_tensor(a)
    for ep in range(P.MAX_EPOCHS):
        model.train()
        rng = np.random.RandomState(SEED + ep); rng.shuffle(idx)
        for b in range(0, n, P.BATCH):
            j = idx[b:b + P.BATCH]
            opt.zero_grad()
            out = model(T(Xf[j]))
            loss = lossf(out, T(yfit[j]).float())
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            va = roc_auc_score(yval, torch.sigmoid(model(T(Xv))).numpy())
        if va > best + 1e-4:
            best, best_state, bad = va, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= P.PATIENCE:
                break
    print(f"    [onehot {regime} {tag}] stop ep{ep + 1} val AUC={best:.4f}", flush=True)
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(T(Xt))).numpy()


def fit_gbm(Xfit, yfit, Xval, yval, early_stop=True, **kw):
    params = dict(n_estimators=2000, learning_rate=0.03, num_leaves=31,
                  min_child_samples=50, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.6, reg_lambda=2.0, random_state=SEED,
                  n_jobs=1, verbose=-1)
    params.update(kw)
    gbm = lgb.LGBMClassifier(**params)
    if early_stop:
        gbm.fit(Xfit, yfit, eval_set=[(Xval, yval)], eval_metric="auc",
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    else:
        gbm.fit(Xfit, yfit)
    return gbm


def m_all(yte, proba):
    pred = (proba >= 0.5).astype(int)
    return {"acc": accuracy_score(yte, pred),
            "f1_blue": f1_score(yte, pred, pos_label=1, zero_division=0),
            "f1_macro": f1_score(yte, pred, average="macro", zero_division=0),
            "auc": roc_auc_score(yte, proba)}


# ---- significance ---------------------------------------------------------
def boot_auc_ci(y, s, n_boot=N_BOOT, seed=SEED):
    rng = np.random.RandomState(seed)
    y = np.asarray(y); s = np.asarray(s); N = len(y)
    base = roc_auc_score(y, s)
    aucs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, N, N)
        yi = y[idx]
        aucs[b] = np.nan if yi.min() == yi.max() else roc_auc_score(yi, s[idx])
    lo, hi = np.nanpercentile(aucs, [2.5, 97.5])
    return base, lo, hi


def boot_auc_diff(y, sa, sb, n_boot=N_BOOT, seed=SEED):
    """Paired bootstrap of AUC(a) - AUC(b) on the same resampled test rows."""
    rng = np.random.RandomState(seed)
    y = np.asarray(y); sa = np.asarray(sa); sb = np.asarray(sb); N = len(y)
    base = roc_auc_score(y, sa) - roc_auc_score(y, sb)
    d = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, N, N)
        yi = y[idx]
        if yi.min() == yi.max():
            d[b] = np.nan; continue
        d[b] = roc_auc_score(yi, sa[idx]) - roc_auc_score(yi, sb[idx])
    d = d[~np.isnan(d)]
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = min(1.0, 2.0 * min((d <= 0).mean(), (d >= 0).mean()))
    return base, lo, hi, p


# ===========================================================================
def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    t_start = time.time()
    P.set_seed(SEED)
    df = load_matches()
    df = df[df["winner"].isin([1, 2])].copy().reset_index(drop=True)
    y = df["blue_win"].values

    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.20,
                                      random_state=SEED, stratify=y)
    df_tr, df_te = df.iloc[idx_tr].reset_index(drop=True), df.iloc[idx_te].reset_index(drop=True)
    ytr, yte = y[idx_tr], y[idx_te]
    fit_i, val_i = train_test_split(np.arange(len(df_tr)), test_size=0.10,
                                    random_state=SEED, stratify=ytr)
    df_fit, df_val = df_tr.iloc[fit_i].reset_index(drop=True), df_tr.iloc[val_i].reset_index(drop=True)
    yfit, yval = ytr[fit_i], ytr[val_i]

    vocabs = F.fit_vocabs(df_tr)
    champ2idx = P.build_champ_index(df_tr)
    vsize = len(champ2idx) + 1
    maj_acc = max(y.mean(), 1 - y.mean())

    buf = []   # combined summary lines

    def flush_summary():
        with open(os.path.join(MODELS_DIR, "ckpt4_summary.txt"), "w") as fh:
            fh.write("\n".join(buf) + "\n")

    def w(line=""):
        print(line, flush=True)
        buf.append(line)

    w("=" * 100)
    w("CHECKPOINT 4 — ablations / significance / transfer / patch / hyper-params / cases")
    w("=" * 100)
    w(f"labeled matches={len(y)}  train={len(idx_tr)} (fit={len(fit_i)} val={len(val_i)})  test={len(idx_te)}")
    w(f"floor (predict red)={maj_acc:.4f}   seed={SEED}   bootstrap={N_BOOT}x")
    w(f"champ-embed vocab={vsize - 1}(+pad)  emb_dim={P.EMB_DIM}  hidden={P.HIDDEN}")
    w("")
    flush_summary()

    # store predictions: R[(regime, key)] = {"proba","nfeat", **metrics}
    R = {}

    # =======================================================================
    # SECTION 1 — ablation grid (every model x regime x region)
    # =======================================================================
    w("#" * 100)
    w("# SECTION 1 — ABLATION GRID  (model x regime x region; feature-matched)")
    w("#" * 100)

    for regime in ("A", "B"):
        t0 = time.time()
        rlab = "draft-only" if regime == "A" else "draft+objectives"
        w(f"\n[regime {regime} = {rlab}] building features ...")

        # one-hot/multi-hot matrices, no-region and +region
        mats = {}
        for tag, addr in (("nr", False), ("r", True)):
            mats[tag] = {
                "tr": F.transform(df_tr, vocabs, regime, add_region=addr)[0],
                "fit": F.transform(df_fit, vocabs, regime, add_region=addr)[0],
                "val": F.transform(df_val, vocabs, regime, add_region=addr)[0],
                "te": F.transform(df_te, vocabs, regime, add_region=addr)[0],
            }

        # linear baselines (full train fit), both region settings
        for tag in ("nr", "r"):
            Xtr, Xte = mats[tag]["tr"], mats[tag]["te"]
            for nm, clf in (("LogReg", LogisticRegression(C=1.0, solver="liblinear",
                                                          max_iter=1000, random_state=SEED)),
                            ("NB", BernoulliNB())):
                clf.fit(Xtr, ytr)
                pr = clf.predict_proba(Xte)[:, 1]
                R[(regime, f"{nm}|{tag}")] = {"proba": pr, "nfeat": Xtr.shape[1], **m_all(yte, pr)}

        # LightGBM (fit/val early stop), both region settings
        for tag in ("nr", "r"):
            gbm = fit_gbm(mats[tag]["fit"], yfit, mats[tag]["val"], yval)
            pr = gbm.predict_proba(mats[tag]["te"])[:, 1]
            R[(regime, f"LGBM|{tag}")] = {"proba": pr, "nfeat": mats[tag]["te"].shape[1],
                                          **m_all(yte, pr)}

        # ChampPoolNet (pooled embeddings) + OneHotMLP (multi-hot), both region settings
        for tag, addr in (("nr", False), ("r", True)):
            side_fit = side_info(df_fit, vocabs, regime, addr)
            side_val = side_info(df_val, vocabs, regime, addr)
            side_te = side_info(df_te, vocabs, regime, addr)
            bp_f, rp_f, bb_f, rb_f = P.encode_indices(df_fit, champ2idx)
            bp_v, rp_v, bb_v, rb_v = P.encode_indices(df_val, champ2idx)
            bp_t, rp_t, bb_t, rb_t = P.encode_indices(df_te, champ2idx)
            pr_emb = P.train_net(
                (bp_f, rp_f, bb_f, rb_f, side_fit, yfit),
                (bp_v, rp_v, bb_v, rb_v, side_val, yval),
                (bp_t, rp_t, bb_t, rb_t, side_te),
                vsize, side_fit.shape[1], f"{regime}/{tag}")
            R[(regime, f"Emb|{tag}")] = {"proba": pr_emb,
                                         "nfeat": 4 * P.EMB_DIM + side_fit.shape[1],
                                         **m_all(yte, pr_emb)}

            pr_oh = train_onehot_mlp(mats[tag]["fit"], yfit, mats[tag]["val"], yval,
                                     mats[tag]["te"], regime, tag)
            R[(regime, f"OneHot|{tag}")] = {"proba": pr_oh, "nfeat": mats[tag]["te"].shape[1],
                                            **m_all(yte, pr_oh)}
        w(f"[regime {regime}] grid done in {time.time() - t0:.1f}s")

    # ---- print grid table ----
    w("")
    w(f"{'regime':<8}{'model':<16}{'region':<8}{'nfeat':<8}{'acc':<9}{'F1_blue':<10}{'F1_macro':<10}{'AUC':<8}")
    w("-" * 80)
    NICE = {"LogReg": "LogReg", "NB": "BernoulliNB", "LGBM": "LightGBM",
            "Emb": "ChampPoolNet", "OneHot": "OneHotMLP"}
    for regime in ("A", "B"):
        for key in ("LogReg", "NB", "LGBM", "Emb", "OneHot"):
            for tag in ("nr", "r"):
                r = R[(regime, f"{key}|{tag}")]
                reg_s = "—" if tag == "nr" else "+region"
                w(f"{regime:<8}{NICE[key]:<16}{reg_s:<8}{r['nfeat']:<8}"
                  f"{r['acc']:.4f}   {r['f1_blue']:.4f}    {r['f1_macro']:.4f}    {r['auc']:.4f}")
        w("-" * 80)
    w("Region effect = compare the two rows (— vs +region) of the SAME model. Regime A judged on AUC.")
    with open(os.path.join(MODELS_DIR, "ckpt4_grid.json"), "w") as fh:
        json.dump({f"{rg}|{k}": {kk: vv for kk, vv in v.items() if kk != "proba"}
                   for (rg, k), v in R.items()}, fh, indent=2)
    flush_summary()

    # =======================================================================
    # SECTION 2 — significance (bootstrap CIs + paired diffs)
    # =======================================================================
    w("\n" + "#" * 100)
    w("# SECTION 2 — SIGNIFICANCE  (1000x bootstrap 95% CI; paired AUC differences)")
    w("#" * 100)
    w(f"\n{'regime':<8}{'model':<16}{'region':<9}{'AUC':<9}{'95% CI (bootstrap)':<22}")
    w("-" * 64)
    for regime in ("A", "B"):
        for key in ("LogReg", "NB", "LGBM", "Emb", "OneHot"):
            for tag in ("nr", "r"):
                pr = R[(regime, f"{key}|{tag}")]["proba"]
                a, lo, hi = boot_auc_ci(yte, pr)
                reg_s = "—" if tag == "nr" else "+region"
                w(f"{regime:<8}{NICE[key]:<16}{reg_s:<9}{a:.4f}   [{lo:.4f}, {hi:.4f}]")
        w("-" * 64)

    # ---- headline paired contrasts ----
    def diff_line(regime, ka, kb, label):
        a = R[(regime, ka)]["proba"]; b = R[(regime, kb)]["proba"]
        d, lo, hi, p = boot_auc_diff(yte, a, b)
        sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "within noise"
        w(f"  {label:<52} dAUC={d:+.4f}  95%CI[{lo:+.4f},{hi:+.4f}]  p={p:.3f}  -> {sig}")
        return {"regime": regime, "label": label, "dauc": d, "ci": [lo, hi], "p": p, "sig": sig}

    sig_rows = []
    w("\n--- REGIME A : region-vs-nonlinearity decomposition (judged on AUC) ---")
    w("  [region alone] same model, +region minus —region:")
    sig_rows.append(diff_line("A", "LGBM|r", "LGBM|nr", "LightGBM: +region − —region"))
    sig_rows.append(diff_line("A", "LogReg|r", "LogReg|nr", "LogReg:  +region − —region"))
    w("  [nonlinear − linear, FEATURE-MATCHED] same region setting:")
    sig_rows.append(diff_line("A", "LGBM|nr", "LogReg|nr", "LightGBM − LogReg  (both —region)"))
    sig_rows.append(diff_line("A", "LGBM|r", "LogReg|r", "LightGBM − LogReg  (both +region)"))
    sig_rows.append(diff_line("A", "Emb|r", "LogReg|r", "ChampPoolNet − LogReg (both +region)"))

    w("\n--- REGIME B : between-model contrasts (all +region, feature-matched) ---")
    sig_rows.append(diff_line("B", "LGBM|r", "LogReg|r", "LightGBM − LogReg"))
    sig_rows.append(diff_line("B", "LGBM|r", "NB|r", "LightGBM − BernoulliNB"))
    sig_rows.append(diff_line("B", "Emb|r", "LogReg|r", "ChampPoolNet − LogReg"))
    sig_rows.append(diff_line("B", "LGBM|r", "Emb|r", "LightGBM − ChampPoolNet"))
    with open(os.path.join(MODELS_DIR, "ckpt4_significance.json"), "w") as fh:
        json.dump(sig_rows, fh, indent=2)
    flush_summary()

    # =======================================================================
    # SECTION 3 — embedding vs one-hot (matched), pulled out explicitly
    # =======================================================================
    w("\n" + "#" * 100)
    w("# SECTION 3 — EMBEDDING vs ONE-HOT  (same MLP head + same side features; champions swapped)")
    w("#" * 100)
    w(f"\n{'regime':<8}{'champion repr':<16}{'region':<9}{'nfeat':<8}{'acc':<9}{'AUC':<9}")
    w("-" * 60)
    for regime in ("A", "B"):
        for tag in ("nr", "r"):
            for key, lab in (("Emb", "pooled-embed"), ("OneHot", "multi-hot")):
                r = R[(regime, f"{key}|{tag}")]
                reg_s = "—" if tag == "nr" else "+region"
                w(f"{regime:<8}{lab:<16}{reg_s:<9}{r['nfeat']:<8}{r['acc']:.4f}   {r['auc']:.4f}")
        w("-" * 60)
    w("Paired AUC difference (embedding − one-hot), +region:")
    diff_line("A", "Emb|r", "OneHot|r", "regime A: ChampPoolNet − OneHotMLP")
    diff_line("B", "Emb|r", "OneHot|r", "regime B: ChampPoolNet − OneHotMLP")
    flush_summary()

    # =======================================================================
    # SECTION 4 — leakage CONTROL (labeled; NOT a real result)
    # =======================================================================
    w("\n" + "#" * 100)
    w("# SECTION 4 — LEAKAGE CONTROL  (regime-B features + END-OF-GAME leaky cols)")
    w("# *** NOT A REAL RESULT — illustrates how post-hoc state inflates the score ***")
    w("#" * 100)

    def leaky_block(d):
        cols = []
        for c in LEAKY_KILL_COLS:
            cols.append(d[c].astype(float).to_numpy().reshape(-1, 1))
        for c in ("firstInhibitor", "firstBaron"):
            ov = d[c].astype(float).to_numpy()
            cols.append((ov == 1.0).astype(np.float32).reshape(-1, 1))   # blue
            cols.append((ov == 2.0).astype(np.float32).reshape(-1, 1))   # red
        cols.append(d["gameDuration"].astype(float).to_numpy().reshape(-1, 1))
        return sp.csr_matrix(np.hstack(cols).astype(np.float32))

    Xfit_b = F.transform(df_fit, vocabs, "B", add_region=True)[0]
    Xval_b = F.transform(df_val, vocabs, "B", add_region=True)[0]
    Xte_b = F.transform(df_te, vocabs, "B", add_region=True)[0]
    Lfit = sp.hstack([Xfit_b, leaky_block(df_fit)]).tocsr()
    Lval = sp.hstack([Xval_b, leaky_block(df_val)]).tocsr()
    Lte = sp.hstack([Xte_b, leaky_block(df_te)]).tocsr()
    gbm_leak = fit_gbm(Lfit, yfit, Lval, yval)
    pr_leak = gbm_leak.predict_proba(Lte)[:, 1]
    leak_m = m_all(yte, pr_leak)
    honest = R[("B", "LGBM|r")]
    w(f"\n  honest  regime-B LightGBM (+region)         : acc={honest['acc']:.4f}  AUC={honest['auc']:.4f}")
    w(f"  LEAKY   regime-B + kills/inhib/baron/duration: acc={leak_m['acc']:.4f}  AUC={leak_m['auc']:.4f}")
    w(f"  -> leakage moves AUC by {leak_m['auc'] - honest['auc']:+.4f} "
      f"and accuracy by {leak_m['acc'] - honest['acc']:+.4f}  (why those columns are banned).")
    flush_summary()

    # =======================================================================
    # SECTION 5 — cross-region 3x3 transfer (regime-A LightGBM, NO region)
    # =======================================================================
    w("\n" + "#" * 100)
    w("# SECTION 5 — CROSS-REGION 3x3 TRANSFER  (strongest draft-only: regime-A LightGBM, NO region)")
    w("#" * 100)

    def region_transfer(regime):
        """Train per region (own train-subset vocab), test on all regions. Returns matrix dict."""
        out = {}
        for rtr in REGIONS:
            d_tr_r = df_tr[df_tr["region"] == rtr].reset_index(drop=True)
            y_tr_r = d_tr_r["blue_win"].values
            fi, vi = train_test_split(np.arange(len(d_tr_r)), test_size=0.10,
                                      random_state=SEED, stratify=y_tr_r)
            d_fit_r, d_val_r = d_tr_r.iloc[fi].reset_index(drop=True), d_tr_r.iloc[vi].reset_index(drop=True)
            voc_r = F.fit_vocabs(d_tr_r)   # train-subset vocab only
            Xf = F.transform(d_fit_r, voc_r, regime, add_region=False)[0]
            Xv = F.transform(d_val_r, voc_r, regime, add_region=False)[0]
            gbm = fit_gbm(Xf, y_tr_r[fi], Xv, y_tr_r[vi])
            for rte in REGIONS:
                d_te_r = df_te[df_te["region"] == rte].reset_index(drop=True)
                y_te_r = d_te_r["blue_win"].values
                Xt = F.transform(d_te_r, voc_r, regime, add_region=False)[0]
                pr = gbm.predict_proba(Xt)[:, 1]
                out[(rtr, rte)] = {"n": len(y_te_r), **m_all(y_te_r, pr)}
        return out

    for regime, note in (("A", "draft-only — the headline transfer test"),
                         ("B", "draft+objectives — control")):
        tm = region_transfer(regime)
        w(f"\n  regime {regime} ({note})   rows: " +
          "  ".join(f"{r}={int((df_te['region'] == r).sum())}" for r in REGIONS))
        w(f"    {'train\\test':<12}" + "".join(f"{r:<22}" for r in REGIONS))
        for rtr in REGIONS:
            cells = []
            for rte in REGIONS:
                c = tm[(rtr, rte)]
                cells.append(f"acc{c['acc']:.3f}/AUC{c['auc']:.3f}")
            w(f"    {rtr:<12}" + "".join(f"{c:<22}" for c in cells))
        diag = np.mean([tm[(r, r)]["auc"] for r in REGIONS])
        offd = np.mean([tm[(a, b)]["auc"] for a in REGIONS for b in REGIONS if a != b])
        w(f"    mean diagonal (same-region) AUC={diag:.4f}   mean off-diagonal (transfer) AUC={offd:.4f}"
          f"   gap={diag - offd:+.4f}")
    flush_summary()

    # =======================================================================
    # SECTION 6 — patch robustness
    # =======================================================================
    w("\n" + "#" * 100)
    w("# SECTION 6 — PATCH ROBUSTNESS  (pooled regime-B LightGBM; data is mostly 16.10)")
    w("#" * 100)
    pr_b = R[("B", "LGBM|r")]["proba"]
    patches = sorted(df_te["patch"].unique(), key=lambda s: [int(x) for x in s.split(".")])
    w(f"\n  {'patch':<10}{'n_test':<9}{'blue_rate':<11}{'acc':<9}{'F1_blue':<10}{'AUC':<9}")
    w("  " + "-" * 56)
    for pt in patches:
        mask = (df_te["patch"] == pt).to_numpy()
        if mask.sum() == 0:
            continue
        ysub, psub = yte[mask], pr_b[mask]
        pred = (psub >= 0.5).astype(int)
        auc = roc_auc_score(ysub, psub) if len(np.unique(ysub)) > 1 else float("nan")
        w(f"  {pt:<10}{int(mask.sum()):<9}{ysub.mean():<11.4f}"
          f"{accuracy_score(ysub, pred):<9.4f}{f1_score(ysub, pred, pos_label=1, zero_division=0):<10.4f}{auc:<9.4f}")
    w("  " + "-" * 56)

    # add patch one-hot to the pooled regime-B model
    patch_levels = sorted(df_tr["patch"].unique())
    p2i = {p: i for i, p in enumerate(patch_levels)}

    def patch_oh(d):
        m = np.zeros((len(d), len(patch_levels)), dtype=np.float32)
        pv = d["patch"].to_numpy()
        for i, p in enumerate(pv):
            if p in p2i:
                m[i, p2i[p]] = 1.0
        return sp.csr_matrix(m)

    Pfit = sp.hstack([Xfit_b, patch_oh(df_fit)]).tocsr()
    Pval = sp.hstack([Xval_b, patch_oh(df_val)]).tocsr()
    Pte = sp.hstack([Xte_b, patch_oh(df_te)]).tocsr()
    gbm_p = fit_gbm(Pfit, yfit, Pval, yval)
    pr_p = gbm_p.predict_proba(Pte)[:, 1]
    a_no, _, _ = boot_auc_ci(yte, pr_b)
    dd, lo, hi, p = boot_auc_diff(yte, pr_p, pr_b)
    w(f"\n  pooled regime-B LightGBM  AUC (no patch feature) = {a_no:.4f}")
    w(f"  + patch one-hot ({'/'.join(patch_levels)})        AUC = {roc_auc_score(yte, pr_p):.4f}")
    w(f"  -> dAUC={dd:+.4f}  95%CI[{lo:+.4f},{hi:+.4f}]  p={p:.3f}  "
      f"({'helps' if lo > 0 else 'no significant change'})")
    flush_summary()

    # =======================================================================
    # SECTION 7 — hyper-parameter sensitivity (regime-B LightGBM)
    # =======================================================================
    w("\n" + "#" * 100)
    w("# SECTION 7 — HYPER-PARAMETER SENSITIVITY  (regime-B LightGBM, +region, test AUC)")
    w("#" * 100)

    def sweep(name, values, fixed_kw, use_es=True):
        w(f"\n  sweep {name} (others at default):")
        w(f"    {name:<14}{'AUC':<9}{'best_iter':<10}")
        for v in values:
            kw = dict(fixed_kw); kw[name] = v
            gbm = fit_gbm(Xfit_b, yfit, Xval_b, yval, early_stop=use_es, **kw)
            pr = gbm.predict_proba(Xte_b)[:, 1]
            bi = getattr(gbm, "best_iteration_", None) or kw.get("n_estimators", "-")
            w(f"    {str(v):<14}{roc_auc_score(yte, pr):<9.4f}{str(bi):<10}")

    sweep("learning_rate", [0.01, 0.03, 0.1, 0.3], {})
    sweep("num_leaves", [15, 31, 63, 127], {})
    sweep("n_estimators", [100, 300, 1000, 3000], {"learning_rate": 0.03}, use_es=False)
    flush_summary()

    # =======================================================================
    # SECTION 8 — case studies (regime-B LightGBM)
    # =======================================================================
    w("\n" + "#" * 100)
    w("# SECTION 8 — CASE STUDIES  (regime-B LightGBM, +region)")
    w("#" * 100)
    proba = R[("B", "LGBM|r")]["proba"]
    pred = (proba >= 0.5).astype(int)

    def decode_obj(row):
        who = {0: "none", 1: "BLUE", 2: "RED"}
        parts = [f"{o.replace('first', '')}->{who.get(int(row[o]), '?')}" for o in OBJECTIVE_COLS]
        nb = sum(int(row[o]) == 1 for o in OBJECTIVE_COLS)
        nr = sum(int(row[o]) == 2 for o in OBJECTIVE_COLS)
        return ", ".join(parts), nb, nr

    def show(i, why):
        row = df_te.iloc[i]
        objs, nb, nr = decode_obj(row)
        w(f"\n  [{why}]")
        w(f"    matchId={row['matchId']}  region={row['region']}  patch={row['patch']}")
        w(f"    objectives: {objs}   (blue got {nb}/4, red got {nr}/4)")
        w(f"    model P(blue win)={proba[i]:.3f} -> predict {'BLUE' if pred[i] else 'RED'}    "
          f"actual={'BLUE' if yte[i] == 1 else 'RED'}  ({'correct' if pred[i] == yte[i] else 'WRONG'})")

    correct = np.where(pred == yte)[0]
    wrong = np.where(pred != yte)[0]
    conf = np.abs(proba - 0.5)
    # 1) most confident correct
    show(correct[np.argmax(conf[correct])], "most confident CORRECT")
    # 2) most confident wrong (the biggest upset)
    show(wrong[np.argmax(conf[wrong])], "most confident WRONG — upset (early objectives lost the game anyway)")
    # 3) a confident-wrong KR match if any, else a split-objective match
    obj_split = np.array([abs(decode_obj(df_te.iloc[i])[1] - decode_obj(df_te.iloc[i])[2]) for i in range(len(df_te))])
    kr_wrong = [i for i in wrong if df_te.iloc[i]["region"] == "kr"]
    if kr_wrong:
        show(kr_wrong[int(np.argmax(conf[kr_wrong]))], "KR cross-region upset (confident-wrong in Korea)")
    split_idx = np.where(obj_split == 0)[0]
    if len(split_idx):
        show(split_idx[int(np.argmax(conf[split_idx]))],
             "split objectives 2-2 — model still leans on which objectives + side prior")
    flush_summary()

    w("\n" + "=" * 100)
    w(f"CHECKPOINT 4 complete in {time.time() - t_start:.1f}s. "
      f"Wrote models/ckpt4_summary.txt, ckpt4_grid.json, ckpt4_significance.json")
    w("=" * 100)
    flush_summary()


if __name__ == "__main__":
    main()
