"""
Checkpoint 2 — preprocessing + feature regimes A/B + the two baselines.

Pooled (all three regions together). For each regime we train:
  * Logistic Regression  (L2, liblinear) on the team-symmetric multi-hot features
  * Bernoulli Naive Bayes on the same binary features

and report Accuracy / F1(blue) / F1(macro) / ROC-AUC on a fixed stratified
hold-out, against the majority-class floor (always predict red).

Run:  PYTHONPATH=src python3 src/baselines.py
"""
from __future__ import annotations
import json
import os

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from data_utils import load_matches, TARGET, SEED
import features as F

MODELS_DIR = "models"
REGIMES = {"A": "draft-only", "B": "draft + early objectives"}


def clean(df):
    """Drop rows whose outcome is not a real win label (winner in {1,2})."""
    before = len(df)
    df = df[df[TARGET].isin([1, 2])].copy()
    dropped = before - len(df)
    return df, dropped


def evaluate(name, clf, Xtr, ytr, Xte, yte):
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "model": name,
        "acc": accuracy_score(yte, pred),
        "f1_blue": f1_score(yte, pred, pos_label=1),
        "f1_macro": f1_score(yte, pred, average="macro"),
        "auc": roc_auc_score(yte, proba),
    }


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = load_matches()
    df, dropped = clean(df)

    y = df["blue_win"].values  # 1 = blue/team1 won, 0 = red/team2 won
    idx_tr, idx_te = train_test_split(
        np.arange(len(df)), test_size=0.20, random_state=SEED, stratify=y
    )
    df_tr, df_te = df.iloc[idx_tr], df.iloc[idx_te]
    ytr, yte = y[idx_tr], y[idx_te]

    # vocab is FIT ON TRAIN ONLY (saved for downstream checkpoints / the demo)
    vocabs = F.fit_vocabs(df_tr)
    with open(os.path.join(MODELS_DIR, "vocabs.json"), "w") as fh:
        json.dump(vocabs, fh)

    # ---- headline context ----
    blue_rate = y.mean()
    maj_acc = max(blue_rate, 1 - blue_rate)            # always-predict-red
    maj_pred_te = np.zeros_like(yte)                   # red == blue_win 0
    print("=" * 78)
    print("CHECKPOINT 2 — preprocessing + regimes A/B + baselines (pooled, all regions)")
    print("=" * 78)
    print(f"rows loaded (raw)        : {len(y) + dropped}")
    print(f"dropped (winner not 1/2) : {dropped}")
    print(f"labeled matches          : {len(y)}")
    print(f"  blue wins              : {int(y.sum())}  ({blue_rate:.4f})")
    print(f"  red  wins              : {int((1 - y).sum())}  ({1 - blue_rate:.4f})")
    print(f"split (stratified, seed={SEED}) : train={len(idx_tr)}  test={len(idx_te)}")
    print(f"vocab sizes              : champs={len(vocabs['champ'])}  "
          f"bans={len(vocabs['ban'])}  spells={len(vocabs['spell'])}")
    print()
    print(f"MAJORITY FLOOR (predict red): acc={maj_acc:.4f}  "
          f"f1_blue={f1_score(yte, maj_pred_te, pos_label=1, zero_division=0):.4f}  "
          f"f1_macro={f1_score(yte, maj_pred_te, average='macro', zero_division=0):.4f}  auc=0.5000")
    print()

    rows = []
    for regime, label in REGIMES.items():
        Xtr, names = F.transform(df_tr, vocabs, regime)
        Xte, _ = F.transform(df_te, vocabs, regime)
        for name, clf in [
            ("LogisticRegression", LogisticRegression(C=1.0, solver="liblinear",
                                                       max_iter=1000, random_state=SEED)),
            ("BernoulliNB", BernoulliNB()),
        ]:
            r = evaluate(name, clf, Xtr, ytr, Xte, yte)
            r["regime"] = f"{regime} ({label})"
            r["nfeat"] = Xtr.shape[1]
            rows.append(r)

    # ---- results table ----
    print(f"{'regime':<26}{'model':<22}{'nfeat':<8}{'acc':<9}{'F1_blue':<10}{'F1_macro':<10}{'AUC':<8}")
    print("-" * 93)
    last = None
    for r in rows:
        reg = r["regime"] if r["regime"] != last else ""
        last = r["regime"]
        d_acc = r["acc"] - maj_acc
        print(f"{reg:<26}{r['model']:<22}{r['nfeat']:<8}"
              f"{r['acc']:.4f}   {r['f1_blue']:.4f}    {r['f1_macro']:.4f}    {r['auc']:.4f}"
              f"   ({d_acc:+.4f} vs floor)")
    print("-" * 93)
    print("Notes: F1_blue = F1 with blue-win as positive class (the minority side).")
    print("       'vs floor' = test accuracy minus the 54.2% always-predict-red baseline.")


if __name__ == "__main__":
    main()
