"""
Checkpoint 5 (training half) — fit and PERSIST the demo models.

The served headline model is the pooled **regime-B LightGBM (+region)** from
checkpoints 3/4. But we ALSO persist a **regime-A (draft-only, +region) model**,
used only for the pre-game baseline, because the "all objectives = none" state is
genuinely OUT-OF-DISTRIBUTION for the regime-B model: in the training data all
four first-objectives are simultaneously "none" in 0.000% of matches (first blood
is decided in 99.98% of games, first tower in 99.3%). Feeding regime-B all-zeros
is extrapolation and — as the self-test caught — makes it spuriously sensitive to
champion swaps, which would contradict the paper's finding that draft is flat.

So the demo does the honest thing:
  * all objectives NONE  -> show the regime-A draft-only model (genuinely flat,
                            AUC~0.54, near the region base rate)
  * any objective set    -> show the regime-B model (the real signal, AUC~0.79)

Both models: regime features + region one-hot, NO summoner spells (the UI does not
collect them; ckpt 4 showed spells carry no measurable signal). Vocab fit on TRAIN
ONLY, same stratified split (seed 42) as every other checkpoint.

Artifacts -> models/demo/ :
    lgbm_A.joblib / lgbm_B.joblib            fitted LGBMClassifiers
    explainer_A.joblib / explainer_B.joblib  shap.TreeExplainer per model
    vocabs.json                              champ/ban/spell vocab (champ+ban used)
    meta.json                                per-regime feature_names + metrics,
                                             champ/ban lists, region base rates

Run:  PYTHONPATH=src python3 src/train_demo_model.py
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import json
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import lightgbm as lgb
import shap

from data_utils import load_matches, SEED, REGIONS, OBJECTIVE_COLS
import features as F

DEMO_DIR = os.path.join("models", "demo")


def fit_regime(regime, df_fit, yfit, df_val, yval, df_te, yte, vocabs):
    kw = dict(regime=regime, add_region=True, include_spells=False)
    names = F.feature_names(vocabs, regime, add_region=True, include_spells=False)
    Xfit = F.transform(df_fit, vocabs, **kw)[0]
    Xval = F.transform(df_val, vocabs, **kw)[0]
    Xte = F.transform(df_te, vocabs, **kw)[0]
    gbm = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.03, num_leaves=31,
                             min_child_samples=50, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.6, reg_lambda=2.0, random_state=SEED,
                             n_jobs=1, verbose=-1)
    gbm.fit(Xfit, yfit, eval_set=[(Xval, yval)], eval_metric="auc",
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    proba = gbm.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {"test_auc": float(roc_auc_score(yte, proba)),
               "test_acc": float(accuracy_score(yte, pred)),
               "test_f1_blue": float(f1_score(yte, pred, pos_label=1, zero_division=0)),
               "best_iteration": int(gbm.best_iteration_ or 0)}
    explainer = shap.TreeExplainer(gbm)
    return gbm, explainer, names, metrics


def main():
    os.makedirs(DEMO_DIR, exist_ok=True)
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
    region_base = {r: float(df.loc[df["region"] == r, "blue_win"].mean()) for r in REGIONS}
    meta = {
        "served_model": "B", "baseline_model": "A",
        "add_region": True, "include_spells": False,
        "champ_list": vocabs["champ"], "ban_list": vocabs["ban"],
        "regions": REGIONS, "objective_cols": OBJECTIVE_COLS,
        "region_base_rate": region_base, "overall_base_rate": float(y.mean()),
        "seed": SEED, "regimes": {},
    }

    print("=" * 78)
    print("CHECKPOINT 5 — demo models (regime A draft-only baseline + regime B served)")
    print("=" * 78)
    for regime in ("A", "B"):
        gbm, explainer, names, metrics = fit_regime(regime, df_fit, yfit, df_val, yval,
                                                    df_te, yte, vocabs)
        joblib.dump(gbm, os.path.join(DEMO_DIR, f"lgbm_{regime}.joblib"))
        try:
            joblib.dump(explainer, os.path.join(DEMO_DIR, f"explainer_{regime}.joblib"))
            expl = "saved"
        except Exception as e:                      # pragma: no cover
            expl = f"NOT saved ({e}); demo rebuilds / native fallback"
        ev = explainer.expected_value
        ev = float(np.atleast_1d(ev)[-1])
        meta["regimes"][regime] = {"feature_names": names, "n_features": len(names),
                                   "shap_expected_value": ev, **metrics}
        tag = "draft-only baseline" if regime == "A" else "SERVED headline"
        print(f"[regime {regime}] {tag:20}  feats={len(names):4}  "
              f"AUC={metrics['test_auc']:.4f}  acc={metrics['test_acc']:.4f}  "
              f"best_iter={metrics['best_iteration']}  explainer={expl}")

    with open(os.path.join(DEMO_DIR, "vocabs.json"), "w") as fh:
        json.dump(vocabs, fh)
    with open(os.path.join(DEMO_DIR, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    # remove any stale single-model artifacts from an earlier run
    for stale in ("lgbm.joblib", "explainer.joblib"):
        p = os.path.join(DEMO_DIR, stale)
        if os.path.exists(p):
            os.remove(p)

    print(f"region base rates (blue-win): { {r: round(v, 3) for r, v in region_base.items()} }")
    print(f"[written] {DEMO_DIR}/lgbm_A.joblib lgbm_B.joblib explainer_A/B.joblib vocabs.json meta.json")


if __name__ == "__main__":
    main()
