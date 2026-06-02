"""
Checkpoint 5 (serving half) — artifact loading + single-row encoding + predict +
SHAP attribution. NO streamlit dependency, so the self-test and the Streamlit app
share exactly one code path (the encoder that turns UI state into a feature row is
the SAME features.transform used at training time, with the saved train-only vocab).

Two persisted models:
  * regime A (draft-only, +region)  -> the PRE-GAME baseline (genuinely flat).
  * regime B (draft+objectives, +region) -> the SERVED headline model.
The demo shows regime A while all objectives are "none" (the all-none state is
0.000%-frequency / out-of-distribution for regime B), and switches to regime B the
moment any objective is set. See train_demo_model.py for the rationale.
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import json
import warnings
import numpy as np
import pandas as pd
import joblib

from data_utils import OBJECTIVE_COLS, SLOTS
import features as F

warnings.filterwarnings("ignore", message=".*valid feature names.*")
warnings.filterwarnings("ignore", message=".*output has changed to a list.*")

DEMO_DIR = os.path.join("models", "demo")
EMPTY_PICK = ""          # maps to no column (out-of-vocab) -> contributes nothing
NO_BAN = "None"          # in NO_BAN_TOKENS -> dropped by the ban encoder
OBJ_CN = {"firstBlood": "一血 First Blood", "firstTower": "一塔 First Tower",
          "firstDragon": "首龙 First Dragon", "firstRiftHerald": "首先锋 First Herald"}

_ART = None


def _load_explainer(demo_dir, regime, model):
    try:
        return joblib.load(os.path.join(demo_dir, f"explainer_{regime}.joblib"))
    except Exception:
        try:
            import shap
            return shap.TreeExplainer(model)
        except Exception:
            return None                                 # -> native TreeSHAP fallback


def load_artifacts(demo_dir: str = DEMO_DIR):
    """Load (and cache) the persisted models / vocab / meta / SHAP explainers."""
    global _ART
    if _ART is not None:
        return _ART
    with open(os.path.join(demo_dir, "meta.json")) as fh:
        meta = json.load(fh)
    with open(os.path.join(demo_dir, "vocabs.json")) as fh:
        vocabs = json.load(fh)
    models, explainers = {}, {}
    for regime in ("A", "B"):
        models[regime] = joblib.load(os.path.join(demo_dir, f"lgbm_{regime}.joblib"))
        explainers[regime] = _load_explainer(demo_dir, regime, models[regime])
    _ART = {"meta": meta, "vocabs": vocabs, "models": models, "explainers": explainers}
    return _ART


def default_state(art) -> dict:
    """A valid starting draft: 10 DISTINCT champions, no bans, all objectives none."""
    champs = art["meta"]["champ_list"]
    regions = art["meta"]["regions"]
    return {
        "region": "na1" if "na1" in regions else regions[0],
        "blue": list(champs[0:5]),
        "red": list(champs[5:10]),
        "blue_bans": [None] * 5,
        "red_bans": [None] * 5,
        "obj": {o: 0 for o in OBJECTIVE_COLS},
    }


def any_objective(state) -> bool:
    return any(int(v) > 0 for v in state["obj"].values())


def duplicate_picks(state) -> list:
    """Champions selected more than once across both teams (validation)."""
    picks = [c for c in list(state["blue"]) + list(state["red"]) if c and c != EMPTY_PICK]
    seen, dup = set(), set()
    for c in picks:
        if c in seen:
            dup.add(c)
        seen.add(c)
    return sorted(dup)


def _row_df(state) -> pd.DataFrame:
    row = {}
    for i, c in zip(SLOTS, state["blue"]):
        row[f"t1_champ{i}"] = c or EMPTY_PICK
    for i, c in zip(SLOTS, state["red"]):
        row[f"t2_champ{i}"] = c or EMPTY_PICK
    for i, c in zip(SLOTS, state.get("blue_bans") or [None] * 5):
        row[f"t1_ban{i}"] = c or NO_BAN
    for i, c in zip(SLOTS, state.get("red_bans") or [None] * 5):
        row[f"t2_ban{i}"] = c or NO_BAN
    for o in OBJECTIVE_COLS:
        row[o] = int(state["obj"].get(o, 0))
    row["region"] = state["region"]
    return pd.DataFrame([row])


def encode(state, art, regime):
    """UI state -> (1, n_features) feature row in `regime`'s exact column order."""
    X, _ = F.transform(_row_df(state), art["vocabs"], regime,
                       add_region=art["meta"]["add_region"],
                       include_spells=art["meta"].get("include_spells", False))
    return X


def predict_prob(state, art, regime) -> float:
    return float(art["models"][regime].predict_proba(encode(state, art, regime))[:, 1][0])


def _shap_row(X_dense, explainer, model):
    """Per-feature SHAP (log-odds toward blue win). Robust to list / 3-D returns;
    falls back to LightGBM native TreeSHAP (pred_contrib)."""
    if explainer is not None:
        sv = explainer.shap_values(X_dense)
        if isinstance(sv, list):                        # [neg_class, pos_class]
            sv = sv[-1]
        sv = np.asarray(sv)
        if sv.ndim == 3:                                # (n, feat, classes)
            sv = sv[..., -1]
        return sv.reshape(-1)
    contrib = model.booster_.predict(X_dense, pred_contrib=True)[0]
    return np.asarray(contrib[:-1])                     # drop trailing base value


def predict_full(state, art, topk: int = 6):
    """Pick the honest model (A if all-none, else B) and return:
        dict(prob, prob_draft, regime, contribs[(name, shap)], delta)
    where prob_draft is the regime-A pre-game baseline and delta = prob - prob_draft.
    """
    prob_draft = predict_prob(state, art, "A")           # draft-only baseline (flat)
    use_B = any_objective(state)
    regime = "B" if use_B else "A"
    X = encode(state, art, regime).toarray()
    prob = float(art["models"][regime].predict_proba(X)[:, 1][0])
    sv = _shap_row(X, art["explainers"][regime], art["models"][regime])
    names = art["meta"]["regimes"][regime]["feature_names"]
    order = np.argsort(-np.abs(sv))
    contribs = []
    for j in order:
        if abs(float(sv[j])) < 1e-6:
            break
        contribs.append((names[j], float(sv[j])))
        if len(contribs) >= topk:
            break
    return {"prob": prob, "prob_draft": prob_draft, "regime": regime,
            "contribs": contribs, "delta": prob - prob_draft}


def friendly(name: str) -> str:
    """Human-readable label for a feature column name (for the SHAP panel)."""
    if name.startswith("region="):
        return f"地区基准 Region = {name.split('=', 1)[1].upper()}"
    for o in OBJECTIVE_COLS:
        if name == f"{o}_blue":
            return f"蓝方拿到 {OBJ_CN[o]}"
        if name == f"{o}_red":
            return f"红方拿到 {OBJ_CN[o]}"
    if name.startswith("blue_pick="):
        return f"蓝方选 {name.split('=', 1)[1]}"
    if name.startswith("red_pick="):
        return f"红方选 {name.split('=', 1)[1]}"
    if name.startswith("blue_ban="):
        return f"蓝方 ban {name.split('=', 1)[1]}"
    if name.startswith("red_ban="):
        return f"红方 ban {name.split('=', 1)[1]}"
    return name
