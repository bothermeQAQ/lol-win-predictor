"""
Checkpoint 5 — FastAPI backend for the LoL Blue-vs-Red win predictor.

This is an HTTP layer ONLY. It does not retrain, does not touch the model files,
and does not re-implement any encoding / prediction / SHAP / two-model-switch
logic — it reuses demo_core.predict_full, the exact same code path the Streamlit
demo and the self-test use. So /predict returns numbers identical to demo_core
(proven by src/api/verify_api.py).

Endpoints:
  GET  /health   -> 200 + "model loaded" status
  GET  /meta     -> regions(+base win rate), champion vocab, ban vocab, objectives
  POST /predict  -> {blue_win_prob, which_model:A|B, top_factors:[...]}

Deployment-friendly:
  * binds to env PORT (default 8000) — platforms like Render inject PORT.
  * CORS allow-origins from env FRONTEND_ORIGIN (comma-separated); localhost on
    any port is always allowed for local dev.
  * model files in models/demo/ are loaded ONCE at import/startup (cached in
    demo_core), by ABSOLUTE path, so the service works from any working directory.

Run locally:
  PYTHONPATH=src uvicorn api.server:app --reload --port 8000
  # or:  PORT=8000 python3 src/api/server.py
"""
from __future__ import annotations
import os
import sys

# make src/ importable however we're launched; resolve model dir absolutely
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../src
_ROOT = os.path.dirname(_SRC)                                        # project root
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import demo_core as C

# --------------------------------------------------------------- load artifacts
DEMO_DIR = os.path.join(_ROOT, "models", "demo")
ART = C.load_artifacts(DEMO_DIR)        # loaded ONCE; demo_core caches it
META = ART["meta"]
CHAMP_SET = set(META["champ_list"])
BAN_SET = set(META["ban_list"])
REGIONS = list(META["regions"])
TEAM_SIZE = len(C.SLOTS)                 # 5

REGION_LABEL = {"na1": "North America (NA)", "kr": "Korea (KR)", "euw1": "Europe West (EUW)"}
# frontend (snake_case)  ->  model field (camelCase used inside demo_core)
OBJ_API_TO_MODEL = {"first_blood": "firstBlood", "first_tower": "firstTower",
                    "first_dragon": "firstDragon", "first_rift_herald": "firstRiftHerald"}
OBJ_LABEL = {"first_blood": "First Blood", "first_tower": "First Tower",
             "first_dragon": "First Dragon", "first_rift_herald": "First Rift Herald"}
SIDE_TO_INT = {"none": 0, "blue": 1, "red": 2}


def _feature_label(name: str) -> str:
    """English label for a raw model feature name (display only)."""
    if name.startswith("region="):
        return f"Regional baseline · {name.split('=', 1)[1].upper()}"
    for api_key, field in OBJ_API_TO_MODEL.items():
        if name == f"{field}_blue":
            return f"Blue took {OBJ_LABEL[api_key]}"
        if name == f"{field}_red":
            return f"Red took {OBJ_LABEL[api_key]}"
    if name.startswith("blue_pick="):
        return f"Blue picks {name.split('=', 1)[1]}"
    if name.startswith("red_pick="):
        return f"Red picks {name.split('=', 1)[1]}"
    if name.startswith("blue_ban="):
        return f"Blue bans {name.split('=', 1)[1]}"
    if name.startswith("red_ban="):
        return f"Red bans {name.split('=', 1)[1]}"
    return name


# ----------------------------------------------------------------- request models
class Objectives(BaseModel):
    first_blood: str = "none"
    first_tower: str = "none"
    first_dragon: str = "none"
    first_rift_herald: str = "none"


class PredictRequest(BaseModel):
    region: str
    blue: List[str]
    red: List[str]
    bans: Optional[List[str]] = None
    objectives: Objectives = Field(default_factory=Objectives)


# ----------------------------------------------------------------- app + CORS
app = FastAPI(title="LoL Blue-vs-Red Win Predictor API", version="1.0",
              description="HTTP wrapper over the DSC148 demo_core two-model predictor.")

_explicit_origins = [o.strip() for o in os.environ.get("FRONTEND_ORIGIN", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_explicit_origins,                                   # e.g. Vercel domain in prod
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",    # any localhost port in dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------- helpers
def _validate_side(names: List[str], side: str) -> List[str]:
    picks = [str(c).strip() for c in (names or []) if c is not None and str(c).strip() != ""]
    if len(picks) != TEAM_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"{side} team must have exactly {TEAM_SIZE} champions (received {len(picks)}).")
    unknown = [c for c in picks if c not in CHAMP_SET]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown champion name(s) on {side} team: {', '.join(unknown)}. "
                   f"Use names from GET /meta -> champions.")
    return picks


def _build_state(req: PredictRequest):
    if req.region not in REGIONS:
        raise HTTPException(status_code=400,
                            detail=f"Unknown region '{req.region}'. Valid regions: {', '.join(REGIONS)}.")
    blue = _validate_side(req.blue, "Blue")
    red = _validate_side(req.red, "Red")
    dups = C.duplicate_picks({"blue": blue, "red": red})
    if dups:
        raise HTTPException(
            status_code=400,
            detail=f"Duplicate champion(s): {', '.join(dups)}. A champion cannot be picked twice "
                   f"(across or within teams).")

    obj = {}
    for api_key, field in OBJ_API_TO_MODEL.items():
        raw = str(getattr(req.objectives, api_key, "none")).strip().lower()
        if raw not in SIDE_TO_INT:
            raise HTTPException(
                status_code=400,
                detail=f"objectives.{api_key} must be one of none|blue|red (received '{raw}').")
        obj[field] = SIDE_TO_INT[raw]

    # bans are optional + low-signal: pass through, unknown/empty are ignored (as in the demo).
    bans = [str(b).strip() for b in (req.bans or []) if b and str(b).strip()]
    blue_bans, red_bans = bans[:TEAM_SIZE], bans[TEAM_SIZE:2 * TEAM_SIZE]

    return {"region": req.region, "blue": blue, "red": red,
            "blue_bans": blue_bans, "red_bans": red_bans, "obj": obj}


# ----------------------------------------------------------------- endpoints
@app.get("/")
def root():
    return {"service": "LoL Blue-vs-Red Win Predictor API",
            "endpoints": ["/health", "/meta", "/predict (POST)", "/docs"]}


@app.get("/health")
def health():
    return {"status": "ok",
            "models_loaded": sorted(ART["models"].keys()),     # ["A", "B"]
            "n_champions": len(META["champ_list"]),
            "n_bans": len(META["ban_list"]),
            "regions": REGIONS}


@app.get("/meta")
def meta():
    """Everything the frontend needs to build aligned dropdowns / toggles."""
    return {
        "regions": [{"code": r, "label": REGION_LABEL.get(r, r.upper()),
                     "base_blue_win_rate": META["region_base_rate"].get(r, META["overall_base_rate"])}
                    for r in REGIONS],
        "team_size": TEAM_SIZE,
        "champions": META["champ_list"],
        "bans": META["ban_list"],
        "objectives": [{"key": k, "label": OBJ_LABEL[k], "model_field": OBJ_API_TO_MODEL[k]}
                       for k in OBJ_API_TO_MODEL],
        "objective_values": ["none", "blue", "red"],
        "models": {
            "A": {"role": "draft-only pre-game baseline (served when all objectives = none)",
                  "test_auc": META["regimes"]["A"]["test_auc"]},
            "B": {"role": "headline model: draft + early objectives (served when any objective set)",
                  "test_auc": META["regimes"]["B"]["test_auc"]},
        },
        "overall_base_blue_win_rate": META["overall_base_rate"],
    }


@app.post("/predict")
def predict(req: PredictRequest):
    """Reuses demo_core.predict_full — identical numbers to the Streamlit demo.

    which_model: "A" when all objectives are none (pre-game baseline), "B" once any
    objective is set (the two-model switch is decided inside demo_core, untouched).
    """
    state = _build_state(req)
    res = C.predict_full(state, ART)
    top_factors = [{"feature": name,
                    "label": _feature_label(name),
                    "shap_value": float(val),
                    "direction": "blue" if val > 0 else "red"}
                   for name, val in res["contribs"]]
    return {
        "blue_win_prob": float(res["prob"]),
        "which_model": res["regime"],                      # "A" or "B"
        "pre_game_baseline": float(res["prob_draft"]),     # regime-A draft-only baseline
        "delta_from_objectives": float(res["delta"]),      # prob - baseline
        "top_factors": top_factors,
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
