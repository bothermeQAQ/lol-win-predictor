"""
verify_api.py — prove the FastAPI layer changes NOTHING about the predictions.

For a battery of inputs (all-none pre-game, first_tower->blue, multiple objectives,
different regions, a champion swap), blue_win_prob is computed two ways:
  (a) directly via demo_core.predict_full   — the exact path the Streamlit demo and
                                               demo_selftest.py use, and
  (b) through the HTTP API                   — FastAPI TestClient -> POST /predict
and asserts they are bit-equal (|a - b| <= 1e-9). which_model (A/B) must match too.

This is the key evidence that wrapping demo_core in HTTP did not perturb any result.

Run:  PYTHONPATH=src python3 src/api/verify_api.py
"""
from __future__ import annotations
import os
import sys
import warnings

warnings.filterwarnings("ignore")
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import demo_core as C
from fastapi.testclient import TestClient
import api.server as server          # imports the app; loads models once (cached)

TOL = 1e-9
ART = server.ART
client = TestClient(server.app)

CHAMPS = ART["meta"]["champ_list"]
BLUE = CHAMPS[0:5]
RED = CHAMPS[5:10]
OTHER = CHAMPS[10:15]                 # distinct from BLUE/RED for the swap case

S2I = {"none": 0, "blue": 1, "red": 2}
A2M = {"first_blood": "firstBlood", "first_tower": "firstTower",
       "first_dragon": "firstDragon", "first_rift_herald": "firstRiftHerald"}


def direct(region, blue, red, obj_api):
    """blue_win_prob the way a demo_core caller would compute it."""
    state = {"region": region, "blue": list(blue), "red": list(red),
             "blue_bans": [None] * 5, "red_bans": [None] * 5,
             "obj": {field: S2I[obj_api.get(api_key, "none")]
                     for api_key, field in A2M.items()}}
    res = C.predict_full(state, ART)
    return res["prob"], res["regime"]


def via_api(region, blue, red, obj_api):
    """blue_win_prob the way the HTTP frontend would get it."""
    payload = {"region": region, "blue": list(blue), "red": list(red), "objectives": obj_api}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200, f"POST /predict -> {r.status_code}: {r.text}"
    j = r.json()
    return j["blue_win_prob"], j["which_model"]


CASES = [
    ("all-none pre-game (na1)",        "na1",  BLUE,  RED, {}),
    ("champion swap, all-none (na1)",  "na1",  OTHER, RED, {}),
    ("first_tower->blue (na1)",        "na1",  BLUE,  RED, {"first_tower": "blue"}),
    ("tower+dragon->blue (kr)",        "kr",   BLUE,  RED, {"first_tower": "blue", "first_dragon": "blue"}),
    ("all four->red (euw1)",           "euw1", BLUE,  RED, {"first_blood": "red", "first_tower": "red",
                                                            "first_dragon": "red", "first_rift_herald": "red"}),
    ("mixed objectives (kr)",          "kr",   BLUE,  RED, {"first_blood": "blue", "first_tower": "red"}),
]


def main():
    print("=" * 92)
    print(f"{'case':34}{'demo_core':>14}{'API':>14}{'|diff|':>12}   model")
    print("=" * 92)
    all_ok = True
    for name, region, blue, red, obj_api in CASES:
        dp, dm = direct(region, blue, red, obj_api)
        ap, am = via_api(region, blue, red, obj_api)
        diff = abs(dp - ap)
        ok = (diff <= TOL) and (dm == am)
        all_ok &= ok
        print(f"{name:34}{dp:14.10f}{ap:14.10f}{diff:12.2e}   {dm}=={am}  {'OK' if ok else 'FAIL'}")
    print("=" * 92)

    # also confirm the friendly 400s (not 500s) for bad input
    print("input-validation (expect HTTP 400, clear message):")
    bad = [
        ("duplicate champion", {"region": "na1", "blue": [RED[0]] + BLUE[1:5], "red": RED,
                                "objectives": {}}),
        ("only 4 champions",   {"region": "na1", "blue": BLUE[:4], "red": RED, "objectives": {}}),
        ("unknown champion",   {"region": "na1", "blue": ["NotAChampion"] + BLUE[1:5], "red": RED,
                                "objectives": {}}),
        ("unknown region",     {"region": "xx9", "blue": BLUE, "red": RED, "objectives": {}}),
        ("bad objective value", {"region": "na1", "blue": BLUE, "red": RED,
                                 "objectives": {"first_tower": "purple"}}),
    ]
    val_ok = True
    for label, payload in bad:
        r = client.post("/predict", json=payload)
        ok = r.status_code == 400
        val_ok &= ok
        msg = r.json().get("detail", r.text) if r.headers.get("content-type", "").startswith("application/json") else r.text
        print(f"  [{ 'OK ' if ok else 'BAD'}] {label:22} -> {r.status_code}  {str(msg)[:70]}")

    print("=" * 92)
    ok = all_ok and val_ok
    print("RESULT:", "API OUTPUT == demo_core OUTPUT (all equal within 1e-9) + validation 400s OK"
          if ok else "MISMATCH / VALIDATION PROBLEM")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
