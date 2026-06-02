# LoL Win Predictor — Backend API (FastAPI)

A thin HTTP layer over `src/demo_core.py`. It does **not** retrain or change any
model logic — `/predict` calls the same `demo_core.predict_full` the Streamlit demo
and `demo_selftest.py` use, so the numbers are identical (proven by `verify_api.py`).

> The Streamlit app (`src/demo_app.py`) is left untouched as a reference. This API
> is the backend for a future "real" web frontend.

---

## Run locally

```bash
# from the project root (DSC148/)
PYTHONPATH=src uvicorn api.server:app --reload --port 8000
# or, equivalently:
PORT=8000 python3 src/api/server.py
```

Interactive docs (Swagger UI) are auto-served at `http://localhost:8000/docs`.

**Config via environment variables (for deployment):**

| Var | Default | Purpose |
|---|---|---|
| `PORT` | `8000` | Port to bind. Platforms like Render inject this automatically. |
| `FRONTEND_ORIGIN` | *(empty)* | Comma-separated allowed CORS origins, e.g. `https://your-app.vercel.app`. **localhost on any port is always allowed** for local dev, so you only set this in production. |

---

## Endpoints

### `GET /health`
Confirms the service is up and the models are loaded.
```json
{ "status": "ok", "models_loaded": ["A","B"], "n_champions": 171, "n_bans": 172, "regions": ["na1","kr","euw1"] }
```

### `GET /meta`
Everything the frontend needs to build dropdowns/toggles **aligned to the model vocab**
(so options never mismatch the encoder):
```json
{
  "regions": [ { "code": "na1", "label": "North America (NA)", "base_blue_win_rate": 0.450 }, ... ],
  "team_size": 5,
  "champions": ["Aatrox","Ahri", ...],
  "bans": ["Aatrox", ...],
  "objectives": [ { "key": "first_blood", "label": "First Blood", "model_field": "firstBlood" }, ... ],
  "objective_values": ["none","blue","red"],
  "models": { "A": {"role": "...", "test_auc": 0.54}, "B": {"role": "...", "test_auc": 0.79} }
}
```

### `POST /predict`
**Request**
```json
{
  "region": "na1",
  "blue": ["Aatrox","Ahri","Akali","Alistar","Amumu"],
  "red":  ["Anivia","Annie","Ashe","Aurelion Sol","Azir"],
  "bans": [],
  "objectives": {
    "first_blood": "none", "first_tower": "blue",
    "first_dragon": "none", "first_rift_herald": "none"
  }
}
```
- `objectives` values are `none | blue | red` (omitted objectives default to `none`).
- `bans` is optional (default no ban). Unknown/empty bans are ignored; if given, the
  first 5 are treated as Blue bans and the next 5 as Red bans.

**Response**
```json
{
  "blue_win_prob": 0.7650,
  "which_model": "B",
  "pre_game_baseline": 0.4618,
  "delta_from_objectives": 0.3032,
  "top_factors": [
    { "feature": "firstTower_blue", "label": "Blue took First Tower", "shap_value": 0.46, "direction": "blue" }
  ]
}
```
`which_model` is **A** when all objectives are `none` (draft-only pre-game baseline,
AUC ≈ 0.54) and **B** the moment any objective is set (headline model, AUC ≈ 0.79).
This two-model switch is decided inside `demo_core` and is identical to the demo.

**Validation** (returns `400` with a clear message, never a `500`): duplicate champion,
a team without exactly 5 champions, unknown champion name, unknown region, or an
invalid objective value.

---

## Verify the API didn't change any result

```bash
PYTHONPATH=src python3 src/api/verify_api.py
```
For each case it computes `blue_win_prob` via `demo_core` directly **and** via the API,
and asserts they're equal within `1e-9` (they match exactly). It also checks the
validation `400`s.

---

## Deployment note (for later — Render etc.)

The trained artifacts in **`models/demo/`** (`lgbm_A.joblib`, `lgbm_B.joblib`,
`explainer_A.joblib`, `explainer_B.joblib`, `vocabs.json`, `meta.json`) **must be
deployed alongside the service** — they are loaded **once at startup** (cached in
`demo_core`) by absolute path, so the API works regardless of the process working
directory. Nothing is trained at startup. `data/` and `checkpoints/` are **not**
needed at runtime and stay gitignored.

(No `Dockerfile` / `render.yaml` yet — those come once the frontend direction is set.)
