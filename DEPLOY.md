# Deploying — Backend on Render, Frontend on Vercel

The code is one self-contained repo. Backend (FastAPI) → **Render**, frontend
(Next.js) → **Vercel**. Deploy the **backend first** so you have its public URL to
give the frontend.

Everything that can be automated from the CLI is already done (repo + commit +
`render.yaml` + slim `requirements-api.txt`). The only remaining steps are the
one-time **OAuth "connect repo"** clicks in each dashboard — those need your
Render/Vercel login and can't be scripted without your account tokens.

---

## 1) Backend → Render (do this first)

1. Go to **dashboard.render.com → New → Blueprint** and pick this GitHub repo.
   Render reads **`render.yaml`** and pre-fills everything (Python service, build
   `pip install -r requirements-api.txt`, start `uvicorn api.server:app …`, health
   check `/health`).
2. When prompted, set the one `sync:false` var:
   - `FRONTEND_ORIGIN` → leave blank for now (fill in after step 2 with your Vercel
     URL). CORS already allows any `localhost` for local dev regardless.
3. Click **Apply / Deploy**. First build takes a few minutes.
4. Note the service URL, e.g. `https://lol-win-predictor-api.onrender.com`.
   Verify: open `…/health` (should be `{"status":"ok","models_loaded":["A","B"], …}`)
   and `…/meta`.

> Free instances sleep after ~15 min idle and cold-start in ~30–60 s on the next
> request — fine for a demo. `models/demo/` ships in the repo and loads once at boot.

## 2) Frontend → Vercel

1. Go to **vercel.com → Add New → Project** and import this GitHub repo.
2. **Important:** set **Root Directory = `frontend`** (the Next.js app lives in the
   `frontend/` subfolder). Framework auto-detects as **Next.js**; leave build/output
   at the defaults.
3. Add an environment variable:
   | Key | Value |
   |---|---|
   | `NEXT_PUBLIC_API_BASE_URL` | your Render URL from step 1, e.g. `https://lol-win-predictor-api.onrender.com` |

   (`NEXT_PUBLIC_` is required so it's readable in the browser. No trailing slash.)
4. **Deploy.** You'll get a URL like `https://lol-win-predictor.vercel.app`.

## 3) Close the CORS loop

Back on **Render → your service → Environment**, set
`FRONTEND_ORIGIN=https://<your-app>.vercel.app` and save (it redeploys). Now the
browser app is allowed to call the API.

---

## Checklist

- [ ] Render `/health` and `/meta` return JSON over **https**.
- [ ] Vercel site loads; **all objectives None → ~46% Regime A**; flip **First Tower →
      Blue → ~76% Regime B**. (If it shows "Backend unreachable", the
      `NEXT_PUBLIC_API_BASE_URL` is wrong or the backend is still cold-starting.)
- [ ] `FRONTEND_ORIGIN` on Render matches the Vercel domain (no CORS errors in the
      browser console).

## Notes / gotchas

- **https→http is blocked by browsers.** The deployed frontend must call an **https**
  backend (Render gives you https automatically). Don't point a deployed site at
  `http://localhost:8008`.
- The backend uses the **slim** `requirements-api.txt` (no torch/streamlit/shap).
  Native LightGBM TreeSHAP is used for `top_factors` and is identical to the shap
  explainer (verified, max|diff| = 0.0).
- To re-verify results anytime: `PYTHONPATH=src python3 src/api/verify_api.py`.
