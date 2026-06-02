# Draft Lab — Blue-Side Win Model (frontend)

A broadcast-grade Next.js frontend for the DSC148 League of Legends **blue-side win
predictor**. It talks to the existing FastAPI backend (`src/api/server.py`) and renders
the model's "two-engine" story:

- **Dead Draft** — while all early objectives are `none`, the readout is served by the
  *draft-only* baseline (regime **A**, AUC ≈ 0.54). Swapping champions barely moves the
  number. That flatness is the point.
- **Live Map** — the moment you flip any early objective, the readout switches to the
  *draft + early-objectives* model (regime **B**, AUC ≈ 0.79). The number jumps, the win
  bar fires a shockwave, and the SHAP bars stretch. This is the model pricing in map state.

Everything (regions, baselines, the 172-champion vocab, objective fields) comes from the
backend's `GET /meta` — nothing is hardcoded. Champion portraits load straight from the
official **Data Dragon** CDN (the live patch is resolved from `versions.json`), so new
champions (Ambessa / Aurora / Mel …) get art automatically. No images are scraped or stored.

---

## 1. Run the backend (FastAPI, port 8000)

From the DSC148 project root:

```bash
PYTHONPATH=src uvicorn api.server:app --reload --port 8000
# or:  PORT=8000 python3 src/api/server.py
```

> **On this machine** port 8000 was already taken by another project's API, so the LoL
> backend is run on **8008** instead (`PORT=8008 … python3 src/api/server.py`) and
> `.env.local` points at `http://localhost:8008`. If 8000 is free for you, use it and set
> `.env.local` back to `:8000`. The chosen port is the only thing that has to match.

Sanity check: open <http://localhost:8008/meta> — you should see regions + champions.
(The backend already allows any `localhost` origin in dev, so no CORS setup is needed.)

## 2. Run this frontend (Next.js)

```bash
# from this folder
cp .env.local.example .env.local      # default points at http://localhost:8000
npm install
npm run dev
```

Open <http://localhost:3000>.

If the backend isn't running you'll get a clear "Backend unreachable" screen with the exact
URL it tried and a **Retry** button — never a blank page.

---

## How to read the screen

1. **Region tabs** (NA / KR / EUW) — broadcast-style channel selectors; each shows the
   region's real blue-side baseline straight from `/meta` (e.g. *KR baseline 49.1% blue side*).
2. **Cockpit** — Blue side (left) and Red side (right) each have 5 lane slots
   (TOP/JNG/MID/BOT/SUP). Click a slot to open the searchable champion wall; pick a portrait
   to fill it. A champion can't be taken twice (across or within teams). The center column is
   the big win-probability readout, bar, and current-model tag.
3. **Map Events Console** — First Blood / First Tower / First Dragon / First Rift Herald, each
   a Blue / — / Red tri-state. Any non-`none` flips the model to regime B. First Tower → Blue
   is the loudest swing.
4. **Contribution Breakdown** — the backend's `top_factors` as blue-ward / red-ward SHAP bars.

### Two honesty guardrails (printed in the UI)

- **Positions are just for arranging your draft** — the model only uses *which champions* are
  picked, not their lanes.
- **Two engines, one screen** — the pre-game % is a draft-only baseline; flipping an objective
  switches to the model that includes early objectives, which is *why* that flip jumps so much.

---

## 3. Deploy to Vercel

1. Push this folder to a Git repo and import it in Vercel (framework auto-detected as Next.js).
2. In **Project → Settings → Environment Variables**, set:

   | Key | Value |
   |---|---|
   | `NEXT_PUBLIC_API_BASE_URL` | the public URL of your deployed FastAPI backend |

   (`NEXT_PUBLIC_` is required so the value is available in the browser.)
3. Deploy. Then, on the **backend**, add your Vercel origin to `FRONTEND_ORIGIN` so CORS allows
   it, e.g. `FRONTEND_ORIGIN=https://your-app.vercel.app`.

> The backend must be reachable over **HTTPS** from the deployed site (browsers block
> https→http requests). For a quick demo you can keep the frontend local and point it at
> `http://localhost:8000`.

---

## Project layout

```
app/
  layout.jsx        root layout + metadata
  page.jsx          'use client' — state, fetch wiring, animation logic
  globals.css       the whole broadcast design system
components/
  TopBar.jsx        wordmark + region tabs + live chip
  TeamColumn.jsx    blue/red 5-slot draft columns (+ ChampSlot)
  WinCore.jsx       win % + bar + shockwave + impact readout
  MapConsole.jsx    early-objectives tri-state grid
  ShapPanel.jsx     SHAP contribution bars
  Banners.jsx       engine-switch banner + honesty guardrails
  ChampPicker.jsx   searchable champion wall overlay
  Status.jsx        boot / error / predict-error states
lib/
  api.js            /meta + /predict calls, Data Dragon helpers, name maps
preview/            standalone static demo (mock backend) — not part of the build
```

The `preview/` folder is a self-contained HTML mock of the same UI (no backend needed) used
for design review; it isn't compiled or deployed.
