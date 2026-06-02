"""
Checkpoint 5 demo — Streamlit UI for the pooled regime-B LightGBM (+region).

Start:  python3 -m streamlit run src/demo_app.py
(models are loaded from models/demo/ — nothing is retrained at startup.)

This file is UI / copy ONLY. All prediction / encoding / SHAP / two-model
switching logic lives in demo_core (left untouched). The English labels below are
purely for display; internal region codes (na1/kr/euw1) and champion names are
passed through unchanged so the model vocab still lines up.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # make src importable w/o PYTHONPATH

import streamlit as st
import demo_core as C

st.set_page_config(page_title="LoL — Blue vs Red Win Predictor", layout="wide")
art = C.load_artifacts()
meta = art["meta"]
champs, bans, regions = meta["champ_list"], meta["ban_list"], meta["regions"]
AUC_A = meta["regimes"]["A"]["test_auc"]
AUC_B = meta["regimes"]["B"]["test_auc"]

# --------------------------------------------------------------- display labels
EMPTY_LABEL = "(empty)"
NOBAN_LABEL = "(no ban)"
PICK_OPTS = [EMPTY_LABEL] + champs
BAN_OPTS = [NOBAN_LABEL] + bans
RADIO = {"None": 0, "Blue": 1, "Red": 2}
REGION_LABEL = {"na1": "North America (NA)", "kr": "Korea (KR)", "euw1": "Europe West (EUW)"}
OBJ_EN = {"firstBlood": "First Blood", "firstTower": "First Tower",
          "firstDragon": "First Dragon", "firstRiftHerald": "First Rift Herald"}
BLUE, RED = "#2f6bff", "#e23b3b"


def friendly(name: str) -> str:
    """Human-readable English label for a model feature name (SHAP panel)."""
    if name.startswith("region="):
        return f"Regional baseline · {name.split('=', 1)[1].upper()}"
    for o, lab in OBJ_EN.items():
        if name == f"{o}_blue":
            return f"Blue took {lab}"
        if name == f"{o}_red":
            return f"Red took {lab}"
    if name.startswith("blue_pick="):
        return f"Blue picks {name.split('=', 1)[1]}"
    if name.startswith("red_pick="):
        return f"Red picks {name.split('=', 1)[1]}"
    if name.startswith("blue_ban="):
        return f"Blue bans {name.split('=', 1)[1]}"
    if name.startswith("red_ban="):
        return f"Red bans {name.split('=', 1)[1]}"
    return name


# ============================================================ INTRO / ONBOARDING
def render_intro():
    st.title("🎮 League of Legends — Who Wins, Blue or Red?")
    st.markdown(
        "#### Predicts which team (Blue vs Red) is more likely to win a high-elo "
        "League of Legends game.")
    st.caption(
        "📊 Built on ~51,000 ranked games — 17,000 each from **NA**, **EUW**, and **KR** — "
        "patches **16.9–16.11** (mostly 16.10), played **late April to the end of May 2026**.")
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🧩 What this is")
        st.markdown(
            "Pick **five champions for each team** and — if you want — mark who grabbed the "
            "**early objectives** (first blood, first tower, first dragon, first rift herald). "
            "The model returns the probability that the **Blue** team wins.")
    with c2:
        st.markdown("### 🕹️ How to use it")
        st.markdown(
            "1. **Pick a region** (NA, EUW, or KR).\n"
            "2. **Choose 5 champions** for each team.\n"
            "3. *(Optional)* **Flip the objective toggles** to mark which side took each early objective.\n"
            "4. **Read Blue's win probability** and the factors driving it.")

    st.markdown("")
    st.info(
        "💡 **The surprising part — this is the finding, not a bug.**\n\n"
        "Surprisingly, **who you pick barely predicts the winner.** In this high-elo data, "
        "changing champions moves the prediction *almost not at all.* What actually predicts the "
        "game is **early objectives — especially first tower.**\n\n"
        "So when no objective is set, the win rate stays near the **regional average** and "
        "**hardly moves as you swap champions** — that's the real finding, not a glitch.\n\n"
        "👉 **Flip a 'First Tower' toggle and watch the prediction jump.**")

    st.caption(
        "⚠️ UCSD **DSC 148** course project — based on ~51k NA / KR / EUW high-elo games, "
        "for learning and demonstration only.")

    with st.expander("🔬 Technical details (optional)"):
        st.markdown(
            f"""
**Two-model design.** With **no objective set**, the app serves a *draft-only* model
(**regime A**, test AUC ≈ **{AUC_A:.2f}**) that produces the pre-game baseline. The moment
**any objective is set**, it switches to the headline model (**regime B** — a LightGBM with
region one-hot, test AUC ≈ **{AUC_B:.2f}**).

**Why two models?** The "all four early objectives unknown" state occurs in **0.000%** of the
training games (first blood is decided in 99.98% of matches). Feeding that all-zero state to
regime B is out-of-distribution extrapolation and makes it *spuriously* champion-sensitive,
which would contradict the core finding. The draft-only model is genuinely flat there, so it is
the honest choice for the pre-game baseline.

**Data.** ~51,000 Riot **Match-V5** ranked games, 17k each from NA / EUW / KR, patches
**16.9–16.11** (mostly 16.10), late April–May 2026. Features use only **pre-game + early-objective**
information; post-game stats (kills, gold, game duration, inhibitors, baron, …) are excluded to
avoid label leakage.
""")

    st.markdown("")
    if st.button("Continue →", type="primary"):
        st.session_state.entered = True
        st.rerun()


# ============================================================ MAIN PREDICTOR UI
def _objectives_currently_set() -> bool:
    """Read the live objective radios from session_state (populated before rerun)."""
    return any(st.session_state.get(o, "None") != "None" for o in C.OBJECTIVE_COLS)


def render_main():
    head_l, head_r = st.columns([6, 1])
    with head_l:
        st.title("🎮 LoL — Blue vs Red Win Predictor")
        st.caption("Predicts which team is more likely to win a high-elo League of Legends game · "
                   "DSC 148 · loaded from models/demo/ (no retraining at startup)")
    with head_r:
        st.markdown("")
        if st.button("ⓘ How it works"):
            st.session_state.entered = False
            st.rerun()

    # ---------------------------------------------------------------- region
    region = st.selectbox(
        "Region", regions,
        index=regions.index("na1") if "na1" in regions else 0,
        format_func=lambda r: REGION_LABEL.get(r, r.upper()))
    base = meta["region_base_rate"].get(region, meta["overall_base_rate"])

    # -------------------------------------------- mode hint (above the picks)
    if _objectives_currently_set():
        st.success("🟢 **Now using early-game signals** — first tower / dragon / etc. "
                   "This is where the prediction really moves.")
    else:
        st.info("🔵 **Pre-game estimate (draft only)** — this barely changes as you swap "
                "champions, *by design*. Flip an objective toggle below to see it move.")

    # ---------------------------------------------------------------- picks
    cb, cr = st.columns(2)
    blue, red = [], []
    with cb:
        st.subheader("🔵 Blue team")
        for i in range(5):
            blue.append(st.selectbox(f"Blue {i + 1}", PICK_OPTS, index=1 + i, key=f"b{i}"))
    with cr:
        st.subheader("🔴 Red team")
        for i in range(5):
            red.append(st.selectbox(f"Red {i + 1}", PICK_OPTS, index=6 + i, key=f"r{i}"))
    blue = ["" if x == EMPTY_LABEL else x for x in blue]
    red = ["" if x == EMPTY_LABEL else x for x in red]

    # ---------------------------------------------------- bans (optional)
    with st.expander("Bans (optional — defaults to no ban)"):
        cbb, crb = st.columns(2)
        blue_bans, red_bans = [], []
        with cbb:
            for i in range(5):
                blue_bans.append(st.selectbox(f"Blue ban {i + 1}", BAN_OPTS, key=f"bb{i}"))
        with crb:
            for i in range(5):
                red_bans.append(st.selectbox(f"Red ban {i + 1}", BAN_OPTS, key=f"rb{i}"))
        blue_bans = ["" if x == NOBAN_LABEL else x for x in blue_bans]
        red_bans = ["" if x == NOBAN_LABEL else x for x in red_bans]

    # ---------------------------------------------------- objectives
    st.subheader("Early objectives")
    st.caption("Default is all **None** = a pure pre-game / draft-only estimate. "
               "Set who took each objective to switch on the early-game signals.")
    ocols = st.columns(4)
    obj = {}
    for k, o in enumerate(C.OBJECTIVE_COLS):
        with ocols[k]:
            choice = st.radio(OBJ_EN[o], list(RADIO.keys()), index=0, key=o, horizontal=True)
            obj[o] = RADIO[choice]

    state = {"region": region, "blue": blue, "red": red,
             "blue_bans": blue_bans, "red_bans": red_bans, "obj": obj}

    # ---------------------------------------------------- friendly validation
    dups = C.duplicate_picks(state)
    if dups:
        st.warning(f"⚠️ Duplicate champion(s): **{', '.join(dups)}** — the same champion can't be "
                   f"picked twice (across or within teams). Please fix the picks above to see a "
                   f"prediction.")
        st.stop()
    n_empty = sum(1 for x in blue + red if not x)
    if n_empty:
        st.caption(f"ℹ️ {n_empty} champion slot(s) left empty — treated as “no pick”; "
                   f"the prediction still works.")

    # ---------------------------------------------------- predict
    res = C.predict_full(state, art)
    prob, prob_draft, delta, used = res["prob"], res["prob_draft"], res["delta"], res["regime"]
    contribs = res["contribs"]

    st.divider()
    blue_pct, red_pct = prob * 100, (1 - prob) * 100
    favored = ("Blue is favored" if prob > 0.515 else
               "Red is favored" if prob < 0.485 else "It's basically even")
    fav_color = BLUE if prob > 0.515 else RED if prob < 0.485 else "#888"

    st.markdown(
        f"<div style='text-align:center;margin-bottom:6px'>"
        f"<span style='font-size:68px;font-weight:800;color:{BLUE};line-height:1'>{blue_pct:.0f}%</span>"
        f"<div style='font-size:17px;color:#888;margin-top:2px'>Blue win probability</div>"
        f"<div style='font-size:15px;font-weight:600;color:{fav_color};margin-top:2px'>{favored}</div>"
        f"</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='display:flex;height:30px;border-radius:8px;overflow:hidden;"
        f"font-size:13px;font-weight:700;color:white;margin:6px 0 14px 0'>"
        f"<div style='width:{blue_pct:.4f}%;background:{BLUE};display:flex;align-items:center;"
        f"justify-content:center'>Blue {blue_pct:.0f}%</div>"
        f"<div style='width:{red_pct:.4f}%;background:{RED};display:flex;align-items:center;"
        f"justify-content:center'>Red {red_pct:.0f}%</div></div>", unsafe_allow_html=True)

    m1, m2 = st.columns(2)
    m1.metric("Pre-game baseline (draft only)", f"{prob_draft * 100:.1f}%",
              help=f"Draft-only (regime A) output ≈ {REGION_LABEL.get(region, region.upper())} "
                   f"empirical base rate of {base * 100:.1f}%")
    m2.metric("Change from early objectives", f"{delta * 100:+.1f} pp",
              help="Current win probability minus the pre-game baseline for the same draft.")

    if used == "A":
        st.info(
            f"🔵 **This is a draft-only / pre-game estimate** (all objectives = None). This study "
            f"found pure draft to be **almost unpredictable** (regime A, AUC ≈ {AUC_A:.2f}). The win "
            f"rate sits near the **{REGION_LABEL.get(region, region.upper())} base rate "
            f"({base * 100:.1f}%)** and **barely moves when you swap champions** — that's the real "
            f"finding, **not a broken demo**.\n\n"
            f"(Note: the “all objectives unknown” state is 0.000% of real games, so for the pre-game "
            f"view we deliberately use the draft-only model — it's the honest baseline.)\n\n"
            f"👉 Flip any objective toggle above and watch the win rate jump.")
    else:
        st.success(
            f"🟢 **Early objectives are set — now serving the headline model (regime B, AUC ≈ "
            f"{AUC_B:.2f}).** These objectives moved Blue's win probability by "
            f"**{delta * 100:+.1f} percentage points** versus the pre-game baseline. This is the "
            f"study's main point: **early objectives, not the draft, decide the game.**")

    # ---------------------------------------------------- SHAP attribution
    st.subheader(f"Top contributing factors (SHAP · log-odds · model {used})")
    if contribs:
        for name, val in contribs:
            arrow = f"<span style='color:{BLUE};font-weight:700'>▲ pushes Blue</span>" if val > 0 \
                else f"<span style='color:{RED};font-weight:700'>▼ pushes Red</span>"
            st.markdown(
                f"- **{friendly(name)}** — {arrow} &nbsp;`{val:+.3f} log-odds`",
                unsafe_allow_html=True)
    else:
        st.write("(no notable contributing factors)")
    st.caption("SHAP values are log-odds contributions: positive = pushes Blue's win probability up, "
               "negative = pushes Red's up. In the pure pre-game state you'll see essentially only the "
               "“regional baseline” mattering — the champion terms contribute almost nothing.")


# ----------------------------------------------------------------------- router
if not st.session_state.get("entered", False):
    render_intro()
else:
    render_main()
