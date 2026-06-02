"""
Checkpoint 5 — headless self-test of the demo's prediction path (no web server).

Confirms the behaviour the writeup claims, with the honest two-model demo:
  * all objectives NONE -> regime-A draft-only model; swapping champions barely
    moves P(blue win)                                            [regime A flat]
  * flipping early-objective toggles -> regime-B model; moves it a LOT, correct
    direction                                                    [regime B signal]
  * SHAP signs are sane (blue objective -> +log-odds; draft-only is region-led)
  * duplicate-pick validation fires; empty draft does not crash

Run:  PYTHONPATH=src python3 src/demo_selftest.py
"""
from __future__ import annotations
import demo_core as C

EPS_FLAT = 0.03      # champion swap (regime A) must move prob less than this
EPS_OBJ = 0.10       # two objectives to one side must move prob more than this


def main():
    art = C.load_artifacts()
    champs = art["meta"]["champ_list"]
    OBJ = C.OBJECTIVE_COLS
    fails = []

    def chk(cond, msg):
        print(("  PASS " if cond else "  FAIL ") + msg)
        if not cond:
            fails.append(msg)

    base = C.default_state(art)
    base["region"] = "na1"
    rb = C.predict_full(base, art)
    print("=" * 76)
    print("DEMO SELF-TEST (two-model: A=draft baseline, B=served)")
    print("=" * 76)
    print(f"region=na1  all objectives NONE -> regime {rb['regime']}  P(blue)={rb['prob']:.4f}"
          f"   region base rate={art['meta']['region_base_rate']['na1']:.4f}")
    chk(rb["regime"] == "A", "all-none state is served by the draft-only regime-A model")

    # --- 1) pure champion swap should barely move the (regime-A) prediction ---
    swapped = {**base, "blue": list(base["blue"]), "red": list(base["red"])}
    pool = [c for c in champs if c not in base["blue"] + base["red"]]
    swapped["blue"][0], swapped["blue"][1], swapped["red"][0] = pool[0], pool[1], pool[2]
    rs = C.predict_full(swapped, art)
    print(f"\n[1] swap 3 champions (objectives still NONE)   P(blue)={rs['prob']:.4f}"
          f"   |delta|={abs(rs['prob'] - rb['prob']):.4f}")
    chk(abs(rs["prob"] - rb["prob"]) < EPS_FLAT,
        f"champion swap barely moves prob (|delta|={abs(rs['prob'] - rb['prob']):.4f} < {EPS_FLAT})  [regime A flat]")

    # --- 2) one objective to BLUE already switches to regime B and moves up ---
    one_blue = {**base, "obj": {**base["obj"], "firstTower": 1}}
    r1 = C.predict_full(one_blue, art)
    print(f"\n[2] firstTower -> BLUE        -> regime {r1['regime']}  P(blue)={r1['prob']:.4f}"
          f"   delta vs baseline={r1['delta']:+.4f}")
    chk(r1["regime"] == "B", "setting an objective switches to the served regime-B model")
    chk(r1["delta"] > 0, f"first tower->blue raises prob (delta={r1['delta']:+.4f} > 0)")

    # --- 3) two objectives to BLUE vs RED ---
    blue_obj = {**base, "obj": {**base["obj"], "firstTower": 1, "firstDragon": 1}}
    red_obj = {**base, "obj": {**base["obj"], "firstTower": 2, "firstDragon": 2}}
    rbo, rro = C.predict_full(blue_obj, art), C.predict_full(red_obj, art)
    print(f"\n[3] tower+dragon->BLUE P(blue)={rbo['prob']:.4f} (delta {rbo['delta']:+.4f})   "
          f"tower+dragon->RED P(blue)={rro['prob']:.4f} (delta {rro['delta']:+.4f})")
    chk(rbo["delta"] > EPS_OBJ, f"2 objectives->blue raises a lot (delta={rbo['delta']:+.4f} > {EPS_OBJ})")
    chk(rro["delta"] < -EPS_OBJ, f"2 objectives->red lowers a lot (delta={rro['delta']:+.4f} < -{EPS_OBJ})")

    # --- 4) all 4 one side = extreme spread ---
    p_ab = C.predict_full({**base, "obj": {o: 1 for o in OBJ}}, art)["prob"]
    p_ar = C.predict_full({**base, "obj": {o: 2 for o in OBJ}}, art)["prob"]
    print(f"\n[4] all 4->BLUE P(blue)={p_ab:.4f}   all 4->RED P(blue)={p_ar:.4f}   spread={p_ab - p_ar:+.4f}")
    chk(p_ab > rb["prob"] > p_ar, "directional: all-blue > draft baseline > all-red")
    chk(p_ab - p_ar > 0.40, f"blue-sweep vs red-sweep spread is large ({p_ab - p_ar:.3f} > 0.40)")

    # --- 5) SHAP signs ---
    tower_blue = [v for n, v in rbo["contribs"] if n == "firstTower_blue"]
    print("\n[5] SHAP top factors when tower+dragon -> BLUE (regime B):")
    for n, v in rbo["contribs"][:5]:
        print(f"      {C.friendly(n):<34} {v:+.3f}")
    chk(bool(tower_blue) and tower_blue[0] > 0,
        "SHAP: firstTower_blue has POSITIVE log-odds (pushes toward blue)")

    has_region = any(n.startswith("region=") for n, _ in rb["contribs"])
    print("\n[5b] SHAP top factors in draft-only state (regime A; should be region-led):")
    for n, v in rb["contribs"][:5]:
        print(f"      {C.friendly(n):<34} {v:+.3f}")
    chk(has_region, "SHAP: region appears among top factors in the draft-only (regime A) state")

    # --- 6) duplicate validation ---
    dup_state = {**base, "blue": [base["red"][0]] + base["blue"][1:]}
    dups = C.duplicate_picks(dup_state)
    print(f"\n[6] duplicate-pick check: forced '{base['red'][0]}' onto both teams -> detected={dups}")
    chk(base["red"][0] in dups, "duplicate pick detected by validation")

    # --- 7) missing picks must not crash ---
    sparse = {**base, "blue": [""] * 5, "red": [""] * 5}
    try:
        ps = C.predict_full(sparse, art)["prob"]
        print(f"\n[7] empty draft (no picks) does not crash     P(blue)={ps:.4f}")
        chk(True, "empty draft handled without crashing")
    except Exception as e:
        chk(False, f"empty draft crashed: {e}")

    print("\n" + "=" * 76)
    print(f"RESULT: {'ALL PASS' if not fails else str(len(fails)) + ' FAILURE(S)'}")
    for m in fails:
        print("  - " + m)
    print("=" * 76)
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
