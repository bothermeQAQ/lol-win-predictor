"""
Checkpoint 1 — Load + EDA for the LoL winner classifier.

Produces figures in eda/ and prints a findings summary. Read-only w.r.t. data.
Run:  python3 src/eda.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from data_utils import (
    load_matches, REGIONS, CHAMP_NAME_COLS, BAN_COLS, OBJECTIVE_COLS,
    BLUE_CHAMP_COLS, RED_CHAMP_COLS, LEAKY_KILL_COLS, NO_BAN_TOKENS, SEED,
)

OUT = "eda"
os.makedirs(OUT, exist_ok=True)
np.random.seed(SEED)
plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight", "font.size": 9})

SIDE = {0: "none", 1: "blue", 2: "red"}


def hr(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def main():
    df = load_matches()
    n = len(df)

    # ---------------------------------------------------------------- overview
    hr("1. DATASET OVERVIEW")
    print(f"matches            : {n}")
    print(f"columns            : {df.shape[1]} (73 raw + derived patch/blue_win)")
    print(f"per-region counts  : {df['region'].value_counts().to_dict()}")
    print(f"patch distribution : {df['patch'].value_counts().to_dict()}")
    dt = pd.to_datetime(df['gameCreation'], unit='ms', utc=True)
    print(f"date range (UTC)   : {dt.min():%Y-%m-%d %H:%M}  ->  {dt.max():%Y-%m-%d %H:%M}")
    print(f"queueId values     : {df['queueId'].value_counts().to_dict()}")
    print(f"winner values      : {df['winner'].value_counts().to_dict()}  "
          f"(expect only 1/2; any 0 = no-win parse)")
    dur = df['gameDuration']
    print(f"gameDuration (s)   : min={dur.min()} median={dur.median():.0f} max={dur.max()} "
          f"(<300 should be filtered)")
    nblank = (df[CHAMP_NAME_COLS] == "").sum().sum() + df[CHAMP_NAME_COLS].isna().sum().sum()
    print(f"blank champ cells  : {int(nblank)} (missing picks)")
    dupes = n - df['matchId'].nunique()
    print(f"duplicate matchIds : {dupes}")

    # ---------------------------------------------------------------- win rate by side
    hr("2. WIN RATE BY SIDE (blue = team1 = teamId100)")
    valid = df[df['winner'].isin([1, 2])]
    k = int((valid['winner'] == 1).sum())
    m = len(valid)
    p_blue = k / m
    lo, hi = wilson(k, m)
    bt = stats.binomtest(k, m, 0.5)
    print(f"OVERALL blue win   : {p_blue:.4f}  (red {1-p_blue:.4f})  n={m}")
    print(f"  95% CI (blue)    : [{lo:.4f}, {hi:.4f}]   deviation from .50 = "
          f"{(p_blue-0.5)*100:+.2f} pp")
    print(f"  binomial p-value : {bt.pvalue:.2e}  -> "
          f"{'SIGNIFICANT' if bt.pvalue < 0.05 else 'not significant'} "
          f"(direction: {'BLUE' if p_blue>0.5 else 'RED'} favored)")
    print("  NOTE: expected ~1-2pp BLUE edge; observed edge is larger and toward RED.")

    print("\n  by region:")
    reg_rows = []
    for r in REGIONS:
        sub = valid[valid['region'] == r]
        pk = (sub['winner'] == 1).mean()
        reg_rows.append((r, len(sub), pk))
        print(f"    {r:5s} blue={pk:.4f} red={1-pk:.4f}  n={len(sub)}")
    print("\n  by patch:")
    patch_rows = []
    for pa in sorted(df['patch'].unique()):
        sub = valid[valid['patch'] == pa]
        pk = (sub['winner'] == 1).mean()
        patch_rows.append((pa, len(sub), pk))
        print(f"    {pa:6s} blue={pk:.4f} red={1-pk:.4f}  n={len(sub)}")

    # corroboration: is the red edge internally consistent across independent signals?
    hr("2b. IS THE RED EDGE INTERNALLY CONSISTENT? (corroboration, not a model)")
    print("  share of games where each first-objective went to blue / red:")
    for c in OBJECTIVE_COLS:
        vc = df[c].value_counts(normalize=True)
        print(f"    {c:16s} blue={vc.get(1,0):.3f}  red={vc.get(2,0):.3f}  none={vc.get(0,0):.3f}")
    print("  mean end-of-game objective kills by side (LEAKY cols, EDA diagnostic only):")
    for k_ in ("tower", "dragon", "baron", "inhibitor", "riftHerald"):
        b = df[f"t1_{k_}Kills"].mean()
        rd = df[f"t2_{k_}Kills"].mean()
        print(f"    {k_:11s} blue={b:.3f}  red={rd:.3f}  (red-blue={rd-b:+.3f})")

    # plot: win rate by side overall/region/patch
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    ax[0].bar(["blue", "red"], [p_blue, 1 - p_blue], color=["#3b6fd6", "#d64545"])
    ax[0].axhline(0.5, ls="--", c="gray"); ax[0].set_title(f"overall (n={m})")
    ax[0].set_ylim(0.4, 0.6); ax[0].set_ylabel("win rate")
    xs = [r[0] for r in reg_rows]
    ax[1].bar(xs, [r[2] for r in reg_rows], color="#3b6fd6", label="blue")
    ax[1].bar(xs, [1 - r[2] for r in reg_rows], bottom=[r[2] for r in reg_rows], color="#d64545", label="red")
    ax[1].axhline(0.5, ls="--", c="white"); ax[1].set_title("by region"); ax[1].legend(fontsize=7)
    xp = [r[0] for r in patch_rows]
    ax[2].bar(xp, [r[2] for r in patch_rows], color="#3b6fd6")
    ax[2].axhline(0.5, ls="--", c="gray"); ax[2].set_ylim(0.4, 0.6); ax[2].set_title("blue win by patch")
    fig.suptitle("Win rate by side  (dashed = 50%)")
    fig.savefig(f"{OUT}/winrate_by_side.png"); plt.close(fig)

    # ---------------------------------------------------------------- win rate by first objective
    hr("3. WIN RATE GIVEN EACH FIRST OBJECTIVE  (P(blue wins | who took it))")
    obj_plot = {}
    for c in OBJECTIVE_COLS:
        row = {}
        for v in (1, 2, 0):
            sub = df[df[c] == v]
            if len(sub):
                row[SIDE[v]] = (sub['blue_win'].mean(), len(sub))
        obj_plot[c] = row
        s = "  ".join(f"{kk}:{vv[0]:.3f}(n={vv[1]})" for kk, vv in row.items())
        lift = (row.get('blue', (np.nan,))[0] - row.get('red', (np.nan,))[0])
        print(f"  {c:16s} P(blue win) -> {s}   | blue-minus-red lift={lift:+.3f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = OBJECTIVE_COLS
    blue_take = [obj_plot[c].get('blue', (np.nan,))[0] for c in labels]
    red_take = [obj_plot[c].get('red', (np.nan,))[0] for c in labels]
    x = np.arange(len(labels)); w = 0.38
    ax.bar(x - w/2, blue_take, w, label="blue took it", color="#3b6fd6")
    ax.bar(x + w/2, red_take, w, label="red took it", color="#d64545")
    ax.axhline(0.5, ls="--", c="gray")
    ax.set_xticks(x); ax.set_xticklabels([l.replace("first", "") for l in labels])
    ax.set_ylabel("P(blue wins)"); ax.set_title("Blue win probability conditioned on each first objective")
    ax.legend()
    fig.savefig(f"{OUT}/winrate_by_objective.png"); plt.close(fig)

    # ---------------------------------------------------------------- champion pick frequency
    hr("4. CHAMPION PICK FREQUENCY")
    picks = df[CHAMP_NAME_COLS].values.ravel()
    picks = pd.Series([p for p in picks if isinstance(p, str) and p])
    pick_counts = picks.value_counts()
    pick_rate = pick_counts / n  # per-match presence (out of 10 picks/game)
    print(f"unique champions picked : {pick_counts.size}")
    print(f"top 15 by pick rate     :")
    for c, v in pick_rate.head(15).items():
        print(f"    {c:16s} {v*100:5.1f}% of games  ({pick_counts[c]} picks)")

    top = pick_rate.head(30)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top.index[::-1], (top.values[::-1]) * 100, color="#3b6fd6")
    ax.set_xlabel("% of games picked"); ax.set_title("Top 30 champions by pick rate")
    fig.savefig(f"{OUT}/champ_pick_top30.png"); plt.close(fig)

    # ---------------------------------------------------------------- ban frequency
    hr("5. CHAMPION BAN FREQUENCY")
    bans = df[BAN_COLS].values.ravel()
    bans = pd.Series([b for b in bans if isinstance(b, str) and b not in NO_BAN_TOKENS])
    ban_counts = bans.value_counts()
    ban_rate = ban_counts / n
    noban = sum(1 for b in df[BAN_COLS].values.ravel()
                if (not isinstance(b, str)) or b in NO_BAN_TOKENS)
    print(f"unique champions banned : {ban_counts.size}")
    print(f"empty/no-ban slots      : {noban} of {n*10} ({noban/(n*10)*100:.1f}%)")
    print(f"top 15 by ban rate      :")
    for c, v in ban_rate.head(15).items():
        print(f"    {c:16s} {v*100:5.1f}% of games  ({ban_counts[c]} bans)")
    top = ban_rate.head(30)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top.index[::-1], (top.values[::-1]) * 100, color="#d64545")
    ax.set_xlabel("% of games banned"); ax.set_title("Top 30 champions by ban rate")
    fig.savefig(f"{OUT}/champ_ban_top30.png"); plt.close(fig)

    # ---------------------------------------------------------------- per-region meta differences
    hr("6. PER-REGION META DIFFERENCES (pick-rate divergence across NA/KR/EUW)")
    reg_pick = {}
    for r in REGIONS:
        sub = df[df['region'] == r]
        s = pd.Series([p for p in sub[CHAMP_NAME_COLS].values.ravel()
                       if isinstance(p, str) and p]).value_counts() / len(sub)
        reg_pick[r] = s
    pr = pd.DataFrame(reg_pick).fillna(0.0)
    pr['range'] = pr[REGIONS].max(axis=1) - pr[REGIONS].min(axis=1)
    pr['mean'] = pr[REGIONS].mean(axis=1)
    divergent = pr[pr['mean'] > 0.03].sort_values('range', ascending=False).head(15)
    print("  champions with the largest cross-region pick-rate spread (mean pick > 3%):")
    print(f"    {'champ':16s} {'na1':>7s} {'kr':>7s} {'euw1':>7s} {'spread':>7s}")
    for c, row in divergent.iterrows():
        print(f"    {c:16s} {row['na1']*100:6.1f}% {row['kr']*100:6.1f}% "
              f"{row['euw1']*100:6.1f}% {row['range']*100:6.1f}pp")

    heat = pr.loc[divergent.index, REGIONS] * 100
    fig, ax = plt.subplots(figsize=(6, 7))
    im = ax.imshow(heat.values, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(heat))); ax.set_yticklabels(heat.index)
    ax.set_xticks(range(3)); ax.set_xticklabels(REGIONS)
    ax.set_title("Pick rate (%) of most region-divergent champions")
    for i in range(len(heat)):
        for j in range(3):
            ax.text(j, i, f"{heat.values[i,j]:.0f}", ha="center", va="center",
                    color="white", fontsize=7)
    fig.colorbar(im, ax=ax, label="% of games")
    fig.savefig(f"{OUT}/pickrate_region_heatmap.png"); plt.close(fig)

    # overview counts figure
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    df['region'].value_counts().reindex(REGIONS).plot.bar(ax=ax[0], color="#3b6fd6")
    ax[0].set_title("matches per region"); ax[0].tick_params(axis='x', rotation=0)
    df['patch'].value_counts().sort_index().plot.bar(ax=ax[1], color="#6aa84f")
    ax[1].set_title("matches per patch"); ax[1].tick_params(axis='x', rotation=0)
    fig.savefig(f"{OUT}/overview_counts.png"); plt.close(fig)

    hr("FIGURES SAVED")
    for f in sorted(os.listdir(OUT)):
        if f.endswith(".png"):
            print(f"  eda/{f}")
    print("\nEDA complete.")


if __name__ == "__main__":
    main()
