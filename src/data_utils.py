"""
Shared data loading + schema / leakage definitions for the DSC148 LoL winner
classifier. Every downstream stage (EDA, features, models, demo) imports the
column groups from here so the leakage discipline is defined in exactly ONE place.

Side convention (from the collector):
    winner == 1  ->  team1 == blue == teamId 100
    winner == 2  ->  team2 == red  == teamId 200
Objective flags (firstBlood/firstTower/...):  0 = none, 1 = blue/team1, 2 = red/team2
"""
import os
import pandas as pd

SEED = 42
REGIONS = ["na1", "kr", "euw1"]
DEFAULT_CSV = os.path.join("data", "raw_matches.csv")

# ----------------------------------------------------------------------------- schema groups
ID_COLS = ["matchId", "region", "gameVersion", "gameCreation", "gameDuration", "queueId"]
TARGET = "winner"  # 1 = blue, 2 = red

TEAMS = (1, 2)            # 1 = blue, 2 = red
SLOTS = (1, 2, 3, 4, 5)

CHAMP_NAME_COLS = [f"t{t}_champ{i}" for t in TEAMS for i in SLOTS]
CHAMP_ID_COLS = [f"t{t}_champ{i}_id" for t in TEAMS for i in SLOTS]
SPELL_COLS = [f"t{t}_champ{i}_spell{s}" for t in TEAMS for i in SLOTS for s in (1, 2)]
BAN_COLS = [f"t{t}_ban{i}" for t in TEAMS for i in SLOTS]

# DRAFT / SAFE: the 10 picks, their summoner spells, the 10 bans
SAFE_DRAFT_COLS = CHAMP_NAME_COLS + CHAMP_ID_COLS + SPELL_COLS + BAN_COLS

# ALLOWED early objectives (region-agnostic early-game state)
OBJECTIVE_COLS = ["firstBlood", "firstTower", "firstDragon", "firstRiftHerald"]

# LEAKY: post-hoc / end-of-game state. NEVER use in a real model. One labeled
# "leaky" contrast run only, for the writeup.
LEAKY_KILL_COLS = [f"t{t}_{k}Kills" for t in TEAMS
                   for k in ("tower", "dragon", "baron", "inhibitor", "riftHerald")]
LEAKY_COLS = LEAKY_KILL_COLS + ["firstInhibitor", "firstBaron", "gameDuration", "gameCreation"]

# columns that should hold a champion name (for vocab building)
ALL_CHAMP_NAME_COLS = CHAMP_NAME_COLS
BLUE_CHAMP_COLS = [f"t1_champ{i}" for i in SLOTS]
RED_CHAMP_COLS = [f"t2_champ{i}" for i in SLOTS]
NO_BAN_TOKENS = {"None", "-1", "", "nan"}


def major_minor(ver) -> str:
    """'16.10.1' -> '16.10'."""
    parts = str(ver).split(".")
    return parts[0] + "." + parts[1] if len(parts) >= 2 else str(ver)


def load_matches(path: str = DEFAULT_CSV, validate: bool = True) -> pd.DataFrame:
    """Load the raw match CSV and attach a few derived convenience columns.

    Adds:
      patch     -> major.minor string (e.g. '16.10')
      blue_win  -> 1 if blue/team1 won else 0  (the modeling target)
    """
    df = pd.read_csv(path)
    if validate:
        expected = set(ID_COLS + [TARGET] + SAFE_DRAFT_COLS + OBJECTIVE_COLS + LEAKY_KILL_COLS
                       + ["firstInhibitor", "firstBaron"])
        missing = expected - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing expected columns: {sorted(missing)}")
    df["patch"] = df["gameVersion"].map(major_minor)
    df["blue_win"] = (df[TARGET] == 1).astype(int)
    return df


def champion_vocabulary(df: pd.DataFrame) -> list:
    """Sorted unique champion names seen in any pick slot (the model vocab)."""
    vals = pd.unique(df[CHAMP_NAME_COLS].values.ravel())
    return sorted(str(v) for v in vals if pd.notna(v) and str(v) != "")
