#!/usr/bin/env python3
"""
DSC 148 — LoL "which team wins" classifier: data-collection stage.

Pulls fresh, current-patch RANKED_SOLO matches from three regions (NA, KR, EUW)
via the Riot Games API and writes one wide-format CSV (one row per match) that the
modeling pipeline consumes.

Leakage categories (must match the modeling pipeline):
  - DRAFT (champ picks/ids, summoner spells, bans)         -> SAFE
  - first-objective flags (firstBlood/Tower/Dragon/Herald) -> ALLOWED
  - end-of-game kill counts + gameDuration                 -> LEAKY (contrast-only)

Key requirements honored here:
  - API key read from env RIOT_API_KEY (never hardcoded).
  - Regions run CONCURRENTLY (one worker per regional routing host).
  - Header-aware token-bucket rate limiting (X-App-Rate-Limit / X-Method-Rate-Limit),
    429 Retry-After, exponential backoff on 5xx/network errors, skip on 404.
  - Resumable: match-id seen-set persisted in SQLite, rows appended+flushed
    incrementally, per-region kept counts recomputed from the CSV on restart.
  - Output contains NO player identifiers (no puuids, no summoner names).
"""

import argparse
import csv
import json
import logging
import os
import random
import sqlite3
import sys
import threading
import time
from collections import Counter, deque
from datetime import datetime, timezone

import requests

# ----------------------------------------------------------------------------- routing
# Match-V5 uses REGIONAL routing; everything else uses the PLATFORM host.
PLATFORM_HOSTS = {
    "na1": "na1.api.riotgames.com",
    "kr": "kr.api.riotgames.com",
    "euw1": "euw1.api.riotgames.com",
}
REGIONAL_HOSTS = {
    "na1": "americas.api.riotgames.com",
    "kr": "asia.api.riotgames.com",
    "euw1": "europe.api.riotgames.com",
}

ROLE_ORDER = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]

DDRAGON_VERSIONS = "https://ddragon.leagueoflegends.com/api/versions.json"
DDRAGON_CHAMPS = "https://ddragon.leagueoflegends.com/cdn/{ver}/data/en_US/champion.json"

DEFAULT_OUT = "data/raw_matches.csv"
DEFAULT_SEEN = "checkpoints/seen_match_ids.sqlite"
DEFAULT_PROG = "checkpoints/progress.json"

# Dev-key defaults; overwritten as soon as real headers arrive.
DEFAULT_APP_LIMITS = [(20, 1), (100, 120)]

log = logging.getLogger("collect")


# ============================================================================= limiter
class Limiter:
    """Sliding-window rate limiter supporting multiple (limit, window) buckets.

    A single shared timestamp deque is evaluated against every bucket, so limits
    can be replaced live from response headers without losing request history.
    """

    def __init__(self, limits):
        self.lock = threading.Lock()
        self.limits = list(limits)
        self.events = deque()

    def _prune(self, now):
        if not self.limits:
            return
        maxw = max(w for _, w in self.limits)
        while self.events and self.events[0] <= now - maxw:
            self.events.popleft()

    def acquire(self):
        while True:
            with self.lock:
                now = time.monotonic()
                self._prune(now)
                wait = 0.0
                for limit, window in self.limits:
                    within = [t for t in self.events if t > now - window]
                    if len(within) >= limit:
                        idx = len(within) - limit
                        wait = max(wait, within[idx] + window - now)
                if wait <= 0:
                    self.events.append(now)
                    return
            time.sleep(min(wait, 2.0))

    def set_limits(self, limits):
        if not limits:
            return
        with self.lock:
            self.limits = list(limits)


class Limiters:
    """Per-host registry of app limiters and per-(host, method) limiters."""

    def __init__(self):
        self.lock = threading.Lock()
        self._app = {}
        self._method = {}

    def app(self, host):
        with self.lock:
            if host not in self._app:
                self._app[host] = Limiter(DEFAULT_APP_LIMITS)
            return self._app[host]

    def method(self, host, key, create=False):
        k = (host, key)
        with self.lock:
            lim = self._method.get(k)
            if lim is None and create:
                lim = self._method[k] = Limiter([])
            return lim


def parse_limit_header(h):
    out = []
    for part in (h or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            a, b = part.split(":")
            out.append((int(a), int(b)))
        except ValueError:
            continue
    return out


# =============================================================================== seen
class SeenStore:
    """Persistent set of processed match IDs (kept, filtered, or 404'd).

    Backed by SQLite for durability; mirrored in an in-memory set for O(1)
    membership checks on the hot path. Thread-safe.
    """

    def __init__(self, path):
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("CREATE TABLE IF NOT EXISTS seen(match_id TEXT PRIMARY KEY)")
        self.conn.commit()
        self.ids = set(r[0] for r in self.conn.execute("SELECT match_id FROM seen"))
        self._pending = 0

    def __contains__(self, mid):
        return mid in self.ids

    def add(self, mid):
        with self.lock:
            if mid in self.ids:
                return
            self.ids.add(mid)
            self.conn.execute("INSERT OR IGNORE INTO seen(match_id) VALUES(?)", (mid,))
            self._pending += 1

    def add_many(self, mids):
        with self.lock:
            new = [(m,) for m in mids if m and m not in self.ids]
            for (m,) in new:
                self.ids.add(m)
            if new:
                self.conn.executemany("INSERT OR IGNORE INTO seen(match_id) VALUES(?)", new)
                self.conn.commit()

    def commit(self):
        with self.lock:
            if self._pending:
                self.conn.commit()
                self._pending = 0

    def __len__(self):
        return len(self.ids)


# =============================================================================== sink
class CsvSink:
    """Thread-safe incremental CSV writer. Appends (resume-safe), flushes per row."""

    def __init__(self, path, fieldnames):
        self.lock = threading.Lock()
        self.fieldnames = fieldnames
        fresh = (not os.path.exists(path)) or os.path.getsize(path) == 0
        self.f = open(path, "a", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.f, fieldnames=fieldnames, extrasaction="ignore")
        if fresh:
            self.w.writeheader()
            self.f.flush()
        self._n = 0

    def write(self, row):
        with self.lock:
            self.w.writerow(row)
            self._n += 1
            self.f.flush()
            if self._n % 25 == 0:
                os.fsync(self.f.fileno())

    def close(self):
        with self.lock:
            self.f.flush()
            try:
                os.fsync(self.f.fileno())
            except OSError:
                pass
            self.f.close()


# ============================================================================= context
class Ctx:
    def __init__(self, args, key, limiters, seen, sink, champ_map, accepted_prefixes):
        self.args = args
        self.key = key
        self.limiters = limiters
        self.seen = seen
        self.sink = sink
        self.champ_map = champ_map
        self.accepted_prefixes = accepted_prefixes
        self.stop = threading.Event()
        self.max_retries = args.max_retries
        self.kept = {}
        self.targets = {}
        self.stats = {}
        self.errors = Counter()
        self._lock = threading.Lock()
        self._noted = set()

    def bump(self, key, n=1):
        with self._lock:
            self.errors[key] += n

    def note_limit(self, host, al, alc, ml, mlc):
        with self._lock:
            if host in self._noted:
                return
            self._noted.add(host)
        log.info(
            "[rate] %s app=%s count=%s | method=%s count=%s (header-aware limiter adapting)",
            host, al, alc, ml, mlc,
        )

    def save_progress(self):
        data = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "targets": self.targets,
            "kept": dict(self.kept),
            "seen_total": len(self.seen),
            "errors": dict(self.errors),
        }
        tmp = self.args.progress + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.args.progress)


# ============================================================================ http core
def riot_get(sess, host, path, params, ctx, method_key):
    """GET with header-aware limiting, 429 Retry-After, backoff on 5xx/network, 404->None."""
    url = "https://{}{}".format(host, path)
    app_lim = ctx.limiters.app(host)
    transient = 0
    while True:
        if ctx.stop.is_set():
            return None
        app_lim.acquire()
        m_lim = ctx.limiters.method(host, method_key)
        if m_lim is not None:
            m_lim.acquire()
        try:
            r = sess.get(url, params=params, headers={"X-Riot-Token": ctx.key}, timeout=20)
        except requests.RequestException as e:
            transient += 1
            ctx.bump("net_err")
            if transient > ctx.max_retries:
                log.warning("net give-up %s%s: %s", host, path[:48], e)
                return None
            slp = min(2 ** transient, 60)
            log.warning("net %s on %s%s; backoff %ss (try %d)", type(e).__name__, host, path[:40], slp, transient)
            time.sleep(slp)
            continue

        al = r.headers.get("X-App-Rate-Limit")
        if al:
            app_lim.set_limits(parse_limit_header(al))
            ctx.note_limit(host, al, r.headers.get("X-App-Rate-Limit-Count"),
                           r.headers.get("X-Method-Rate-Limit"), r.headers.get("X-Method-Rate-Limit-Count"))
        ml = r.headers.get("X-Method-Rate-Limit")
        if ml:
            ctx.limiters.method(host, method_key, create=True).set_limits(parse_limit_header(ml))

        sc = r.status_code
        if sc == 200:
            try:
                return r.json()
            except ValueError as e:
                log.warning("json parse fail %s: %s", path[:48], e)
                return None
        if sc == 429:
            ra = r.headers.get("Retry-After")
            wait = float(ra) if (ra and ra.isdigit()) else 2.0
            ctx.bump("http_429")
            log.warning("[%s] 429 on %s; sleeping %.1fs (Retry-After=%s)", host, method_key, wait, ra)
            time.sleep(wait + 0.2)
            continue
        if sc == 404:
            ctx.bump("http_404")
            return None
        if sc in (500, 502, 503, 504):
            transient += 1
            ctx.bump("http_5xx")
            if transient > ctx.max_retries:
                log.warning("[%s] %d give-up on %s", host, sc, path[:48])
                return None
            slp = min(2 ** transient, 60)
            log.warning("[%s] %d on %s; backoff %ss (try %d)", host, sc, method_key, slp, transient)
            time.sleep(slp)
            continue
        if sc in (401, 403):
            # Almost always an expired/invalid dev key. Stop cleanly; state is persisted
            # so a re-run with a fresh RIOT_API_KEY resumes from here.
            ctx.bump("http_auth")
            log.error("[%s] HTTP %d — RIOT_API_KEY invalid or expired. Stopping; "
                      "re-run with a fresh key to resume.", host, sc)
            ctx.stop.set()
            return None
        ctx.bump("http_other")
        log.warning("[%s] HTTP %d on %s %s body=%s", host, sc, method_key, path[:60], r.text[:120])
        return None


# ============================================================================ ddragon
def get_patch_info(sess, n_patches):
    v = sess.get(DDRAGON_VERSIONS, timeout=20).json()
    latest = v[0]
    mm, seen = [], set()
    for ver in v:
        parts = ver.split(".")
        if len(parts) >= 2:
            m = parts[0] + "." + parts[1]
            if m not in seen:
                seen.add(m)
                mm.append(m)
    accepted = mm[: max(1, n_patches)]
    prefixes = [m + "." for m in accepted]
    return latest, accepted, prefixes, v


def get_champion_map(sess, versions):
    """championId -> championName, for translating ban IDs. Tries a few recent
    versions in case the very latest CDN data isn't published yet."""
    for ver in versions[:4]:
        try:
            d = sess.get(DDRAGON_CHAMPS.format(ver=ver), timeout=20).json()
        except (requests.RequestException, ValueError):
            continue
        data = d.get("data", {})
        if not data:
            continue
        cmap = {}
        for name, info in data.items():
            try:
                cmap[int(info["key"])] = info["id"]
            except (KeyError, ValueError, TypeError):
                continue
        if cmap:
            log.info("champion map: %d champions from ddragon %s", len(cmap), ver)
            return cmap
    log.warning("could not build champion map; bans will use numeric IDs")
    return {}


# ============================================================================= parsing
def normalize_duration(d):
    if d is None:
        return 0
    # current patch reports seconds; guard against legacy milliseconds.
    return d // 1000 if d > 100000 else d


def ban_name(cid, champ_map):
    if cid is None or cid < 0:
        return "None"
    return champ_map.get(cid, str(cid))


def order_team(parts):
    """Return exactly 5 participants ordered by ROLE_ORDER, falling back to
    participant order for any role whose teamPosition is missing/duplicated."""
    by_pos, leftover = {}, []
    for p in parts:
        pos = (p.get("teamPosition") or "").upper()
        if pos in ROLE_ORDER and pos not in by_pos:
            by_pos[pos] = p
        else:
            leftover.append(p)
    ordered, li = [], 0
    for role in ROLE_ORDER:
        if role in by_pos:
            ordered.append(by_pos[role])
        elif li < len(leftover):
            ordered.append(leftover[li])
            li += 1
        else:
            ordered.append(None)
    return ordered


def parse_match(m, region, champ_map):
    info = m["info"]
    meta = m["metadata"]
    participants = info.get("participants", [])
    teams = info.get("teams", [])

    tp = {100: [], 200: []}
    for p in participants:
        if p.get("teamId") in tp:
            tp[p["teamId"]].append(p)

    tbynum = {}
    for t in teams:
        tbynum[1 if t.get("teamId") == 100 else 2] = t

    row = {
        "matchId": meta.get("matchId"),
        "region": region,
        "gameVersion": info.get("gameVersion", ""),
        "gameCreation": info.get("gameCreation"),
        "gameDuration": normalize_duration(info.get("gameDuration")),
        "queueId": info.get("queueId"),
    }

    winner = 0
    for n, t in tbynum.items():
        if t.get("win"):
            winner = n
    row["winner"] = winner

    for n, tid in ((1, 100), (2, 200)):
        for k, p in enumerate(order_team(tp.get(tid, [])), start=1):
            if p is None:
                row["t{}_champ{}".format(n, k)] = ""
                row["t{}_champ{}_id".format(n, k)] = ""
                row["t{}_champ{}_spell1".format(n, k)] = ""
                row["t{}_champ{}_spell2".format(n, k)] = ""
            else:
                row["t{}_champ{}".format(n, k)] = p.get("championName", "")
                row["t{}_champ{}_id".format(n, k)] = p.get("championId", "")
                row["t{}_champ{}_spell1".format(n, k)] = p.get("summoner1Id", "")
                row["t{}_champ{}_spell2".format(n, k)] = p.get("summoner2Id", "")

    for n in (1, 2):
        bans = (tbynum.get(n, {}).get("bans") or [])
        bans = sorted(bans, key=lambda b: b.get("pickTurn", 0))
        names = [ban_name(b.get("championId", -1), champ_map) for b in bans[:5]]
        while len(names) < 5:
            names.append("None")
        for k in range(1, 6):
            row["t{}_ban{}".format(n, k)] = names[k - 1]

    def first_of(key):
        for n, t in tbynum.items():
            if t.get("objectives", {}).get(key, {}).get("first"):
                return n
        return 0

    def kills_of(n, key):
        return tbynum.get(n, {}).get("objectives", {}).get(key, {}).get("kills", 0)

    row["firstBlood"] = first_of("champion")
    row["firstTower"] = first_of("tower")
    row["firstDragon"] = first_of("dragon")
    row["firstRiftHerald"] = first_of("riftHerald")
    for n in (1, 2):
        row["t{}_towerKills".format(n)] = kills_of(n, "tower")
        row["t{}_dragonKills".format(n)] = kills_of(n, "dragon")
        row["t{}_baronKills".format(n)] = kills_of(n, "baron")
        row["t{}_inhibitorKills".format(n)] = kills_of(n, "inhibitor")
        row["t{}_riftHeraldKills".format(n)] = kills_of(n, "riftHerald")
    row["firstInhibitor"] = first_of("inhibitor")
    row["firstBaron"] = first_of("baron")
    return row


def build_fieldnames():
    cols = ["matchId", "region", "gameVersion", "gameCreation", "gameDuration", "queueId", "winner"]
    for t in (1, 2):
        for k in range(1, 6):
            cols += ["t{}_champ{}".format(t, k), "t{}_champ{}_id".format(t, k),
                     "t{}_champ{}_spell1".format(t, k), "t{}_champ{}_spell2".format(t, k)]
    for t in (1, 2):
        for k in range(1, 6):
            cols.append("t{}_ban{}".format(t, k))
    cols += ["firstBlood", "firstTower", "firstDragon", "firstRiftHerald"]
    for t in (1, 2):
        cols += ["t{}_towerKills".format(t), "t{}_dragonKills".format(t), "t{}_baronKills".format(t),
                 "t{}_inhibitorKills".format(t), "t{}_riftHeraldKills".format(t)]
    cols += ["firstInhibitor", "firstBaron"]
    return cols


def passes_filter(m, args, ctx):
    info = m.get("info", {})
    gv = info.get("gameVersion", "")
    if not any(gv.startswith(p) for p in ctx.accepted_prefixes):
        return False, "filt_patch"
    if normalize_duration(info.get("gameDuration")) < args.min_duration:
        return False, "filt_duration"
    if info.get("queueId") != args.queue:
        return False, "filt_queue"
    return True, ""


# ============================================================================== seeding
def seed_puuids(sess, region, ctx):
    plat = PLATFORM_HOSTS[region]
    out, seen = [], set()
    for tier in ("challenger", "grandmaster", "master"):
        path = "/lol/league/v4/{}leagues/by-queue/RANKED_SOLO_5x5".format(tier)
        d = riot_get(sess, plat, path, None, ctx, method_key="league-apex")
        if not d:
            log.warning("[%s] no %s data", region, tier)
            continue
        entries = d.get("entries", []) or []
        for e in entries:
            pu = e.get("puuid")
            if not pu:
                sid = e.get("summonerId")
                if sid:
                    sd = riot_get(sess, plat, "/lol/summoner/v4/summoners/{}".format(sid),
                                  None, ctx, method_key="summoner-by-id")
                    pu = (sd or {}).get("puuid")
                if not pu:
                    log.warning("[%s] could not resolve a puuid; skipping entry", region)
                    continue
            if pu not in seen:
                seen.add(pu)
                out.append(pu)
        log.info("[%s] %s: %d entries", region, tier, len(entries))
    return out


# =============================================================================== worker
def worker(region, ctx):
    args = ctx.args
    reg_host = REGIONAL_HOSTS[region]
    target = ctx.targets[region]
    st = ctx.stats[region]
    sess = requests.Session()

    if ctx.kept[region] >= target:
        log.info("[%s] already at target %d (resumed) — nothing to do", region, target)
        return

    puuids = seed_puuids(sess, region, ctx)
    log.info("[%s] seeded %d unique apex puuids", region, len(puuids))

    for i, pu in enumerate(puuids):
        if ctx.stop.is_set() or ctx.kept[region] >= target:
            break
        ids = riot_get(sess, reg_host, "/lol/match/v5/matches/by-puuid/{}/ids".format(pu),
                       {"queue": args.queue, "type": "ranked", "start": 0, "count": 100},
                       ctx, method_key="match-ids") or []
        new_ids = [mid for mid in ids if mid not in ctx.seen]
        st["seen_skip"] += len(ids) - len(new_ids)

        for mid in new_ids:
            if ctx.stop.is_set() or ctx.kept[region] >= target:
                break
            m = riot_get(sess, reg_host, "/lol/match/v5/matches/{}".format(mid),
                         None, ctx, method_key="match-detail")
            ctx.seen.add(mid)  # mark seen whether kept, filtered, or 404
            if m is None:
                continue
            st["fetched"] += 1
            ok, why = passes_filter(m, args, ctx)
            if not ok:
                st[why] += 1
                continue
            try:
                row = parse_match(m, region, ctx.champ_map)
            except Exception as e:  # never let one bad match kill an overnight run
                st["parse_err"] += 1
                log.warning("[%s] parse fail %s: %s", region, mid, e)
                continue
            ctx.sink.write(row)
            ctx.kept[region] += 1
            st["kept"] += 1
            if ctx.kept[region] % 25 == 0:
                ctx.seen.commit()
                ctx.save_progress()
                log.info("[%s] kept %d/%d | fetched %d | seen-skip %d | 429=%d 5xx=%d 404=%d",
                         region, ctx.kept[region], target, st["fetched"], st["seen_skip"],
                         ctx.errors["http_429"], ctx.errors["http_5xx"], ctx.errors["http_404"])
        ctx.seen.commit()
        if (i + 1) % 25 == 0:
            log.info("[%s] seed %d/%d | kept %d/%d", region, i + 1, len(puuids),
                     ctx.kept[region], target)

    ctx.seen.commit()
    ctx.save_progress()
    log.info("[%s] DONE kept=%d/%d (seeds used=%d/%d)",
             region, ctx.kept[region], target, min(i + 1, len(puuids)), len(puuids))


# ============================================================================== summary
def major_minor(ver):
    parts = (ver or "").split(".")
    return parts[0] + "." + parts[1] if len(parts) >= 2 else (ver or "?")


def print_summary(path, regions, accepted):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        log.info("no output rows to summarize at %s", path)
        return
    total = 0
    per_region = Counter()
    patches = Counter()
    side = Counter()
    tmin = tmax = None
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            total += 1
            per_region[row.get("region")] += 1
            patches[major_minor(row.get("gameVersion"))] += 1
            w = row.get("winner")
            if w in ("1", "2"):
                side[w] += 1
            gc = row.get("gameCreation")
            if gc and gc.isdigit():
                ts = int(gc)
                tmin = ts if tmin is None else min(tmin, ts)
                tmax = ts if tmax is None else max(tmax, ts)

    def fmt(ms):
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if ms else "?"

    s1 = side["1"]
    s2 = side["2"]
    sw = s1 + s2
    print("\n" + "=" * 64)
    print("DATA SUMMARY  ({})".format(path))
    print("=" * 64)
    print("total matches      : {}".format(total))
    print("per-region counts  : {}".format({r: per_region.get(r, 0) for r in regions}))
    print("accepted patches   : {}".format(accepted))
    print("patch distribution : {}".format(dict(patches.most_common())))
    print("date range         : {}  ->  {}".format(fmt(tmin), fmt(tmax)))
    if sw:
        print("win-rate by side   : team1={:.3f}  team2={:.3f}  (n={})".format(s1 / sw, s2 / sw, sw))
        if abs(s1 / sw - 0.5) > 0.05:
            print("  WARNING: side win-rate skew >5pp from 50/50 — investigate for bugs.")
    print("=" * 64 + "\n")


# ================================================================================= main
def parse_args():
    ap = argparse.ArgumentParser(description="Riot ranked-match collector (DSC148 LoL winner classifier)")
    ap.add_argument("--regions", default="na1,kr,euw1", help="comma list: na1,kr,euw1")
    ap.add_argument("--per-region-target", type=int, default=17000)
    ap.add_argument("--queue", type=int, default=420, help="420 = ranked solo/duo")
    ap.add_argument("--patches", type=int, default=1, help="how many recent major.minor patches to accept")
    ap.add_argument("--min-duration", type=int, default=300, help="drop games shorter than this (seconds); remakes")
    ap.add_argument("--test", action="store_true", help="small validation run: ~50 matches, NA only")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--seen-db", default=DEFAULT_SEEN)
    ap.add_argument("--progress", default=DEFAULT_PROG)
    ap.add_argument("--max-retries", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def count_existing(path, regions):
    """Recompute per-region kept counts and gather match IDs from an existing CSV
    (authoritative source of truth on resume)."""
    kept = {r: 0 for r in regions}
    ids = []
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                r = row.get("region")
                if r in kept:
                    kept[r] += 1
                mid = row.get("matchId")
                if mid:
                    ids.append(mid)
    return kept, ids


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    random.seed(args.seed)

    key = os.environ.get("RIOT_API_KEY")
    if not key:
        print("ERROR: RIOT_API_KEY environment variable is not set.\n"
              "  Get a key at https://developer.riotgames.com and run:\n"
              "    export RIOT_API_KEY='RGAPI-...'\n", file=sys.stderr)
        sys.exit(2)

    regions = [r.strip() for r in args.regions.split(",") if r.strip()]
    if args.test:
        if regions != ["na1"]:
            log.info("--test: overriding regions -> NA only")
        regions = ["na1"]
        args.per_region_target = min(50, args.per_region_target)
        if args.out == DEFAULT_OUT:
            args.out = "data/raw_matches_test.csv"
        if args.seen_db == DEFAULT_SEEN:
            args.seen_db = "checkpoints/seen_match_ids_test.sqlite"
        if args.progress == DEFAULT_PROG:
            args.progress = "checkpoints/progress_test.json"

    unknown = [r for r in regions if r not in PLATFORM_HOSTS]
    if unknown:
        print("ERROR: unknown region(s) {}; valid: {}".format(unknown, list(PLATFORM_HOSTS)), file=sys.stderr)
        sys.exit(2)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.seen_db) or ".", exist_ok=True)

    meta_sess = requests.Session()
    latest, accepted, prefixes, all_versions = get_patch_info(meta_sess, args.patches)
    champ_map = get_champion_map(meta_sess, all_versions)
    log.info("ddragon latest=%s | accepted patches=%s | prefixes=%s", latest, accepted, prefixes)

    fieldnames = build_fieldnames()
    kept0, existing_ids = count_existing(args.out, regions)
    seen = SeenStore(args.seen_db)
    seen.add_many(existing_ids)  # keep seen-set consistent with the CSV
    sink = CsvSink(args.out, fieldnames)

    ctx = Ctx(args, key, Limiters(), seen, sink, champ_map, prefixes)
    for r in regions:
        ctx.kept[r] = kept0[r]
        ctx.targets[r] = args.per_region_target
        ctx.stats[r] = Counter()

    log.info("regions=%s targets=%s | resumed kept=%s | seen-set loaded=%d | out=%s",
             regions, ctx.targets, kept0, len(seen), args.out)

    threads = [threading.Thread(target=worker, args=(r, ctx), name=r) for r in regions]
    for t in threads:
        t.start()
    try:
        for t in threads:
            while t.is_alive():
                t.join(timeout=0.5)
    except KeyboardInterrupt:
        log.warning("interrupted — signalling workers to stop and persisting state...")
        ctx.stop.set()
        for t in threads:
            t.join()
    finally:
        ctx.seen.commit()
        ctx.sink.close()
        ctx.save_progress()

    log.info("error counters: %s", dict(ctx.errors))
    print_summary(args.out, regions, accepted)


if __name__ == "__main__":
    main()
