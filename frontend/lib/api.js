/* =========================================================================
   lib/api.js — backend contract + Data Dragon helpers.
   The backend base URL comes from NEXT_PUBLIC_API_BASE_URL (default localhost).
   ========================================================================= */

export const API_BASE =
  (process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000").replace(/\/+$/, "");

export const LANES = ["TOP", "JNG", "MID", "BOT", "SUP"];

/* ----------------------------------------------------------- backend calls */
export async function fetchMeta() {
  const r = await fetch(`${API_BASE}/meta`, { cache: "no-store" });
  if (!r.ok) throw new Error(`GET /meta → HTTP ${r.status} ${r.statusText}`);
  return r.json();
}

export async function predict(body) {
  const r = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    let detail = `HTTP ${r.status}`;
    try { const j = await r.json(); if (j && j.detail) detail = j.detail; } catch (_) {}
    const e = new Error(detail);
    e.status = r.status;
    throw e;
  }
  return r.json();
}

/* ----------------------------------------------------------- Data Dragon */
// champ_list value -> Data Dragon image key (handle the few case mismatches)
const DDRAGON_ALIAS = { FiddleSticks: "Fiddlesticks", Renata: "RenataGlasc" };

// pretty display names where the key isn't human-readable
const SPECIAL_NAME = {
  AurelionSol: "Aurelion Sol", MonkeyKing: "Wukong", DrMundo: "Dr. Mundo",
  MissFortune: "Miss Fortune", KSante: "K'Sante", Belveth: "Bel'Veth",
  Chogath: "Cho'Gath", KogMaw: "Kog'Maw", RekSai: "Rek'Sai", TahmKench: "Tahm Kench",
  XinZhao: "Xin Zhao", JarvanIV: "Jarvan IV", LeeSin: "Lee Sin", MasterYi: "Master Yi",
  TwistedFate: "Twisted Fate", Nunu: "Nunu & Willump", Leblanc: "LeBlanc",
  Velkoz: "Vel'Koz", Khazix: "Kha'Zix", FiddleSticks: "Fiddlesticks", Renata: "Renata Glasc",
};

export function displayName(key) {
  if (!key) return "";
  if (SPECIAL_NAME[key]) return SPECIAL_NAME[key];
  return key.replace(/([a-z])([A-Z])/g, "$1 $2");
}

// resolve the live Data Dragon patch (newest first); fall back to a recent one
export async function ddragonVersion() {
  try {
    const r = await fetch("https://ddragon.leagueoflegends.com/api/versions.json");
    const v = await r.json();
    return Array.isArray(v) && v.length ? v[0] : "15.10.1";
  } catch (_) {
    return "15.10.1";
  }
}

export function iconUrl(version, key) {
  const k = DDRAGON_ALIAS[key] || key;
  return `https://ddragon.leagueoflegends.com/cdn/${version}/img/champion/${k}.png`;
}

export const pct = (x, d = 1) => (x * 100).toFixed(d);
