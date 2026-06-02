"use client";
import { useState, useEffect, useRef } from "react";
import { displayName, iconUrl, LANES } from "@/lib/api";

export function ChampPicker({ open, target, meta, picks, version, onPick, onClose }) {
  const [q, setQ] = useState("");
  const inputRef = useRef(null);

  useEffect(() => {
    if (open) { setQ(""); setTimeout(() => inputRef.current && inputRef.current.focus(), 60); }
  }, [open, target]);

  useEffect(() => {
    function onKey(e) { if (e.key === "Escape") onClose(); }
    if (open) window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  const takenBlue = new Set(picks.blue.filter(Boolean));
  const takenRed = new Set(picks.red.filter(Boolean));

  const list = meta.champions.filter((c) =>
    displayName(c).toLowerCase().includes(q.trim().toLowerCase())
  );

  return (
    <div className="picker-backdrop" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="picker">
        <div className="picker-head">
          <span className={"ph-target " + target.side}>
            {target.side === "blue" ? "Blue" : "Red"} · {LANES[target.index]}
          </span>
          <div className="picker-search">
            <span className="ps-icon">⌕</span>
            <input
              ref={inputRef}
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder="Search 172 champions…"
              spellCheck={false}
            />
          </div>
          <button className="picker-close" onClick={onClose} aria-label="Close">×</button>
        </div>
        <div className="wall scroll-y">
          {list.length === 0 && <div className="wall-empty">No champion matches “{q}”.</div>}
          {list.map((c) => {
            const byBlue = takenBlue.has(c);
            const byRed = takenRed.has(c);
            const taken = byBlue || byRed;
            return (
              <ChampCell
                key={c} champKey={c} version={version}
                taken={taken} byClass={byBlue ? "by-blue" : byRed ? "by-red" : ""}
                takenBy={byBlue ? "Blue" : byRed ? "Red" : ""}
                onClick={() => { if (!taken) onPick(c); }}
              />
            );
          })}
        </div>
      </div>
    </div>
  );
}

function ChampCell({ champKey, version, taken, byClass, takenBy, onClick }) {
  const [err, setErr] = useState(false);
  const name = displayName(champKey);
  return (
    <div
      className={"champ-cell " + (taken ? "taken " + byClass : "")}
      onClick={onClick}
      title={name}
    >
      <div className="cc-portrait" data-takenby={takenBy}>
        {err
          ? <div className="cc-fallback">{name}</div>
          : <img src={iconUrl(version, champKey)} alt={name} loading="lazy" onError={() => setErr(true)} />}
      </div>
      <div className="cc-name">{name}</div>
    </div>
  );
}
