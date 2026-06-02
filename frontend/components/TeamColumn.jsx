"use client";
import { useState, useEffect } from "react";
import { displayName, iconUrl, LANES } from "@/lib/api";

function ChampSlot({ side, lane, champKey, active, onClick, version }) {
  const [err, setErr] = useState(false);
  useEffect(() => { setErr(false); }, [champKey]);
  const name = champKey ? displayName(champKey) : null;
  const url = champKey ? iconUrl(version, champKey) : null;

  return (
    <div
      className={"slot" + (side === "red" ? " red-align" : "") + (active ? " active" : "")}
      onClick={onClick}
      role="button"
      tabIndex={0}
    >
      <div className="lane-badge">{lane}</div>
      <div className={"portrait" + (champKey ? "" : " empty")}>
        {champKey
          ? (err
              ? <div className="cc-fallback">{name}</div>
              : <img src={url} alt={name} onError={() => setErr(true)} />)
          : <span>+</span>}
      </div>
      <div className="slot-meta">
        {champKey
          ? <div className="slot-champ" title={name}>{name}</div>
          : <div className="slot-champ empty">Click to pick</div>}
        <div className="slot-lane-label">{lane}</div>
      </div>
    </div>
  );
}

export function TeamColumn({ side, picks, activeIndex, onSlot, version }) {
  const filled = picks.filter(Boolean).length;
  return (
    <section className={"team-col " + side}>
      <div className="team-head">
        <div className="team-name">{side === "blue" ? "Blue Side" : "Red Side"}</div>
        <div className="team-side-tag">{filled}/5 locked</div>
      </div>
      <div className="slots">
        {LANES.map((lane, i) => (
          <ChampSlot
            key={lane}
            side={side}
            lane={lane}
            champKey={picks[i]}
            active={activeIndex === i}
            version={version}
            onClick={() => onSlot(side, i)}
          />
        ))}
      </div>
      <div className="team-foot">
        {side === "blue" ? "Lower number = first pick window" : "Mirror of blue draft order"}
      </div>
    </section>
  );
}
