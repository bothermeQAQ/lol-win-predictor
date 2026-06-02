/* =========================================================================
   PREVIEW COMPONENTS — broadcast Draft Lab UI.
   Mirrors the Next.js component tree 1:1 (ported in components/*.tsx).
   ========================================================================= */
const { useState, useEffect, useRef, useMemo, useCallback } = React;

const LANES = ["TOP", "JNG", "MID", "BOT", "SUP"];

function pct(x, d = 1) { return (x * 100).toFixed(d); }

/* ---------------------------------------------------------------- TopBar */
function TopBar({ meta, region, onRegion }) {
  return (
    <header className="topbar">
      <div className="brand">
        <div className="brand-mark" aria-hidden="true"></div>
        <div className="brand-text">
          <div className="brand-title">Draft Lab</div>
          <div className="brand-sub label">Blue-Side Win Model · Broadcast Analytics</div>
        </div>
      </div>

      <nav className="regions" aria-label="Region">
        {meta.regions.map((r) => (
          <button
            key={r.code}
            className={"region-tab" + (r.code === region ? " active" : "")}
            onClick={() => onRegion(r.code)}
          >
            <div className="rt-code">{r.code === "na1" ? "NA" : r.code === "kr" ? "KR" : "EUW"}</div>
            <div className="rt-base">baseline {pct(r.base_blue_win_rate)}% blue side</div>
          </button>
        ))}
      </nav>

      <div className="live-chip"><span className="live-dot"></span>Live Model</div>
    </header>
  );
}

/* ---------------------------------------------------------------- Slot + TeamColumn */
function ChampSlot({ side, lane, champKey, active, onClick, version }) {
  const [err, setErr] = useState(false);
  useEffect(() => { setErr(false); }, [champKey]);
  const name = champKey ? window.LolDisplayName(champKey) : null;
  const url = champKey ? window.LolDDragon.iconUrl(version, champKey) : null;

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

function TeamColumn({ side, picks, activeIndex, onSlot, version }) {
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

/* ---------------------------------------------------------------- WinCore */
function WinCore({ pred, ready, surge, shockwave }) {
  const prob = pred ? pred.blue_win_prob : 0.5;
  const regime = pred ? pred.which_model : "A";

  return (
    <section className="core">
      <div className="core-head">
        <div className={"model-tag regime-" + regime}>
          <span className="mt-dot"></span>
          {regime === "A" ? "Regime A · Draft-only" : "Regime B · Draft + map"}
        </div>
        <div className="label">{ready ? "AUC " + (regime === "A" ? "0.54" : "0.79") : "—"}</div>
      </div>

      <div className="core-prob">
        <div className="prob-side-label">Blue-side win probability</div>
        <div className={"prob-value" + (surge ? " surge" : "")}>
          {ready ? pct(prob) : "··.·"}<span className="pct">%</span>
        </div>
      </div>

      <div className="winbar-wrap">
        <div className="winbar">
          <div className="winbar-blue" style={{ width: (ready ? prob * 100 : 50) + "%" }}></div>
          <div className="winbar-mid"></div>
          <div className={"shockwave" + (shockwave ? " fire" : "")}></div>
        </div>
        <div className="winbar-readout">
          <span className="wr blue">BLUE {ready ? pct(prob) : "—"}%</span>
          <span className="wr red">RED {ready ? pct(1 - prob) : "—"}%</span>
        </div>
      </div>

      <CoreImpact pred={pred} ready={ready} />
    </section>
  );
}

function CoreImpact({ pred, ready }) {
  if (!ready || !pred) {
    return (
      <div className="impact">
        <div className="draft-impact">
          <div className="di-tier low">Awaiting draft</div>
          <div className="di-note">Fill all 10 champion slots to run the model.</div>
        </div>
      </div>
    );
  }
  const regime = pred.which_model;
  const delta = pred.delta_from_objectives;
  const baseline = pred.pre_game_baseline;

  if (regime === "A") {
    // DEAD WATER — show how little the draft moves off the regional baseline
    return (
      <div className="impact">
        <div className="impact-row">
          <span className="ir-label">Pre-game baseline</span>
          <span className="ir-val flat">{pct(baseline)}%</span>
        </div>
        <div className="draft-impact">
          <div className="di-tier low">Draft impact · LOW</div>
          <div className="di-note">
            Champion composition barely moves the model. Swap picks all you like — the
            draft-only engine stays near the regional baseline. Flip an early objective
            below to see the map take over.
          </div>
        </div>
      </div>
    );
  }

  // EARTHQUAKE — objectives moved the number off the draft baseline
  const sign = delta >= 0 ? "pos" : "neg";
  const tier = Math.abs(delta) >= 0.12 ? "high" : Math.abs(delta) >= 0.05 ? "medium" : "low";
  const word = tier === "high" ? "HIGH" : tier === "medium" ? "MODERATE" : "LOW";
  return (
    <div className="impact">
      <div className="impact-row">
        <span className="ir-label">Draft baseline</span>
        <span className="ir-val flat">{pct(baseline)}%</span>
      </div>
      <div className="impact-row">
        <span className="ir-label">Objective swing</span>
        <span className={"ir-val " + sign}>{delta >= 0 ? "+" : "−"}{pct(Math.abs(delta))}%</span>
      </div>
      <div className="draft-impact">
        <div className={"di-tier " + tier}>Map impact · {word}</div>
        <div className="di-note">
          Early objectives moved blue side by {delta >= 0 ? "+" : "−"}{pct(Math.abs(delta))} points
          off the draft baseline. This is the map talking, not the draft.
        </div>
      </div>
    </div>
  );
}

/* ---------------------------------------------------------------- MapConsole */
const OBJ_GLYPH = { first_blood: "✚", first_tower: "⌖", first_dragon: "◆", first_rift_herald: "◈" };
const OBJ_SUB = {
  first_blood: "First kill of the game",
  first_tower: "First turret destroyed",
  first_dragon: "First dragon secured",
  first_rift_herald: "First herald taken",
};

function MapConsole({ meta, objectives, onObjective }) {
  return (
    <section className="console">
      <div className="console-head">
        <div className="ch-left">
          <div className="panel-title">Map Events Console</div>
        </div>
        <div className="label">Early-game objectives · regime switch</div>
      </div>
      <div className="obj-grid">
        {meta.objectives.map((o) => {
          const val = objectives[o.key] || "none";
          return (
            <div className="obj-row" key={o.key}>
              <div className="obj-id">
                <div className="obj-glyph">{OBJ_GLYPH[o.key]}</div>
                <div>
                  <div className="obj-name">{o.label}</div>
                  <div className="obj-meta">{OBJ_SUB[o.key]}</div>
                </div>
              </div>
              <div className="tri" role="group" aria-label={o.label}>
                {["blue", "none", "red"].map((s) => (
                  <button
                    key={s}
                    className={(val === s ? "on " + s : "") }
                    onClick={() => onObjective(o.key, s)}
                  >
                    {s === "none" ? "—" : s}
                  </button>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

/* ---------------------------------------------------------------- ShapPanel */
const SHAP_SCALE = 1.3; // log-odds that maps to a full-width bar

function ShapPanel({ pred, ready }) {
  return (
    <section className="panel">
      <div className="panel-head">
        <div className="panel-title">Contribution Breakdown</div>
        <div className="label">SHAP · log-odds toward blue</div>
      </div>
      <div className="panel-body">
        {!ready || !pred ? (
          <div className="shap-empty">Run the model to see which factors push the prediction blue or red.</div>
        ) : (
          <div className="shap-list">
            {pred.top_factors.map((f, i) => {
              const w = Math.min(100, (Math.abs(f.shap_value) / SHAP_SCALE) * 100);
              const dir = f.direction; // blue | red
              return (
                <div className="shap-row" key={i}>
                  <div className="shap-top">
                    <span className="shap-label">{f.label}</span>
                    <span className={"shap-val " + dir}>
                      {f.shap_value >= 0 ? "+" : "−"}{Math.abs(f.shap_value).toFixed(2)}
                    </span>
                  </div>
                  <div className="shap-track">
                    <div className="axis"></div>
                    <div className={"shap-fill " + dir} style={{ width: (w / 2) + "%" }}></div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </section>
  );
}

/* ---------------------------------------------------------------- Banners */
function EngineBanner({ regime }) {
  return (
    <div className="engine-banner">
      <span className="eb-tag">{regime === "A" ? "Engine A" : "Engine B"}</span>
      <span className="eb-text">
        {regime === "A"
          ? <>Pre-game number is served by the <b>draft-only baseline</b> (AUC ≈ 0.54). It is deliberately flat — champions alone barely predict the winner.</>
          : <>You flipped an objective, so the readout switched engines to the <b>draft + early-objectives model</b> (AUC ≈ 0.79). The big jump is the new engine pricing in map state — not the draft.</>}
      </span>
    </div>
  );
}

function Guardrails() {
  return (
    <div className="guards">
      <div className="guard">
        <div className="g-mark">i</div>
        <div className="g-text">
          <b>Positions are just for arranging your draft.</b> The model only uses <b>which champions</b> are
          picked — not their lanes. Dragging Aatrox to MID changes nothing.
        </div>
      </div>
      <div className="guard">
        <div className="g-mark">i</div>
        <div className="g-text">
          <b>Two engines, one screen.</b> The pre-game % is a draft-only baseline; the moment you flip an early
          objective it switches to the model that includes map state — which is why that flip jumps so much.
        </div>
      </div>
    </div>
  );
}

/* ---------------------------------------------------------------- ChampPicker */
function ChampPicker({ open, target, meta, picks, version, onPick, onClose }) {
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
    window.LolDisplayName(c).toLowerCase().includes(q.trim().toLowerCase())
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
  const name = window.LolDisplayName(champKey);
  return (
    <div
      className={"champ-cell " + (taken ? "taken " + byClass : "")}
      onClick={onClick}
      title={name}
    >
      <div className="cc-portrait" data-takenby={takenBy}>
        {err
          ? <div className="cc-fallback">{name}</div>
          : <img src={window.LolDDragon.iconUrl(version, champKey)} alt={name} loading="lazy" onError={() => setErr(true)} />}
      </div>
      <div className="cc-name">{name}</div>
    </div>
  );
}

/* ---------------------------------------------------------------- Boot / Error */
function Boot({ msg }) {
  return (
    <div className="fullscreen">
      <div className="boot">
        <div className="brand-title boot-title">Draft Lab</div>
        <div className="boot-bars"><span></span><span></span><span></span><span></span><span></span></div>
        <div className="boot-msg">{msg || "Loading model metadata…"}</div>
      </div>
    </div>
  );
}

function ErrorScreen({ base, detail, onRetry }) {
  return (
    <div className="fullscreen">
      <div className="boot">
        <div className="brand-title boot-title">Backend unreachable</div>
        <div className="boot-msg err">
          Could not reach the prediction API. Make sure the FastAPI server is running and the URL is correct.
        </div>
        <div className="boot-code">GET {base}/meta<br />→ {detail}</div>
        <button className="boot-retry" onClick={onRetry}>Retry connection</button>
      </div>
    </div>
  );
}

/* export to window for main.jsx */
Object.assign(window, {
  TopBar, TeamColumn, WinCore, MapConsole, ShapPanel, EngineBanner,
  Guardrails, ChampPicker, Boot, ErrorScreen, LANES,
});
