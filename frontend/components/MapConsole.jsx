const OBJ_GLYPH = { first_blood: "✚", first_tower: "⌖", first_dragon: "◆", first_rift_herald: "◈" };
const OBJ_SUB = {
  first_blood: "First kill of the game",
  first_tower: "First turret destroyed",
  first_dragon: "First dragon secured",
  first_rift_herald: "First herald taken",
};

export function MapConsole({ meta, objectives, onObjective }) {
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
                    className={val === s ? "on " + s : ""}
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
