import { pct } from "@/lib/api";

export function WinCore({ pred, ready, surge, shockwave }) {
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
