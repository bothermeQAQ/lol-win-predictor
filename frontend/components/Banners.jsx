export function EngineBanner({ regime }) {
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

export function Guardrails() {
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
