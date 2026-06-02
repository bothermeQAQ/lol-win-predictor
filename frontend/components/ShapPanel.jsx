const SHAP_SCALE = 1.3; // log-odds that maps to a full-width bar

export function ShapPanel({ pred, ready }) {
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
              const dir = f.direction; // "blue" | "red"
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
