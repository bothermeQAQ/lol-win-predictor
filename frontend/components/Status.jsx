export function Boot({ msg }) {
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

export function ErrorScreen({ base, detail, onRetry }) {
  return (
    <div className="fullscreen">
      <div className="boot">
        <div className="brand-title boot-title">Backend unreachable</div>
        <div className="boot-msg err">
          Could not reach the prediction API. Make sure the FastAPI server is running and
          NEXT_PUBLIC_API_BASE_URL points at it.
        </div>
        <div className="boot-code">GET {base}/meta<br />→ {detail}</div>
        <button className="boot-retry" onClick={onRetry}>Retry connection</button>
      </div>
    </div>
  );
}

export function PredictError({ detail }) {
  if (!detail) return null;
  return (
    <div className="engine-banner" style={{ borderColor: "var(--red-deep)", background: "var(--red-wash)" }}>
      <span className="eb-tag" style={{ color: "var(--red-soft)", borderColor: "var(--red-deep)" }}>API</span>
      <span className="eb-text">Prediction request failed: {detail}</span>
    </div>
  );
}
