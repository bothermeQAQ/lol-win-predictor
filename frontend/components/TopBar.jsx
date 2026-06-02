import { pct } from "@/lib/api";

export function TopBar({ meta, region, onRegion }) {
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
