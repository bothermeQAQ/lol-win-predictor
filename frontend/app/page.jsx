"use client";
import { useState, useEffect, useRef, useCallback } from "react";
import { API_BASE, fetchMeta, predict, ddragonVersion } from "@/lib/api";
import { TopBar } from "@/components/TopBar";
import { TeamColumn } from "@/components/TeamColumn";
import { WinCore } from "@/components/WinCore";
import { MapConsole } from "@/components/MapConsole";
import { ShapPanel } from "@/components/ShapPanel";
import { EngineBanner, Guardrails } from "@/components/Banners";
import { ChampPicker } from "@/components/ChampPicker";
import { Boot, ErrorScreen, PredictError } from "@/components/Status";

export default function Page() {
  const [meta, setMeta] = useState(null);
  const [version, setVersion] = useState("15.10.1");
  const [error, setError] = useState(null);

  const [region, setRegion] = useState("na1");
  const [picks, setPicks] = useState({ blue: [null, null, null, null, null], red: [null, null, null, null, null] });
  const [objectives, setObjectives] = useState({
    first_blood: "none", first_tower: "none", first_dragon: "none", first_rift_herald: "none",
  });

  const [pred, setPred] = useState(null);
  const [predError, setPredError] = useState(null);
  const [picker, setPicker] = useState({ open: false, side: "blue", index: 0 });

  // animation flags
  const [surge, setSurge] = useState(false);
  const [shock, setShock] = useState(false);
  const [quake, setQuake] = useState(false);
  const causeRef = useRef("draft");
  const prevRegimeRef = useRef("A");

  /* ---- boot: meta + ddragon version ---- */
  const boot = useCallback(async () => {
    setError(null); setMeta(null);
    try {
      const [m, v] = await Promise.all([fetchMeta(), ddragonVersion()]);
      setVersion(v);
      setPicks({ blue: m.champions.slice(0, 5), red: m.champions.slice(5, 10) });
      setRegion(m.regions[0].code);
      setMeta(m);
    } catch (e) {
      setError(String(e && e.message ? e.message : e));
    }
  }, []);
  useEffect(() => { boot(); }, [boot]);

  /* ---- predict whenever inputs change & both teams full ---- */
  const full = !!meta && picks.blue.every(Boolean) && picks.red.every(Boolean);
  const picksKey = JSON.stringify(picks);
  const objKey = JSON.stringify(objectives);

  useEffect(() => {
    if (!meta) return;
    if (!full) { setPred(null); return; }
    let cancelled = false;
    const body = { region, blue: picks.blue, red: picks.red, bans: [], objectives };
    predict(body)
      .then((res) => {
        if (cancelled) return;
        setPredError(null);
        const cause = causeRef.current;
        causeRef.current = "draft";
        applyResult(res, cause);
      })
      .catch((e) => { if (!cancelled) setPredError(String(e && e.message ? e.message : e)); });
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [meta, region, picksKey, objKey, full]);

  function applyResult(res, cause) {
    const prevRegime = prevRegimeRef.current;
    prevRegimeRef.current = res.which_model;
    setPred(res);

    const engineSwitched = prevRegime !== res.which_model;
    const big = cause === "objective" || engineSwitched;
    const mag = Math.abs(res.delta_from_objectives);

    if (big) {
      setSurge(true); setShock(true);
      if (mag > 0.04) setQuake(true);
      setTimeout(() => setSurge(false), 240);
      setTimeout(() => setShock(false), 720);
      setTimeout(() => setQuake(false), 460);
    } else if (cause === "draft" || cause === "region") {
      setSurge(true);
      setTimeout(() => setSurge(false), 160);
    }
  }

  /* ---- handlers ---- */
  const onRegion = (code) => { causeRef.current = "region"; setRegion(code); };
  const onObjective = (key, side) => {
    causeRef.current = "objective";
    setObjectives((o) => ({ ...o, [key]: o[key] === side ? "none" : side }));
  };
  const onSlot = (side, index) => setPicker({ open: true, side, index });
  const onPick = (champ) => {
    causeRef.current = "draft";
    setPicks((p) => {
      const next = { blue: [...p.blue], red: [...p.red] };
      next[picker.side][picker.index] = champ;
      return next;
    });
    setPicker((pk) => ({ ...pk, open: false }));
  };
  const onClosePicker = () => setPicker((pk) => ({ ...pk, open: false }));

  /* ---- render ---- */
  if (error) return <ErrorScreen base={API_BASE} detail={error} onRetry={boot} />;
  if (!meta) return <Boot msg="Loading model metadata…" />;

  const activeIndex = (side) => (picker.open && picker.side === side ? picker.index : -1);

  return (
    <div className={"app" + (quake ? " quake" : "")}>
      <TopBar meta={meta} region={region} onRegion={onRegion} />

      <div className="cockpit">
        <TeamColumn side="blue" picks={picks.blue} activeIndex={activeIndex("blue")} version={version} onSlot={onSlot} />
        <WinCore pred={pred} ready={full} surge={surge} shockwave={shock} />
        <TeamColumn side="red" picks={picks.red} activeIndex={activeIndex("red")} version={version} onSlot={onSlot} />
      </div>

      <PredictError detail={predError} />
      {full && pred && <EngineBanner regime={pred.which_model} />}

      <div className="section-grid">
        <MapConsole meta={meta} objectives={objectives} onObjective={onObjective} />
        <ShapPanel pred={pred} ready={full} />
      </div>

      <Guardrails />

      <ChampPicker
        open={picker.open}
        target={{ side: picker.side, index: picker.index }}
        meta={meta}
        picks={picks}
        version={version}
        onPick={onPick}
        onClose={onClosePicker}
      />
    </div>
  );
}
