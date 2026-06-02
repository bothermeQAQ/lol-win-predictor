/* =========================================================================
   PREVIEW MOCK — stands in for the FastAPI backend so the design can be
   rendered & verified without a running server.
   The real Next.js app fetches the SAME shapes from NEXT_PUBLIC_API_BASE_URL.
   Champion portraits load from the real Data Dragon CDN (works in-browser).
   ========================================================================= */
(function () {
  // ---- exact champ_list from models/demo/meta.json (172 entries) ----
  const CHAMPS = ["Aatrox","Ahri","Akali","Akshan","Alistar","Ambessa","Amumu","Anivia","Annie","Aphelios","Ashe","AurelionSol","Aurora","Azir","Bard","Belveth","Blitzcrank","Brand","Braum","Briar","Caitlyn","Camille","Cassiopeia","Chogath","Corki","Darius","Diana","DrMundo","Draven","Ekko","Elise","Evelynn","Ezreal","FiddleSticks","Fiora","Fizz","Galio","Gangplank","Garen","Gnar","Gragas","Graves","Gwen","Hecarim","Heimerdinger","Hwei","Illaoi","Irelia","Ivern","Janna","JarvanIV","Jax","Jayce","Jhin","Jinx","KSante","Kaisa","Kalista","Karma","Karthus","Kassadin","Katarina","Kayle","Kayn","Kennen","Khazix","Kindred","Kled","KogMaw","Leblanc","LeeSin","Leona","Lillia","Lissandra","Lucian","Lulu","Lux","Malphite","Malzahar","Maokai","MasterYi","Mel","Milio","MissFortune","MonkeyKing","Mordekaiser","Morgana","Naafiri","Nami","Nasus","Nautilus","Neeko","Nidalee","Nilah","Nocturne","Nunu","Olaf","Orianna","Ornn","Pantheon","Poppy","Pyke","Qiyana","Quinn","Rakan","Rammus","RekSai","Rell","Renata","Renekton","Rengar","Riven","Rumble","Ryze","Samira","Sejuani","Senna","Seraphine","Sett","Shaco","Shen","Shyvana","Singed","Sion","Sivir","Skarner","Smolder","Sona","Soraka","Swain","Sylas","Syndra","TahmKench","Taliyah","Talon","Taric","Teemo","Thresh","Tristana","Trundle","Tryndamere","TwistedFate","Twitch","Udyr","Urgot","Varus","Vayne","Veigar","Velkoz","Vex","Vi","Viego","Viktor","Vladimir","Volibear","Warwick","Xayah","Xerath","XinZhao","Yasuo","Yone","Yorick","Yunara","Yuumi","Zaahen","Zac","Zed","Zeri","Ziggs","Zilean","Zoe","Zyra"];

  const REGIONS = [
    { code: "na1",  label: "North America (NA)", base_blue_win_rate: 0.45014412612506616 },
    { code: "kr",   label: "Korea (KR)",         base_blue_win_rate: 0.49070588235294116 },
    { code: "euw1", label: "Europe West (EUW)",  base_blue_win_rate: 0.4324705882352941 },
  ];
  const OBJECTIVES = [
    { key: "first_blood",        label: "First Blood",        model_field: "firstBlood" },
    { key: "first_tower",        label: "First Tower",        model_field: "firstTower" },
    { key: "first_dragon",       label: "First Dragon",       model_field: "firstDragon" },
    { key: "first_rift_herald",  label: "First Rift Herald",  model_field: "firstRiftHerald" },
  ];

  const META = {
    regions: REGIONS,
    team_size: 5,
    champions: CHAMPS,
    bans: CHAMPS.slice(),
    objectives: OBJECTIVES,
    objective_values: ["none", "blue", "red"],
    models: {
      A: { role: "draft-only pre-game baseline (served when all objectives = none)", test_auc: 0.54 },
      B: { role: "headline model: draft + early objectives (served when any objective set)", test_auc: 0.79 },
    },
    overall_base_blue_win_rate: 0.45777368183689876,
  };

  const REGION_BASE = Object.fromEntries(REGIONS.map(r => [r.code, r.base_blue_win_rate]));

  // ---- mock prediction maths (mirrors the demo's "two-engine" behaviour) ----
  function hash01(s) {
    let h = 2166136261;
    for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
    return (h >>> 0) / 4294967295;
  }
  // champion composition barely moves regime A: ~±0.0125 log-odds each
  const champLogit = (name) => (hash01(name) - 0.5) * 0.025;
  // early objectives are the earthquake — First Tower dominates
  const OBJ_LOGIT = { first_blood: 0.46, first_tower: 1.15, first_dragon: 0.34, first_rift_herald: 0.28 };
  const logit = (p) => Math.log(p / (1 - p));
  const sigm  = (x) => 1 / (1 + Math.exp(-x));

  function objLabel(key) {
    return { first_blood: "First Blood", first_tower: "First Tower",
             first_dragon: "First Dragon", first_rift_herald: "First Rift Herald" }[key];
  }

  function mockPredict(body) {
    const { region, blue, red, objectives } = body;
    const base = REGION_BASE[region] ?? META.overall_base_blue_win_rate;

    // draft-only baseline (regime A) — genuinely flat
    let draftL = logit(base);
    const champFactors = [];
    blue.forEach((c) => { const v = champLogit(c);  draftL += v; champFactors.push(["blue_pick=" + c, v]); });
    red.forEach((c)  => { const v = -champLogit(c); draftL += v; champFactors.push(["red_pick=" + c, v]); });
    const probDraft = sigm(draftL);

    const anyObj = OBJECTIVES.some(o => (objectives[o.key] || "none") !== "none");

    let prob, regime, factors;
    if (!anyObj) {
      regime = "A";
      prob = probDraft;
      // regime-A factors: regional baseline + champion picks (all tiny)
      factors = [["region=" + region, logit(base) - logit(0.5)], ...champFactors];
    } else {
      regime = "B";
      let objL = 0;
      const objFactors = [];
      OBJECTIVES.forEach((o) => {
        const side = objectives[o.key] || "none";
        if (side === "none") return;
        const w = OBJ_LOGIT[o.key] * (side === "blue" ? 1 : -1);
        objL += w;
        objFactors.push([`${o.model_field}_${side}`, w]);
      });
      prob = sigm(logit(probDraft) + objL);
      // regime-B factors: objectives dominate, plus a couple champion picks
      factors = [...objFactors, ...champFactors];
    }

    // top-6 by |shap|
    factors.sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
    const top = factors.slice(0, 6).filter(([, v]) => Math.abs(v) > 1e-6);

    const top_factors = top.map(([name, val]) => ({
      feature: name,
      label: featureLabel(name),
      shap_value: val,
      direction: val > 0 ? "blue" : "red",
    }));

    return {
      blue_win_prob: prob,
      which_model: regime,
      pre_game_baseline: probDraft,
      delta_from_objectives: prob - probDraft,
      top_factors,
    };
  }

  function featureLabel(name) {
    if (name.startsWith("region=")) return "Regional baseline · " + name.split("=")[1].toUpperCase();
    const m = name.match(/^(firstBlood|firstTower|firstDragon|firstRiftHerald)_(blue|red)$/);
    if (m) {
      const lab = { firstBlood: "First Blood", firstTower: "First Tower",
                    firstDragon: "First Dragon", firstRiftHerald: "First Rift Herald" }[m[1]];
      return (m[2] === "blue" ? "Blue took " : "Red took ") + lab;
    }
    if (name.startsWith("blue_pick=")) return "Blue picks " + window.LolDisplayName(name.split("=")[1]);
    if (name.startsWith("red_pick="))  return "Red picks "  + window.LolDisplayName(name.split("=")[1]);
    return name;
  }

  // ---- mock fetch layer (latency to feel real) ----
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  window.MockAPI = {
    async meta() { await wait(420); return META; },
    async predict(body) { await wait(160); return mockPredict(body); },
  };

  // ======================================================================
  //  DATA DRAGON helpers (shared by preview + real app)
  // ======================================================================
  // champ_list value -> Data Dragon image key (handle the few mismatches)
  const DDRAGON_ALIAS = { FiddleSticks: "Fiddlesticks", Renata: "RenataGlasc" };

  const SPECIAL_NAME = {
    AurelionSol: "Aurelion Sol", MonkeyKing: "Wukong", DrMundo: "Dr. Mundo",
    MissFortune: "Miss Fortune", KSante: "K'Sante", Belveth: "Bel'Veth",
    Chogath: "Cho'Gath", KogMaw: "Kog'Maw", RekSai: "Rek'Sai", TahmKench: "Tahm Kench",
    XinZhao: "Xin Zhao", JarvanIV: "Jarvan IV", LeeSin: "Lee Sin", MasterYi: "Master Yi",
    TwistedFate: "Twisted Fate", Nunu: "Nunu & Willump", Leblanc: "LeBlanc",
    Velkoz: "Vel'Koz", Khazix: "Kha'Zix", FiddleSticks: "Fiddlesticks", Renata: "Renata Glasc",
  };
  window.LolDisplayName = function (key) {
    if (SPECIAL_NAME[key]) return SPECIAL_NAME[key];
    return key.replace(/([a-z])([A-Z])/g, "$1 $2");
  };

  window.LolDDragon = {
    async version() {
      try {
        const r = await fetch("https://ddragon.leagueoflegends.com/api/versions.json");
        const v = await r.json();
        return v[0];
      } catch (e) { return "15.10.1"; } // safe recent fallback
    },
    iconUrl(version, key) {
      const k = DDRAGON_ALIAS[key] || key;
      return `https://ddragon.leagueoflegends.com/cdn/${version}/img/champion/${k}.png`;
    },
  };
})();
