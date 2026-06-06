# Quick Start / How to Run

**What this is.** A binary classifier for **high-elo League of Legends** that predicts whether the **blue side wins**, with an interactive demo. Headline finding: the champion *draft* is near-uninformative in high elo (AUC ≈ 0.54), while **early objectives** recover a strong, region- and patch-invariant signal (AUC ≈ 0.79).

**🌐 Live demo:** **https://lol-win-predictor-web.onrender.com**
> ⏳ The **first** load can take **~30–60 s** while the backend cold-starts (free tier sleeps when idle) — that's expected, not a bug.

**▶️ Run the backend locally** (from the project root):
```bash
PORT=8008 PYTHONPATH=src python3 src/api/server.py
```

**▶️ Run the frontend locally** (in a second terminal):
```bash
cd frontend && npm install && npm run dev
```
`frontend/.env.local` already points at `http://localhost:8008`. Open the URL the dev server prints (default http://localhost:3000).

**📄 Paper / write-up:** [`paper/DSC148_paper.pdf`](paper/DSC148_paper.pdf).

---

# DSC148 — League of Legends 胜负预测(蓝 vs 红)

二分类:给定一局对局的**赛前 / 早期**信息,预测蓝方(team1 / teamId 100)是否获胜。
数据 `data/raw_matches.csv` 为 Riot Match-V5 真实数据(na1 / kr / euw1,补丁 16.9–16.11,约 5.1 万局)。

**泄漏纪律(单一来源 `src/data_utils.py`):**
- 赛制 A(draft-only):10 个英雄 pick + 召唤师技能 + 10 个 ban。
- 赛制 B(draft + early objectives):A + 4 个首目标旗标(一血 / 一塔 / 首龙 / 首先锋)。
- 永久封禁(赛后状态):各类击杀数、首抑制 / 首男爵、游戏时长等。

**主结论:** 纯 BP(赛制 A)几乎不可预测(AUC≈0.54);加入早期目标(赛制 B)后跳到 AUC≈0.79,
跨地区、跨补丁高度稳定 —— 决定胜负的是早期目标,而非 BP 本身。

---

## 环境

```bash
pip install -r requirements.txt          # Homebrew Python 用户加 --user --break-system-packages
```
主要依赖:lightgbm / scikit-learn / torch(建模),streamlit / shap(demo)。

## 复现各检查点(可选)

```bash
PYTHONPATH=src python3 src/baselines.py          # ckpt2 基线(LogReg / NB)
PYTHONPATH=src python3 src/proposed.py           # ckpt3 LightGBM + ChampPoolNet
PYTHONPATH=src python3 src/ablations.py          # ckpt4 消融 / 显著性 / 迁移 / 补丁 / 超参 / 案例
```
结果写入 `models/`(`ckpt3_table.txt`、`ckpt4_summary.txt` 等)。

---

## Demo(检查点 5)

服务 **pooled regime-B LightGBM(+region)** 作为主模型;另存一个 **draft-only regime-A 模型**专门给赛前基准用。
预测 / 编码 / SHAP 全部从磁盘加载,**启动不重训**。

> **为什么是两个模型?** 「四目标全无」这一赛前状态在训练数据里占 **0.000%**(一血 99.98% 的对局会分出),
> 对 regime-B 属分布外(OOD)外推,会让它对换英雄虚假敏感、与「纯 BP 不可预测」的主结论矛盾。
> 所以 demo 在全『无』时用诚实的 regime-A(genuinely flat,AUC≈0.54)出基准,任一目标打开就切回 regime-B(AUC≈0.79)。

### 1) 训练并存盘模型(只需一次)

```bash
PYTHONPATH=src python3 src/train_demo_model.py
```
产物写入 `models/demo/`:`lgbm_A.joblib` / `lgbm_B.joblib`、`explainer_A.joblib` / `explainer_B.joblib`(SHAP)、`vocabs.json`、`meta.json`。
脚本会打印两个测试 AUC(regime A≈0.54 draft-only 基准、regime B≈0.79 主模型),确认 demo 模型与正文模型一致。

### 2) 启动 demo

```bash
python3 -m streamlit run src/demo_app.py
```
浏览器打开终端给出的地址(默认 http://localhost:8501)。

### 3) TA 怎么试(30 秒)

1. 选 **地区**(na1 / kr / euw1)。
2. 上来所有**目标开关 = 无** → 这是**纯赛前 / draft-only 预测**。
   随便**换几个英雄**,会发现蓝方胜率**几乎不动**、贴近该地区基准(≈48–54%)。
   ⚠️ 这是真实发现(纯 BP 不可预测),**不是 demo 坏了** —— 界面上有黄色提示说明。
3. 现在把某个目标(例如**一塔 → 蓝**)打开 → 蓝方胜率**明显跳动**,
   "早期目标带来的变化"指标会显示移动了多少个百分点。这正是 regime B 的主秀。
4. 下方 **Top 贡献因子(SHAP)** 实时显示哪些因素在推高蓝 / 红方;
   纯赛前状态几乎只有"地区基准"在起作用,打开目标后目标项立刻占据主导。

**护栏:** 同一英雄在两方/同方重复选会报错并提示修正;英雄 / ban 留空有默认值不会崩。

### 自测(可选)

```bash
PYTHONPATH=src python3 src/demo_selftest.py
```
无需 web server,直接验证:换英雄胜率几乎不动、翻目标开关胜率明显且方向正确、SHAP 符号正确、去重校验生效。

---

`data/`、`checkpoints/` 已 gitignore(不提交原始数据);`models/`(含 demo 产物)会提交。
