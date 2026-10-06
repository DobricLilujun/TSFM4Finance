# Plan — 统一可复现的 standard-bench 时间序列评测基准

> 把 assets/data/standard-bench/ 现有数据整理成统一、可复现的时间序列评测基准；
> 小时/日频任务分开；按时间切分（train 2013-2018 / val 2019-2020 / test 2021-2023）；
> 保留原始频率/时间戳/字段，补充统一元数据（target/features/frequency/lookback/horizon）；
> 宏观/基本面特征只在实际发布之后可用（防未来泄漏）；粒度最大日、最小分钟（若有）；
> 训练先不测，直接在 validation set 上评估（资源允许时再试训练）。

---

## 0. 决策（已确认）

- 全部按推荐；**训练有资源就也试**（val 为必需主评估，train 训练/微调为可选）。
- **评估与训练统一用 `amazon/chronos-bolt-base` 与 `google/timesfm-2.0-500m-pytorch`**（均已在 HF 核实可下载，见 §9）。
- 各项默认：Q1=(a) 仅日+小时、标注分钟不可用；Q2=价格水平目标（FX `close`/FI `yield_pct`/Equity `split_adjusted_close`）；
  Q3=日 lookback 52 / horizon 10，小时 lookback 96 / horizon 24；Q4=档 0 price-only 起步（档 1 可选）；
  Q5=每序列一个 TSDataset 按 domain/frequency 聚合；Q6=bases+chronos-bolt+timesfm；Q7=全 universe；Q8=新 `standard/` 桶。

---

## 1. Context（为什么做）

standard-bench/ 下已有三大领域数据（equity / fixedincome / fx），但尚未接入统一评测库 tsfm_eval：
- tsfm_eval 的注册器只扫描 assets/data/open/ 与 assets/data/closed/ 下 {name}/meta.json + train/validation/test.parquet。
- standard-bench/ 用的是另一套布局（每个子域一个文件夹 + *.csv.gz + source_manifest.json + validation.json）。
- 现有 open/ 数据集（sp500_daily 等）都是单序列，TSDataset.iter_rolling 直接在扁平 DataFrame 上滚动；
  而 standard-bench 的 FX（20 对）、equity（50 只）、FI（4 国 x 多期限）是面板，扁平滚动会跨序列错误拼接。

需要：(1) 盘点数据、核实覆盖；(2) 把 standard-bench 桥接成统一 TSDataset（含面板处理）；
(3) 加防泄漏的宏观/基本面特征；(4) 在 val 上评估（+ 可选训练）；(5) 汇报数据集/切分/缺失/待确认问题。

---

## 2. 数据盘点与覆盖核实（已用 Python 实测，未伪造/未补齐）

切分已统一为 train 2013-2018 / validation 2019-2020 / test 2021-2023（三域一致）。

### 2.1 FX（fx_data/）
- 日频 fx_1d_{train,validation,test}.csv.gz：列 open,close,low,high,source_volume,timestamp,pair,frequency,source,price_side,date,split,category,flat_bar。
  - train 36,814 / 21 对 / 2013-01-01..2018-12-31；**validation 12,283 / 20 对 / 2019-01-01..2020-12-31**（比 train 少 1 对）；test 18,299 / 20 对 / 2021-01-01..2023-12-29。
- 小时频 fx_1h_{train,validation,test}.csv.gz：9 个主要/高流动性对（AUDUSD, EURAUD, EURCHF, EURUSD, GBPUSD, NZDUSD, USDCAD, USDCHF, USDJPY）。
  - **validation 112,283 / 9 对 / 2019-01-01..2020-12-31**。
- 宏观/政策利率（防泄漏，档 1 用）：policy_differentials_*（含 lag1 代理）、rate_differentials_monthly_proxy（JPY 用 call-money 代理）、fred_supplement_daily（SOFR 起 2018-04、DFF、DEXINUS）。
- 已知缺失/注意：日频 val/test 只有 20 对（train 21 对），**有 1 对在 val/test 缺失**；JPY 2013-04-04..2016-09-20 无政策利率（不能补齐）；
  利差 carry 最多 14 天、1 日 lag 是假设、BIS/厂商历史是当前重建非 point-in-time；missing_weekday_grid / rate_coverage / excluded_candles 记录缺失与剔除。

### 2.2 Fixed income（fixedincome_data/）
- 日频 yields_{train,validation,test}.csv.gz（长格式）：列 date,yield_pct,country,maturity_years,source,source_series,yield_definition,split。
  - train 33,015 / **validation 10,987** / test 16,530；country in {DE,JP,UK,US}；公共期限 2/5/7/10/20（四国一致），US 另有 3、30。
- 宏观（US only，档 1 用）：macro_monthly_lagged_proxy.csv（132 行；FEDFUNDS/CPI 通胀/实际 GDP 增长/UNRATE，含 *_reference_period_end 与 *_assumed_available_month_end）。
  - 注意：macro_monthly_reference_aligned 含参考月未结束前的值、且跨季度重复 GDP，**不能**作同期特征；宏观是 FRED 当前修订值非 point-in-time；lag 是假设。
- 缺失：各源市场日历不同，缺失/节假日未补齐（coverage.csv 记录）；宏观仅 US（按要求 4 指标）。

### 2.3 Equity（equity_data/）
- 日频 daily_ohlcv_2013..2023.csv.gz（11 文件）：列 permno,gvkey,iid,date,open,high,low,close,volume,ajexdi,trfd,curcdd,split_adjusted_close,total_return_adjusted_close,in_sp500,split；50 只（2012-12 固定 cohort）。
  - train 75,500 / **validation 25,250** / test 37,650；date 2013-01-02..2023-12-29。
- 目标与缺失：total_return_adjusted_close / trfd 缺 **5,536 行**（仅 permno 83443、84788）；**split_adjusted_close 与 raw close 无缺失**（故默认目标取 split_adjusted_close）。
- 月度特征（档 1 用，防泄漏）：monthly_characteristics_with_warmup.csv.gz（security_market_cap_usd, book_to_market_lagged_proxy, momentum_12_2_next_month, ...，含 assumed_available_date）。
  - 月度到日频必须用“已完成的上月”；B/M 缺失 303 行（2013-2023）；B/M 用 6 月会计 lag（假设，非验证）；Compustat 会重述。
- 其他：gics_history（历史有效区间，非 as-known 版本）、annual_fundamentals_with_lag、distributions（2,022）、delistings（0 行）、ccm_link_history。

### 2.4 粒度（“最大日、最小分钟若有”）—— 关键发现
- 全仓检索**没有任何 minute / second / tick / intraday 数据**；最细为**小时（仅 FX 9 对）**，日频三域都有。
- -> “最小分钟”无法满足（见 §11 待确认/风险）。

---

## 3. 架构决策（已确认）

### 3.1 桥接方式（A）
新增 scripts/build_standard_bench.py，把 standard-bench 的 *.csv.gz 组装成统一 meta.json + train/validation/test.parquet，
写入新桶 **assets/data/standard/**；registry.py 增加对 standard/ 的扫描（或新增专用 loader）。复用 TSDataset / iter_rolling / evaluate /
run_benchmark / BaseModel。可复现、不污染现有 open/ 合成数据集。

### 3.2 面板处理（A，已确认）
每序列生成一个 TSDataset（FX 日 20、FX 时 9、FI 4 国 x 期限、equity 50），每个为“timestamp + 少量特征 + target”单/少变量序列；
run_benchmark 汇总多数据集，按 domain/frequency 聚合（与 BIS by_currency/by_horizon 一致）。不改动 iter_rolling 核心语义。

### 3.3 防泄漏特征策略（分档，已确认）
- 档 0（默认，最干净）：只用价格/OHLCV 类特征，不引入宏观/基本面（先跑通 val 评估，零泄漏风险）。
- 档 1（可选增强）：引入已 lag 的宏观/基本面代理（*_lagged_proxy、policy_differentials 的 lag1、月度特征用“已完成上月”），
  显式标注“假设可用、非验证发布日期”，并在 meta/报告写清假设。
- 绝不使用：macro_monthly_reference_aligned、WRDS latest-revision、同日 contemporaneous 政策利差、同日收盘特征预测当日收益。
- 拟合缩放/归一只在 train 上做（即便本次不训练，scaler 也限定在 train 子集）。

---

## 4. 统一基准目录（每序列一个 dataset，全 universe）

| 域 | 频率 | 目标 | 特征(档0) | lookback | horizon | 序列数(val) | val 行数 |
|---|---|---|---|---|---|---|---|
| fx | hour | close | open,high,low,close | 96 | 24 | 9 | 112,283 |
| fx | day | close | open,high,low,close[,source_volume] | 52 | 10 | 20 | 12,283 |
| fi | day | yield_pct | 其余期限 yield（同国曲线） | 52 | 10 | 4 国 x 5 期限(+US 3/30) | 10,987 |
| equity | day | split_adjusted_close | open,high,low,close,volume | 52 | 10 | 50 | 25,250 |

- 保留原始字段（timestamp/频率/源字段）+ 统一 meta；task=forecast（默认）。档 1 时追加已-lag 宏观/基本面列。
- FI 曲线：公共 2/5/7/10/20 四国；US 3y、30y 作为 US 全曲线额外序列（与公共面板分开）。

## 5. Reuse（已有可复用）
- 统一接口/元数据：src/tsfm_eval/tsfm_eval/schemas.py（DatasetMeta、Frequency、Domain、TaskType）。
- 数据接口：data/dataset.py（TSDataset、iter_rolling、split）、data/loader.py、data/registry.py。
- 评估：evaluation/benchmark.py::run_benchmark(..., split="validation")（已支持 split=）、evaluation/evaluator.py::evaluate。
- 指标/综合分：metrics/forecast.py、metrics/score.py（composite_score）。
- 模型接口：models.base.BaseModel / PredictionResult；现有 tsfm4finance/core/models（naive/ma/seasonal_naive/random_walk/arima/chronos/timesfm）。
- 面板评测范式：assets/data/bis/（instances.parquet 的 (lookback,horizon,origin,context,target) + by_currency/by_horizon 结果）。
- 各域 README 的防泄漏规则与覆盖说明。

## 6. Files to modify / add
- 新增 scripts/build_standard_bench.py：读 csv.gz -> 组装每序列 train/val/test.parquet + DatasetMeta。
- 新增 assets/data/standard/（每序列 meta.json + train/val/test.parquet）。
- 扩展 src/tsfm_eval/tsfm_eval/data/registry.py（与 loader.py）：识别 standard/ 桶。
- 新增/适配模型适配器：src/tsfm_eval/tsfm_eval/models/ 下加 ChronosBoltModel（amazon/chronos-bolt-base）与
  TimesFM20PytorchModel（google/timesfm-2.0-500m-pytorch）；适配现有 tsfm4finance/core/models 的加载逻辑。
- 新增 scripts/eval_standard_bench.py：在 split="validation" 上跑模型 -> val 指标/综合分；（可选）scripts/train_standard_bench.py 在 train 上微调。
- 新增 docs/standard_bench_report.md：数据集/切分/覆盖/缺失/待确认。
- 安装依赖：uv pip install torch transformers chronos-forecasting timesfm pmdarima（见 §9）。

## 7. 评估/训练流程
- 主：val 评估。对每个 dataset 在 split="validation" 上滚动（iter_rolling / run_benchmark），模型 = {naive, moving_average, seasonal_naive,
  random_walk, arima, chronos-bolt-base, timesfm-2.0-500m-pytorch}，汇总按 domain/frequency，输出 val 指标 + composite score。
- 可选（资源允许）：在 train 上微调 chronos-bolt / timesfm（fit(train_df, meta)），再在 val 上评估，对比 zero-shot vs fine-tuned（沿用 BIS by_horizon 口径）。
- 不训练时：scaler/归一仅在 train 子集拟合后应用到 val，保证可复现且无泄漏。

## 8. Steps（执行清单）
- [ ] 装依赖（torch/transformers/chronos-forecasting/timesfm/pmdarima），确认两个模型权重可下载（§9 已预验证）。
- [ ] 实现 build_standard_bench.py -> 生成 standard/ 下每序列 meta.json + train/val/test.parquet（保留原字段+统一 meta）。
- [ ] registry/loader 支持 standard/ 桶。
- [ ] 适配 ChronosBolt / TimesFM2.0-Pytorch 加载器；加 bases 与 arima。
- [ ] eval_standard_bench.py：run_benchmark(models, datasets, split="validation") -> val 指标/综合分（按 domain/frequency 聚合）。
- [ ] 跑 val 评估；（资源允许）train 微调后复评。
- [ ] 产出 docs/standard_bench_report.md（含 §11 待确认/风险）。

## 9. 模型与网络可行性（已核实）
- 两模型在 HF 均存在且权重可下载（huggingface.co 可达，权重经 us.aws.cdn.hf.co/xet-bridge-us 返回 HTTP 200）：
  - amazon/chronos-bolt-base：model.safetensors ~821 MB（tags: chronos-forecasting, safetensors, t5）。
  - google/timesfm-2.0-500m-pytorch：model.safetensors ~1.99 GB（+ torch_model.ckpt；tags: timesfm, safetensors）。
- 网络细节：huggingface.co、pypi.org 可达；**cdn-lfs.huggingface.co / resolver1 / mirror 的 DNS 解析失败**，但 HF 现已经
  xet/bridge CDN（us.aws.cdn.hf.co）提供权重，实测 200，故不受影响。当前 venv 未装 torch/transformers/chronos/timesfm/pmdarima，需安装。
- 风险：若个别网络抖动导致 xet CDN 不可达，可设 HF_ENDPOINT / 镜像或改用本地已下载 ckpt（assets/data/bis/ckpts 有 chronos2 先例）。

## 10. Verification
- 加载每个新 TSDataset，断言 train/val/test 日期边界 = 2013-2018 / 2019-2020 / 2021-2023。
- 断言 val 行数与 validation.json/manifest.json 一致（fx 12,283 / fi 10,987 / equity 25,250）。
- 面板：断言每窗口不跨 series id（方案 A 天然满足）。
- 防泄漏：断言无 reference-aligned / WRDS latest / 同日 contemporaneous 特征进入任何 split；scaler 仅 train。
- 在 val 上跑 naive + chronos-bolt + timesfm-2.0-pytorch，产出指标；对照 assets/leaderboard 既有口径。

## 11. 待确认/风险（需你或后续确认）
- **分钟粒度不可用**：全仓无 minute/second/tick 数据，最细为小时（仅 FX 9 对）。已按 (a) 仅日+小时；如需分钟需新增下载步骤。
- **Equity 目标**：默认 split_adjusted_close（无缺失）；若需 total_return_adjusted_close，2 只（83443/84788）缺 5,536 行需明确缺失处理。
- **FX 缺 1 对**：train 21 对、val/test 20 对，需确认缺失的是哪家、是否纳入（当前默认“全可用对”= val 20 对）。
- **档 1 宏观/基本面**：均为“假设可用”非验证发布日期（FRED 修订、6 月/1 月 lag、BIS 重建），若启用需在报告显著标注。
- **资源**：timesfm 权重 ~2GB、训练显存/时间可能受限；val 评估为必需，训练为可选。

---

## §12. Implementation Log (append)

- **standard/ 桶已接入**：`loader.py::_dataset_dir` 与 `registry.py::BUCKETS`/`_scan` 增加 `standard/`；`register_dataset` 支持 `bucket=` 参数。`list_datasets()` 现返回 101 个 `std_*` 数据集。
- **101 数据集已构建**（`scripts/build_standard_bench.py` → `assets/data/standard/`）：
  - FX daily **20** (lookback 52 / horizon 10)；FX hourly **9** (96 / 24)；FI **22** (52 / 10，4 国×公共 5 期限 + US 3/30)；Equity **50** (52 / 10，permno，target `split_adjusted_close`，按日期切分 train≤2018 / val 2019-2020 / test≥2021)。
  - 未变量化：每序列仅 `timestamp` + 目标列；`feature_names=[]`（避免 `iter_rolling` 生成重复列）。不伪造/不补齐；缺失项保持原样。
  - 校验：val 行数与基线一致（fx 12,283 / fi 10,987 / equity 25,250）；`iter_rolling(split="validation")` 产窗正常。
- **依赖已装**：torch(CPU)、chronos-forecasting 2.3.2、timesfm、pmdarima、transformers、scikit-learn、scipy（pypi 可达；经 `uv pip install --python .venv/bin/python`）。
- **模型可行性实测**：
  - `amazon/chronos-bolt-base` ✅ 加载 + 预测（`ChronosBoltPipeline`，输入 torch `[1,T]`，输出 `(1,9,H)`，取中位数分位数作 point）。
  - `google/timesfm-2.0-500m-pytorch` ❌ 无法加载：timesfm 2.3.x 仅有 `TimesFM_2p5_200M_torch`（默认 `google/timesfm-2.5-200m-pytorch`），500m 架构（~50 层 decoder）与 200m 类不匹配 → 缺/多 key。权重本身可下载（1.99GB，HTTP 200）。
  - **用户决策（选项 1）**：TimesFM 代表改用 `google/timesfm-2.5-200m-pytorch`（可加载，~200M，同族），报告中明确标注为 2.0-500m 的替代。
- **eval 脚本**：`scripts/eval_standard_bench.py` — `run_benchmark(..., split="validation")` 返回 `list[LeaderboardEntry]`（`metrics`: mae/smape/mape/rmse/mase/...，`score` 综合分）；按 domain/freq 聚合；写 `docs/standard_bench_report.md` + `.jsonl`。含 naive/moving_average/random_walk/seasonal_naive 基线。
- **smoke 通过**：4 数据集 × 6 模型全跑通，指标合理（如 FI bond：chronos MAE 0.084 / timesfm 0.088 / naive 0.088）。
- **全量 val 评估进行中**：`--step 10`（101 数据集，~15.4k 窗/模型）。
