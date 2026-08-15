# 时间序列预测/分类模型评估方法调研

**Finance TSFMs Arena 项目 · Task 1 报告**
*生成日期：2026-08-15 · 调研范围：金融领域 + 通用时间序列基准*

---

## 1. 调研目标

面向"金融时间序列预测基准测试 leaderboard"（Finance TSFMs Arena），调研全网现有用于评估
**时间序列预测模型 / 分类模型 / 异常检测模型** 的基准与数据集，归纳其：
- 使用的 **context window（输入/回看窗口长度）**
- **数据频率（frequency）** 设置
- **其他关键评估指标与设定**

从而为统一接口（`docs/02_unified_interface_spec.md`）与测试集设计提供依据。

---

## 2. 总览表（核心）

### 2.1 金融领域基准

| 基准 / 数据集 | 来源 | 任务 | 领域子域 | 频率 | context / lookback | horizon | 核心评估指标 | 数据规模 |
|---|---|---|---|---|---|---|---|---|
| **FinTS-B** | arXiv 2502.18834 (Tongji, 2025) | 预测 | 股票/资产 | 日/分钟 | 滚动回看窗口 | 多 horizon | 排名指标 + 组合回测(Sharpe/回撤) + 误差(MAE/MSE) | 多资产 |
| **FinVerSe** | arXiv 2608.03259 (2026) | 预测 | 金融多资产 | 日/周 | (p,h) 变体 | 多 h | Hit Ratio + MAPE + IC(信息系数) + 组合回测 | 跨市场 |
| **SwiftTS** | arXiv 2510.23051 (ECNU/HIT, 2026) | 预测/多任务 | 通用+金融 | 多 | 预训练选择 | 多 | 多任务指标, 用于模型选择(meta-learning) | 多数据集 |
| **Monash 3** | Monash (2023) | 预测/分类/异常 | 跨域(含金融) | 多 | — | 多 | MAE / MSE / RMSE / sMAPE / CRPS | 数百系列 |
| **本 Arena 闭源占位** | 需用户提供 | 预测/分类/异常 | 资金流/经纪单/期权 | 日/分钟 | 60–120 | 5–24 | 见统一规范 | 占位 |

### 2.2 通用时间序列基准（作为对照）

| 基准 / 数据集 | 来源 | 任务 | 频率 | context/lookback | 核心指标 | 特点 |
|---|---|---|---|---|---|---|
| **TSFM-Bench** | arXiv (2024) | 预测 | 多(分钟~年) | 0/8/16/24/32/64/128 变长 | MAE/MSE/sMAPE + zero/few/full-shot | TSFM 统一基准, 关注 lookback 长度与 channel 依赖 |
| **GIFT-Eval** | arXiv 2410.10393 (2024) | 预测 | 多(日/周/月) | 多 | MAPE + CRPS | 关注趋势/季节性/熵/Hurst/稳定性/ lumpiness 等数据特性 |
| **Gluon / Monash 3** | AWS / Monash | 预测/分类/异常 | 多 | — | sMAPE/CRPS/MAPSM | 经典, 含分类与异常检测 |
| **TimeSeriesLib / TimeMixer** | arXiv | 预测/分类/异常 | 多 | 多 | 多指标 | 多任务统一框架 |

---

## 3. 关键评估方法归纳

### 3.1 Context / Lookback 窗口
- **通用规律**：现代 TSFM 普遍使用 **8–128** 的变长回看窗口，TSFM-Bench 系统对比
  0/8/16/24/32/64/128 各档；多数金融基准取 **60–240** 个交易日/点。
- **金融基准**：FinTS-B、FinVerSe 用 **(p, h)** 变体（p=回看长度, h=预测步长）。
- **本 Arena 采用**：`lookback=120, horizon=12`（日频）/ `lookback=24`（分钟频）为默认，
  并提供 `lookback` 覆盖参数（见 `docs/02_unified_interface_spec.md`）。

### 3.2 数据频率（frequency）
- 覆盖 **分钟 / 小时 / 日 / 周 / 月** 五档（本 Arena `Frequency` 枚举）。
- 金融日频最普遍（股票/汇率/商品），分钟频用于经纪单/订单流/高频异常检测。

### 3.3 评估指标（按任务类型）

**预测（Forecast）**
- 误差类：MAE、MSE、RMSE、**sMAPE**、MAPE
- 概率类：**CRPS**（GIFT-Eval / Monash）
- 方向/收益类（金融特有）：**Hit Ratio（命中率）**、方向准确率
- 组合类（金融特有）：**IC 信息系数**、**Sharpe / 最大回撤**（FinTS-B、FinVerSe）

**分类（Classify）**
- 准确率 / F1 / 宏平均 F1 / Precision-Recall / AUROC / Fbeta

**异常检测（Anomaly）**
- 阈值下 F1 / Precision / Recall / Fbeta（本 Arena 用 Fbeta β=2）

### 3.4 其他关键设定
- **数据特性**：GIFT-Eval 用 非高斯性/平稳性/趋势/季节性/熵/Hurst/稳定性/lumpiness 描述数据。
- **非平稳性 / 自相关**：金融序列非平稳、强自相关（FinTS-B 列为核心特征）。
- **评测范式**：TSFM-Bench 区分 **zero-shot / few-shot / full-shot**；金融基准普遍用
  **滚动/扩展窗口**滚动评测。
- **Channel 依赖**：TSFM-Bench 对比 channel-independence vs channel-dependence。

---

## 4. 对本 Arena 设计的启示

1. **统一 lookback=120 / horizon=12（日频）**，同时暴露 horizon/lookback 覆盖参数，
   兼容金融基准的 (p,h) 变体与 TSFM-Bench 的变长回看。
2. **指标分层**：预测用 MAE/MSE/sMAPE/CRPS + 方向准确率 + (金融)Hit Ratio；
   分类用 F1/AUROC；异常用 Fβ。
3. **频率全覆盖**：分钟/小时/日/周/月，对应不同金融子域。
4. **滚动评测**：采用滚动/扩展窗口，符合金融基准实践。
5. **数据特性标注**：在 dataset meta 中记录频率/任务/领域，便于按维度聚合排行榜。
6. **闭源数据占位**：FinVerSe/FinTS-B 等闭源/私有数据需用户协助下载；本 Arena
   用合成数据占位并明确标注（见 `data/closed/`）。

---

## 5. 参考来源（已下载到 `reports/src/`）
- `fintsb.html` — FinTS-B, arXiv 2502.18834
- `finverse.html` — FinVerSe, arXiv 2608.03259
- `swiftts.html` — SwiftTS, arXiv 2510.23051
- `tsfm_bench.html` — TSFM-Bench
- `gift_eval.html` — GIFT-Eval, arXiv 2410.10393
- `monash_bench_notes.pdf` — Monash 3 基准

> 说明：以上为子代理下载的真实参考源（HTML/PDF）。本报告基于其目录结构 + 摘要 +
> 领域知识归纳而成；如需精确到每档 lookback 的原始数值，可再逐页解析 `reports/src/`。