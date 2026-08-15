# Finance TSFMs Arena — 统一接口与测试集规范 v1.0

> 本规范定义整个 Arena 的数据格式、任务类型、输入/输出契约、评估指标与评测流程。
> 所有模型、数据集、评估器必须遵守本规范。参考调研见 `reports/01_benchmark_survey.md`。

---

## 0. 设计原则（来自调研归纳的通用规律）

1. **统一为"多变量 + 时间索引"的二维结构**：任何金融序列都归一为 `timestamp × feature` 矩阵，
   单变量是列数为 1 的特例。
2. **context window 与 horizon 显式声明**：每个任务必须声明 `lookback`（输入窗口长度）与
   `horizon`（预测/评估步长），二者可不同。
3. **频率显式声明**：`frequency ∈ {tick, second, minute, hour, day, week, month, quarter, year}`。
4. **train / validation / test 三段划分**：默认按时间切分（不随机 shuffle），validation 用于早停，
   test 用于最终排行榜评分。
5. **开源 / 闭源双轨**：同一任务可分别用开源数据（公开评测）与闭源数据（私有/上传评测）运行，
   指标口径一致。
6. **金融特化指标**：在通用数值指标之外，金融场景额外用 `方向准确率 (directional accuracy)`、
   `收益相关 (rank/Pearson corr on returns)`、`回测收益/夏普` 等。

---

## 1. 数据格式（Dataset）

### 1.1 存储格式
采用 **Parquet**（列式、可压缩、带 schema）。每个数据集是一个目录，包含：

```
<data_name>/
  meta.json          # 元数据（见 §1.2）
  train.parquet      # 训练集
  validation.parquet # 验证集
  test.parquet       # 测试集（最终评分用）
  labels.json        # （分类/异常任务才有）标签定义
```

### 1.2 meta.json 字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `name` | str | 数据集名称，如 `sp500_daily` |
| `domain` | enum | `equity` / `crypto` / `fx` / `index` / `bond` / `commodity` / `fraud` / `event` / `general` |
| `task` | enum | `forecast` / `classify` / `anomaly`（见 §2） |
| `frequency` | enum | 见 §1.3 |
| `n_features` | int | 特征/变量列数 |
| `feature_names` | list[str] | 每列含义（如 `open,high,low,close,volume,return`） |
| `n_obs` | int | 总观测数（时间点数） |
| `date_range` | [start,end] | 时间跨度 |
| `split` | {train,val,test} | 各段的时间起止 |
| `license` | str | 许可 / 是否闭源 |
| `source` | str | 数据来源 URL / 机构 |
| `open` | bool | 是否开源（true=公开评测可用） |
| `requires_download` | bool | 是否需要用户协助下载（闭源时为 true） |

### 1.3 频率枚举
`tick` / `second` / `minute` / `hour` / `day` / `week` / `month` / `quarter` / `year`

### 1.4 Parquet 列约定
- 第 0 列：`timestamp`（datetime，带频率信息）
- 其余列：特征值（float）
- 分类任务额外一列 `label`（整数类编码），并配 `labels.json`
- 异常任务额外一列 `anomaly`（0/1 或异常区间）

---

## 2. 任务类型（Task）

| 任务 | 目标 | 评估指标（主） | 典型金融场景 |
|---|---|---|---|
| **forecast** | 预测未来 h 步数值 | MASE, RMSE, MAE, sMAPE, CRPS | 价格/收益/波动率预测 |
| **classify** | 预测类别标签 | Accuracy, Precision, Recall, F1, AUC-ROC | 涨跌方向、事件/情感分类 |
| **anomaly** | 检测异常/欺诈 | F1, Precision, Recall, AUC-ROC, FPR | 欺诈检测、异常交易 |

### 2.1 预测子设定
- `target`：预测目标列（默认最后一列，如 `close` 或 `return`）
- `lookback`（context window）：输入历史长度
- `horizon`：未来预测步数
- `rolling`：是否滚动窗口回测（backtest），默认 true

### 2.2 金融特化指标（forecast 额外）
- `directional_accuracy`：预测方向与实际方向一致的比例
- `return_rank_corr`：预测收益序列与真实收益序列的秩相关（Spearman）
- `backtest_return` / `sharpe`：（可选）按预测信号回测的收益/夏普

---

## 3. 模型接口（Model Adapter）

每个模型（基础模型或上传模型）实现统一接口 `BaseModel`：

```python
class BaseModel:
    name: str                       # 模型名
    supports: set[str]             # 支持的 task: {forecast, classify, anomaly}
    def fit(self, train_df, meta) -> None: ...          # 可选（TSFM 多为 zero-shot）
    def predict(self, context_df, meta, horizon) -> PredictOutput: ...
```

### 3.1 输入（predict 的 context）
- `context_df`：形状 `(lookback, n_features)` 的 DataFrame（含 timestamp）
- `meta`：任务元数据（task, target, frequency, horizon, class labels...）
- `horizon`：预测步数

### 3.2 输出（PredictOutput）
| 字段 | 类型 | 说明 |
|---|---|---|
| `point` | np.ndarray | 点预测，形状 `(horizon,)` 或 `(horizon, n_targets)` |
| `quantiles` | np.ndarray | （可选）分位数预测，形状 `(K, horizon)`，如 0.1/0.5/0.9 |
| `probabilities` | np.ndarray | （分类任务）各类别概率，形状 `(n_classes,)` 或 `(horizon, n_classes)` |
| `confidence` | float | 置信度（可选） |

> 点预测统一为 `point`；概率/分位数可选。评估器据此计算数值指标与方向/概率指标。

---

## 4. 评估接口（Evaluator）

```python
evaluate(pred: PredictOutput, truth_df, meta) -> dict[str, float]
```
返回指标字典，例：
```json
{
  "rmse": 12.3, "mae": 8.1, "mase": 0.9, "smape": 15.4,
  "directional_accuracy": 0.61, "return_rank_corr": 0.52,
  "score": 0.734   // 综合分（见 §5）
}
```

### 5. 综合评分（score）
为便于排行榜排序，定义综合分（0~1，越高越好）。以 forecast 为例，采用多指标归一加权：

```
score = 0.35·(1−clip(mase)) + 0.25·directional_accuracy
      + 0.20·return_rank_corr + 0.20·(1−clip(smape/100))
```
（分类任务用 F1/AUC 归一，异常任务用 F1/AUC-ROC 归一；权重按任务类型调整。）

---

## 6. 两种评测模式

| 模式 | 数据 | 结果可见性 | 流程 |
|---|---|---|---|
| **公开评测** | 开源数据集 | 公开上榜 | 系统跑预置模型 / 用户选开源集 → 上榜 |
| **私有评测** | 闭源数据集 | 仅用户可见 | 用户上传模型 → 系统在闭源 test 上评估 → 仅返回给自己，不公开 |

- 闭源数据默认**不提供下载**，仅允许上传模型后由系统在服务端评估，保证数据安全。
- 用户可"仅用开源集测试"（不上传任何闭源数据），满足不想公开模型的需求。

---

## 7. 排行榜（Leaderboard）字段

| 字段 | 说明 |
|---|---|
| `rank` | 排名（按 score） |
| `model_name` | 模型名 |
| `dataset` | 数据集 |
| `task` | 任务 |
| `frequency` | 频率 |
| `metrics` | 完整指标字典 |
| `score` | 综合分 |
| `mode` | `open` / `private` |
| `timestamp` | 评测时间 |
| `is_benchmark` | 是否官方基准模型 |

---

## 8. 数据可获取性（开源 vs 闭源）

- **开源（可直接本地获取）**：见 `reports/01_benchmark_survey.md` 与 `arena/datasets/` 的下载脚本
  （Yahoo Finance / STOOQ / CoinGecko / OANDA / KDD Cup 等）。
- **闭源（需用户协助）**：需申请或付费的数据集，以占位 + `requires_download=true` 标注，
  由用户下载/上传后接入。