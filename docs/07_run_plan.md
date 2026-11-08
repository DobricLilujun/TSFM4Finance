# 执行计划 / Run plan (what to test, which models, what to train, exact commands)

所有命令都在仓库根目录执行：`cd TSFM4Finance`。
所有阶段都是**可断点续跑**的（`reports/state.sqlite3`）：中断后重跑同一条命令即可继续。

## 离线上网受限？先看这一节（本地权重 + uv 环境）

```bash
# 有网机器：抓权重 + 训练栈
.venv2/bin/python -m tsfm4finance.bench.weights --list-clone > fetch_models.sh
git lfs install && bash fetch_models.sh
git clone https://github.com/NeoQuasar/Kronos third_party/Kronos
rsync -a models/ third_party/Kronos/ user@server:~/TSFM4Finance/

# 离线服务器：一键建环境 + 启用本地权重
scripts/setup_envs.sh build                     # uv 项目环境（envs/*/.venv），pin 已冻结
export TSFM_WEIGHTS_ROOT=$PWD/models TSFM_OFFLINE=1
.venv2/bin/python -m tsfm4finance.bench.weights --verify
.venv2/bin/python -m tsfm4finance.bench.weights --link-into-cache .hf-cache && export HF_HOME=$PWD/.hf-cache
```
之后下面所有命令里的 `.venv2/bin/python` 都可换成 `envs/core/.venv/bin/python`；
`config.yaml` 的 `envs:` 用裸名（`core`/`moirai`/…）会自动解析到 uv 环境。
模型地址三种写法都支持：Hub id、`models/<org>/<repo>`、绝对路径（或 `--weights-root`）。
详见 `docs/08_offline_models.md`、`docs/09_uv_environments.md`。

微调阶段的产物命名是契约：`reports/ft_ckpts/<model>__<series>__<budget>.pt`；
评测端按 series 严格匹配，缺 checkpoint 会 skip 而不是退回零样本权重（见 `docs/08_offline_models.md`）。

约定：
- `core` 环境 = `.venv2/bin/python`（sweep 驱动、Chronos/TimesFM3.0/基线）
- 其它模型家族各有自己的 venv（见 `docs/06_multi_model_sweep.md` 的安装矩阵）
- 单卡（1 GPU）时 `--workers 1`；多卡才用 `--workers 2+` 或分 shard 并行

---

## Phase 0 — 环境体检（先跑这个，5–10 分钟）

```bash
./reports/doctor_all.sh          # 每个 venv 能否 import + 载入权重 + 出一次预测
./reports/smoke_all.sh           # 每个引擎在真实窗口上出形状/指标（h=21, 4 windows）
```

单个 env 自检（想手动确认某个模型时）：

```bash
.venv2/bin/python      -m tsfm4finance.bench.smoke --all-tiers A,B --series sp500_long --horizon 21 --n 4
.venv-moirai/bin/python -m tsfm4finance.bench.smoke --all-tiers A,B --series sp500_long --horizon 21 --n 4
.venv-ttm/bin/python    -m tsfm4finance.bench.smoke --all-tiers A,B --series sp500_long --horizon 21 --n 4
.venv-toto/bin/python   -m tsfm4finance.bench.smoke --all-tiers A,B --series sp500_long --horizon 21 --n 4
.venv-tirex/bin/python  -m tsfm4finance.bench.smoke --all-tiers A,B --series sp500_long --horizon 21 --n 4
.venv-kronos/bin/python -m tsfm4finance.bench.smoke --all-tiers A,B --series sp500_long --horizon 21 --n 4
.venv-moment/bin/python -m tsfm4finance.bench.smoke --all-tiers A,B --series sp500_long --horizon 21 --n 4
.venv-timemoe/bin/python -m tsfm4finance.bench.smoke --all-tiers A,B --series sp500_long --horizon 21 --n 4
```

已知两个引擎阻塞（不要浪费时间反复试）：
- `timemoe-*`：需要 `transformers==4.44`（用 `.venv-timemoe`；`core` 里的新版会崩）
- `tirex-2`：本地 Triton 3.9 删了 `tl.make_block_ptr`；服务器用 **torch 2.6–2.8 + triton 3.2–3.4**

---

## Phase 1 — 数据（T2/T3/T4 + 派生任务）

```bash
# T2: fev-bench + GIFT-Eval + 截面面板（Nasdaq-100 close/return）
.venv2/bin/python -m tsfm4finance.bench.download --tier t2 --xs-limit 60

# T3: 币安 1h K线（2019-01 起，体积较大，网络慢时可跳过）
.venv2/bin/python -m tsfm4finance.bench.download --tier t3 --intraday-interval 1h --intraday-start 2019-01

# T4: Scrapling 找可公开获取的补充源；需要 key/授权的部分会明确报出来，不会静默跳过
.venv2/bin/python -m tsfm4finance.bench.download --tier t4

# 已下载清单 + 失败清单
.venv2/bin/python -m tsfm4finance.bench.download --status

# 为每个基础序列生成派生任务：_dir5(方向) / _volreg(波动率三档) / _extreme(极值事件) / _trade(交易)
.venv2/bin/python -m tsfm4finance.bench.sweep --stage derive --groups core12,long12,newx,fred
```

数据集分组（`bench/config.yaml` → `series_groups`）：

| group | 内容 | 用途 |
|---|---|---|
| `core12` | 论文里已有的 12 条日线（2023-08→今） | 与已发表 §4.4 数字对齐 |
| `long12` | 同一 12 个标的的长历史 OHLCV（2004→今，含多轮危机） | 主结果（跨周期、含 OHLCV→Kronos 可用） |
| `newx` | VIX、利率、FX、商品、全球指数、行业 ETF 等 ~32 个 | 广度/稳健性 |
| `fred` | 31 条宏观/利率/信用序列 | 宏观面板、fev-bench 对齐 |
| `intraday` | 币安 1h（T3 下载后自动发现） | 高频泛化 |
| `external` | fev-bench / GIFT-Eval 子集（T2） | 与公开套件对齐 |

---

## Phase 2 — 零样本推理网格（不训练；这是论文主表）

**测什么**：19 个 Tier-A/B 模型 × 数据集 × `h ∈ {1,5,12,21,63}` × 5 层指标
（点精度 MASE/NDp、显著性 DM/SPA/BH、概率校准 CRPS/WQL/PIT/QLIKE/Kupiec、经济层 Sharpe/成本、任务层 PR-AUC/F1）。

```bash
# 2a. 生成 cell 网格（幂等；--force 才重算已完成的）
.venv2/bin/python -m tsfm4finance.bench.sweep --stage plan \
    --tiers A,B --groups core12,long12 --horizons 1,5,12,21,63 --contexts 512

# 2b. 跑预测（单卡）
.venv2/bin/python -m tsfm4finance.bench.sweep --stage forecast --workers 1 --amp bf16 --batch 16

# 2b'. 多卡 / 多机：把同一网格切 4 份（每台机器一条，shard 不同）
.venv2/bin/python -m tsfm4finance.bench.sweep --stage forecast --shard 0 --n-shards 4 --amp bf16
.venv2/bin/python -m tsfm4finance.bench.sweep --stage forecast --shard 1 --n-shards 4 --amp bf16
.venv2/bin/python -m tsfm4finance.bench.sweep --stage forecast --shard 2 --n-shards 4 --amp bf16
.venv2/bin/python -m tsfm4finance.bench.sweep --stage forecast --shard 3 --n-shards 4 --amp bf16

# 2c. 经济层：信号 → 仓位 → 成本（0/10/30bps）→ DSR/capture/Ulcer/cost-slope
.venv2/bin/python -m tsfm4finance.bench.sweep --stage trading --workers 1

# 2d. 判别任务：方向 / 波动率三档 / 极值事件（out-of-fold probe，无泄漏）
.venv2/bin/python -m tsfm4finance.bench.sweep --stage classify --workers 1
.venv2/bin/python -m tsfm4finance.bench.sweep --stage anomaly --workers 1

# 2e. 进度与导出（PROGRESS.md 是唯一权威状态表）
.venv2/bin/python -m tsfm4finance.bench.sweep --stage status
.venv2/bin/python -m tsfm4finance.bench.sweep --stage export     # reports/multi_model_results.json
```

**Tier C（重模型，只在足够大的卡上跑）**：

```bash
.venv2/bin/python -m tsfm4finance.bench.sweep --stage plan --tiers C --groups long12 --horizons 5,21
.venv2/bin/python -m tsfm4finance.bench.sweep --stage forecast --tiers C --groups long12 --horizons 5,21 --amp bf16 --batch 4
.venv-tirex2/bin/python -m tsfm4finance.bench.cell_worker --stage forecast --env tirex2 --amp bf16   # 需 torch/triton 匹配
```

**只跑某几个模型 / 某几个数据集**（调试或补漏）：

```bash
.venv2/bin/python -m tsfm4finance.bench.sweep --stage forecast --models chronos-2,timesfm-3.0,moirai-2.0-R-small --series sp500_long,btc_long
```

---

## Phase 3 — 训练（每个模型各自的 trainer；这是你要在 GPU 服务器上跑的重活）

三个预算（基准定义，不随手调）：`fs16`=16 窗口/40 步，`fs64`=64 窗口/120 步，`full`=全部训练窗口 × 3 epoch。
全部是**全参数**微调（`--heads-only` 是消融开关，会写进 metadata，绝不会伪装成 full）。
窗口只从 train split 采样，checkpoint 里记录 `train_end`。

```bash
# 3.0 先看每个家族的 trainer 状态（哪些已实现、哪些 pending）
.venv2/bin/python -m tsfm4finance.bench.trainers --list

# 3.1 先 smoke（2 步，证明 loop 通了，别当训练）
.venv/bin/python        -m tsfm4finance.bench.trainers --model timesfm-2.5-200m --series sp500_daily --budget fs16 --smoke --bs 2 --ctx 256
.venv2/bin/python       -m tsfm4finance.bench.trainers --model chronos-2        --series sp500_daily --budget fs16 --smoke --bs 2 --ctx 256
.venv-moirai/bin/python -m tsfm4finance.bench.trainers --model moirai-2.0-R-small --series sp500_daily --budget fs16 --smoke
.venv-ttm/bin/python    -m tsfm4finance.bench.trainers --model ttm-r2           --series sp500_daily --budget fs16 --smoke
.venv-moment/bin/python -m tsfm4finance.bench.trainers --model moment-1-large   --series sp500_daily --budget fs16 --smoke --bs 1
.venv-toto/bin/python   -m tsfm4finance.bench.trainers --model toto-2-22m       --series sp500_daily --budget fs16 --smoke
```

```bash
# 3.2 正式训练（Tier-A，12 条长历史序列，fs16 → fs64 → full）
LONG=sp500_long,nasdaq_long,dow_long,russell2000_long,btc_long,eth_long,eurusd_long,gbpusd_long,gold_long,crude_oil_long,us_treasury_7y_long,ig_corp_bond_long

for B in fs16 fs64 full; do
  .venv/bin/python  -m tsfm4finance.bench.trainers --model timesfm-2.5-200m --series $LONG --budget $B --epochs 3 --bs 8 --ctx 512 --amp bf16
  .venv2/bin/python -m tsfm4finance.bench.trainers --model chronos-2        --series $LONG --budget $B --epochs 3 --bs 8 --ctx 512 --amp bf16
done

# Tier-B（按显存选 bs/ctx）
.venv-moirai/bin/python -m tsfm4finance.bench.trainers --model moirai-2.0-R-small --series $LONG --budget fs64
.venv-moirai/bin/python -m tsfm4finance.bench.trainers --model moirai-2.0-R-small --series $LONG --budget full --epochs 3 --bs 4 --amp bf16
.venv-ttm/bin/python    -m tsfm4finance.bench.trainers --model ttm-r2             --series $LONG --budget full --epochs 3 --bs 8
.venv-toto/bin/python   -m tsfm4finance.bench.trainers --model toto-2-22m         --series $LONG --budget full --epochs 3 --bs 4 --amp bf16
.venv-moment/bin/python -m tsfm4finance.bench.trainers --model moment-1-large     --series $LONG --budget full --epochs 3 --bs 2
.venv-moirai/bin/python -m tsfm4finance.bench.trainers --model moirai-1.1-R-base  --series $LONG --budget fs64
```

```bash
# 3.3 Kronos：走它自己的训练栈（先导出 CSV+config，再加 --run-training 真跑）
.venv-kronos/bin/python -m tsfm4finance.bench.trainers --model kronos-small --series sp500_long --budget full
.venv-kronos/bin/python -m tsfm4finance.bench.trainers --model kronos-small --series sp500_long --budget full --run-training
```

```bash
# 3.4 训练完把微调变体加入网格并评测（planner 只会为"已存在 checkpoint"生成微调 cell）
.venv2/bin/python -m tsfm4finance.bench.sweep --stage plan \
    --tiers A,B --groups core12,long12 --horizons 1,5,12,21,63 --adaptations zero,fs16,fs64,full
.venv2/bin/python -m tsfm4finance.bench.sweep --stage forecast --workers 1 --amp bf16
.venv2/bin/python -m tsfm4finance.bench.sweep --stage trading  --workers 1
.venv2/bin/python -m tsfm4finance.bench.sweep --stage status
.venv2/bin/python -m tsfm4finance.bench.sweep --stage gc        # 删掉已评测的 checkpoint，省磁盘
```

小显存备选：`--amp bf16` / `--bs 1..4` / `--ctx 192|256`；消融：`--heads-only --out-dir reports/ft_ckpts_heads`。

---

## Phase 4 — 控制实验（论文里必须有的三张对照）

```bash
# 4a. span 敏感性（同样的协议换到 validation span 上重跑；已发表 §4.4 就是靠它发现"排名随跨度变"）
.venv2/bin/python -m tsfm4finance.bench.sweep --stage plan    --tiers A,B --groups core12,long12 --horizons 5,21 --spans val
.venv2/bin/python -m tsfm4finance.bench.sweep --stage forecast --spans val --workers 1

# 4b. 随机种子稳定性（只在代表性子集上做，成本可控）
.venv2/bin/python -m tsfm4finance.bench.sweep --stage plan --models chronos-2,timesfm-3.0,moirai-2.0-R-small \
    --series sp500_long,btc_long,eurusd_long --horizons 21 --seeds 0,1,2
.venv2/bin/python -m tsfm4finance.bench.sweep --stage forecast --seeds 0,1,2

# 4c. 上下文长度消融（TimesFM 512 vs 更短：短历史金融序列上常见）
.venv2/bin/python -m tsfm4finance.bench.sweep --stage plan  --models chronos-2,timesfm-3.0,chronos-bolt-base --groups long12 --horizons 5,21 --contexts 128,256,512
.venv2/bin/python -m tsfm4finance.bench.sweep --stage forecast --contexts 128,256,512
```

成本敏感性（0/10/30bps）与 buy&hold/momentum/random 基准已经在 `trading` 阶段自动算，不用额外命令。

---

## Phase 5 — 汇总、生成论文表格、同步 Overleaf

```bash
.venv2/bin/python -m tsfm4finance.bench.sweep --stage export    # reports/multi_model_results.json
.venv2/bin/python -m tsfm4finance.bench.sweep --stage status    # reports/PROGRESS.md（含 failed/skipped 明细）
cat reports/PROGRESS.md

# 论文（表格由 results JSON 生成，不手抄数字）
cd ../TSFM_Paper && ./build.sh && git add -A && git commit -m "multi-model results" && git push
```

> 需要我先给 `bench/paper_table.py` 加 `--results reports/multi_model_results.json`，才能把多模型网格自动写进 §4.4 的表（现在它只读单模型 §4.4 的 JSON）。

---

## 测什么（清单）

**模型（29 个注册，19 个 Tier-A/B 已通）**
- Tier A：`naive seasonal_naive moving_average arima arima_auto ets theta`、`chronos-2 chronos-bolt-base`、`timesfm-2.5-200m timesfm-3.0`、`moirai-2.0-R-small`、`ttm-r2`、`tirex`
- Tier B：`chronos-t5-base/large`、`moirai-1.1-R-base moirai-moe-1.0-R-base`、`moment-1-large`、`flowstate-r1`、`kronos-small/base`、`timemoe-50m/200m`、`toto-2-22m/313m`
- Tier C（大卡）：`tirex-2`、`toto-2-1B`、`toto-2-2.5B`
- 微调变体：`zero / fs16 / fs64 / full`（+ `heads-only` 消融）

**任务**：预测（多跨度 test / val / val+test）、量化交易（sign & vol-target，0/10/30bps，含基准）、方向分类、波动率三档、极值事件异常检测（+ embedding/native-head 路由）

**指标（五层，论文 §4.2 已写好理由）**
1. 点精度：MASE、NDp、RMSE/MAE/sMAPE、directional accuracy、relative skill
2. 显著性：DM(Newey-West + MZ)、SPA/Reality Check(stationary bootstrap)、BH-FDR、Deflated Sharpe
3. 概率校准：CRPS/WQL/pinball、PIT 均值+KS、逐水平 coverage 偏差、QLIKE + MZ 方差回归、Kupiec + Christoffersen、VaR/ES
4. 经济层：IC/rank-IC、hit rate、毛/净年化、Sharpe/Sortino、最大回撤与持续期、Ulcer、turnover、市场 capture、Sharpe-成本斜率与 break-even bps、净收益 t 值
5. 任务保真度：PR-AUC、ROC-AUC、F1(最优阈值/5% flag)、F-β、majority-class 参照、out-of-fold probe

**协议**：滚动起点、非重叠窗口 `stride=h`、`h ∈ {1,5,12,21,63}`、context ≤512、MASE 尺度只用 train、每个窗口写 no-leakage 断言、mean±std across origins、span 对照。
