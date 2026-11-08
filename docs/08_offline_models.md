# 离线 / 本地权重（no Hugging Face access）

模型可以来自三种地址，任选其一，**同一份 config 不用改**：
1. Hugging Face id（在线或有缓存时）：`amazon/chronos-2`
2. 本地目录（`git clone` / `hf download` 的产物）：`models/amazon/chronos-2` 或绝对路径
3. 直接指定路径：`--weights-root /data/models`，或直接写 `weights: /mnt/nas/chronos-2`

解析顺序（`tsfm4finance/bench/weights.py`）：路径 → `$TSFM_WEIGHTS_ROOT/<org>/<repo>` →
`<root>/<org>__<repo>` → `<root>/<repo>` → 当作 Hub id。**半克隆的目录会明确报错**，不会静默失败。

## 1. 在有网机器上把权重拿下来

```bash
cd TSFM4Finance
# 生成全部 clone 命令（也可用 huggingface-cli download）
.venv2/bin/python -m tsfm4finance.bench.weights --list-clone > fetch_models.sh
git lfs install
bash fetch_models.sh                      # 会 clone 到 models/<org>/<repo>
# Kronos 的训练栈（trainer 需要）
git clone https://github.com/NeoQuasar/Kronos third_party/Kronos
```

想只搬权重、不用 git-lfs：`huggingface-cli download <org>/<repo> --local-dir models/<org>/<repo>`。

## 2. 拷到离线服务器

```bash
rsync -a --info=progress2 models/ user@server:~/TSFM4Finance/models/
rsync -a third_party/Kronos/ user@server:~/TSFM4Finance/third_party/Kronos/
```

## 3. 离线服务器上启用本地权重

```bash
export TSFM_WEIGHTS_ROOT=$PWD/models
export TSFM_OFFLINE=1            # 设 HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE，禁止联网
# 只用了 from_pretrained(dir) 的库（Chronos / TimesFM / Moirai / MOMENT / TTM / Toto / Kronos）到此就够：
.venv2/bin/python -m tsfm4finance.bench.weights --verify

# 少数库（TiRex / Time-MoE remote code）内部走 Hub API：把 models/ 镜像成一个本地 HF 缓存
.venv2/bin/python -m tsfm4finance.bench.weights --link-into-cache .hf-cache
export HF_HOME=$PWD/.hf-cache
```
之后所有 sweep / trainer 命令都照原样跑（见 `docs/07_run_plan.md`），不再需要网络。

## 4. 检查 / 排错

```bash
.venv2/bin/python -m tsfm4finance.bench.weights --verify            # LOCAL / CACHED / MISSING
.venv2/bin/python -m tsfm4finance.bench.weights --resolve amazon/chronos-2
./reports/doctor_all.sh                                             # 每个 env 能否载入权重
```
- `found a partial directory under ...` → 目录不完整（git-lfs 没装或 clone 中断）：
  `git lfs install && git clone --force` 或 `huggingface-cli download` 重来。
- `LocalEntryNotFoundError` → 忘了 `--link-into-cache` + `HF_HOME`。
- 已有 HF 缓存的机器（本仓库开发机就是）：不用 clone，`TSFM_OFFLINE=1` 即可用缓存。

## 已验证（空 HF 缓存 + 只用 `models/`，`smoke` 全绿）
`chronos-2`、`chronos-bolt-base`、`timesfm-3.0`、`toto-2-22m`、`ttm-r2`、`moirai-2.0-R-small`、
`moment-1-large`、`tirex`（需 `--link-into-cache`）。

## 缺权重时的精确命令（`--verify` 会直接打印）
`chronos-t5-large`、`moirai-1.1-R-base`、`moirai-moe-1.0-R-base`、`kronos-base`、
`timemoe-200m`、`toto-2-313m`、`toto-2-1B`、`toto-2-2.5B`。

## 微调权重的对应关系（防泄漏）
`adaptation != zero` 的 cell 只能加载**同一 series** 训练出来的 checkpoint：
`reports/ft_ckpts/<model>__<series>__<budget>.pt`（trainer 产物）。
- 找不到 → 该 cell 明确 skip：`no fine-tuned checkpoint for <model>/<budget> on <series>`，
  绝不会退回 base weights 冒充微调结果。
- 加载时记录 provenance：结果 JSON 里 `engine.ft = {checkpoint, via, missing, unexpected}`，
  `missing=unexpected=0` 表示权重确实写进了模型。
- 派生任务（`_trade/_dir5/_volreg/_extreme`）沿用底层 series 的 checkpoint。
- checkpoint 也可以是 `save_pretrained` 目录（如 `chronos-2__<series>__fs16_pipe/`），
  这时直接按目录加载。
