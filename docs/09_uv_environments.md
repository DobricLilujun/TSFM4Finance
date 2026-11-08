# uv project layout（每个模型家族一个环境，pin 已冻结）

```bash
scripts/setup_envs.sh build      # uv sync envs/{core,legacy,moirai,moment,ttm,toto,tirex,tirex2,kronos,timemoe}
scripts/setup_envs.sh check      # weights --verify + doctor_all.sh
scripts/setup_envs.sh freeze     # 从当前可用的 venv 重新生成 envs/*/pyproject.toml 的 pin
```

- 每个 `envs/<family>/pyproject.toml` 里是**当前实际能跑**的版本（torch / transformers / numpy /
  hub / datasets / pyarrow / triton + 该家族包），不是 latest，避免某天解析到坏版本。
- `bench/config.yaml` 的 `envs:` 支持三种写法：`core`（裸名，解析到 `envs/core/.venv/bin/python`，
  回退 `.venv-core`、`.venv2`）、`.venv2/bin/python`、绝对路径。老的手搓 venv 仍然可用。
- 关键 pin（踩过坑的）：
  - `moirai`: torch 2.4.1（uni2ts 要 <2.5）、triton 3.0.0
  - `moment`: numpy 1.25.2（momentfm 硬 pin）、`huggingface-hub<1.0`、transformers 4.33.3
  - `toto`: python 3.12（toto-2 要求）
  - `timemoe`: transformers==4.44（4.46+/5.x 会 `DynamicCache.seen_tokens` 崩）
  - `tirex2`: **triton<3.6 + torch 2.7**（Triton 3.9 移除了 `tl.make_block_ptr`，
    `mlstm_kernels`/`flashrnn` 编译失败）；本地 torch 2.15/triton 3.9 跑不了这个模型
- 想再加一个模型：只在 `envs/<family>/pyproject.toml` 加 pin + 在 `config.yaml` 加一行，
  sweep/plan/worker 自动纳入。
