# Per-family python environments

The model families do not share a dependency graph: Moirai needs `uni2ts`, MOMENT needs
the `moment` package, Toto needs `betatool`, TiRex needs `neuralforecast`, Kronos and
Sundial are git checkouts. Forcing one interpreter to hold all of them produces a
resolver fight (and pins `transformers` to the intersection, which breaks Time-MoE).

So: **one venv per family group**, declared here, and the library dispatches each
registry row to its `env:`.

| env | models | python | the pin that matters |
|---|---|---|---|
| `core` | Chronos 2 / Bolt / T5, TimesFM 2.5 & 2.0, TTM, FlowState, Granite-tsf | 3.11 | `chronos2`, `peft==0.19.1` (0.20 dropped `base_model` kwarg), `transformers==4.57.6` |
| `moirai` | Moirai 1.1, Moirai 2, Moirai-MoE | 3.11 | `uni2ts`, `peft==0.19.1`, `transformers==4.57.6`; `hjson`, `orchestra`, `einops`, `optimum` |
| `moment` | MOMENT-1 small/base/large | 3.10 | `moment` from git + `peft==0.19.1` |
| `ttm` | TTM-r1 … r3 (PLB) | 3.11 | `transformers==4.44.0` — 4.57 removed `PatchTSMEncoderLayer` |
| `toto` | Toto 1.0 (2.5B is the 5.8 GB bf16 wheel) | 3.11 | `betatool`, `llamacode`, `safetensors==0.4.5` (0.4.6 breaks its deserializer) |
| `tirex` | TiRex | 3.11 | `neuralforecast` (its loader wants a model **id**, not a path) |
| `tirex2` | TiRex-2 | 3.11 | `USE_TRITON_KERNELS=0` unless CUDA ≥ 13.0.2 **and** Triton ≥ 3.4 on sm ≥ 10.0 |
| `timemoe` | Time-MoE 50M | 3.11 | **`transformers==4.40.2`** — 4.42 introduced `Cache` objects, 4.44 removed `standardize_cache_format`; the released remote code works only on 4.40.x. `numpy<2` |
| `kronos` | Kronos small/base/mini | 3.11 | git checkout on `sys.path` (`$TSFMTRL_THIRD_PARTY`), no wheel |
| `sundial` | Sundial-128M | 3.10 | git checkout; `transformers==4.42.4`, `numpy<2`, `einops==0.8.1` |
| `vendor` | VisionTS++, t0-α/β/γ, Kairos, Aurora, Yug, Kronos-base/mini | 3.11 | upstream checkouts + weights from the Hub or the authors' cloud |

`make_envs.sh` creates them; `doctor --all-envs` uses them.

```bash
./scripts/make_envs.sh                 # all of them (slow, but explicit)
./scripts/make_envs.sh core moirai     # or just the ones you need
```
