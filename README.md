# Finance TSFMs Arena

一个面向**金融时间序列预测**的统一基准测试与排行榜系统 (Arena / Leaderboard)。

## 特性
- 同时支持 **开源 (open)** 与 **闭源 (closed)** 数据集
- 明确划分 **training set** 与 **validation (test) set**
- 统一的**输入/输出接口**规范，覆盖多种金融子领域与任务类型
- 本地后端服务 (FastAPI) + 美观的前端网站 (GitHub Pages)
- 两种评测模式：
  1. **公开评测**：模型在开源评测集上运行，结果公开上榜
  2. **私有评测**：用户上传模型，系统在闭源数据上评估，结果仅自己可见
- 排行榜 (Leaderboard) 展示多个时间序列基础模型 (TSFM) 的对比

## 目录结构
- `arena/`       核心库：统一接口、数据集、评估指标、模型适配器
- `backend/`     FastAPI 后端服务
- `frontend/`    前端网站 (静态，GitHub Pages)
- `data/`        open/ 与 closed/ 数据集
- `docs/`        接口与规范文档
- `reports/`     调研与评估报告
- `tests/`       测试

## 快速开始
```bash
uv venv --python 3.11
uv pip install -e .
uv pip install -e ".[models]"   # 可选：安装模型依赖
uv run python -m backend.app    # 启动后端
```
