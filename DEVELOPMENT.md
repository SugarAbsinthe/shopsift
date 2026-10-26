# 开发说明

[安装与启动](README.md) · [商品目录](datasets/real_catalog/README.md)

## 环境与配置

使用 Python 3.12、Node.js 24；依赖安装和服务命令见 README。复制 `.env.example` 为 `.env`，填写支持工具调用的模型 API 配置。Redis、LangSmith 默认关闭。

主要配置位于 [`src/config.py`](src/config.py)：

| 配置 | 默认值 / 用途 |
|---|---|
| `PRODUCT_DB_PATH` | `data/products.db` |
| `PRODUCT_CHROMA_DIR` | `data/product_chroma_db` |
| `EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5`，须与索引一致 |
| `AGENT_CHECKPOINT_DB_PATH` | `data/agent_checkpoints.db` |
| `LLM_MAX_OUTPUT_TOKENS` | 单次输出上限 4096 |
| `MAX_LLM_CALLS` / `MAX_TOTAL_TOKENS` | 每轮 10 次调用 / 32768 个已报告 Token |
| `MAX_RUN_SECONDS` / `AGENT_REQUEST_TIMEOUT_SECONDS` | 图运行 / 同步请求 120 / 150 秒 |
| `REDIS_ENABLED` / `REDIS_URL` | 默认关闭；缓存 TTL 900 秒 |

Token 预算依赖服务商用量；时间与取消检查不能立即终止已发出的同步模型请求。

### WSL 环境准备

Windows 与 WSL 分别安装 Python、Node.js 和项目依赖，不能共用 `.venv` 或 `node_modules`。建议 WSL 项目放在 `~/projects/`。检查 `command -v python3.12 node npm`，确保使用 Linux 安装。

系统 Python 版本不符时，可通过 [uv](https://docs.astral.sh/uv/getting-started/installation/)安装独立 Python：

```bash
curl -LsSf https://astral.sh/uv/install.sh -o /tmp/shopsift-uv-install.sh
sh /tmp/shopsift-uv-install.sh
source "$HOME/.local/bin/env"
uv python install 3.12
```

Node.js 安装 [24 系列 Linux 发行包](https://nodejs.org/en/download)。若 npm 指向 `/mnt/c/...` 或 `/mnt/d/...`，调整 PATH 并执行 `hash -r`；不要使用 Windows npm 安装 Linux 依赖。

迁移 `.env` 时将 `D:/...` 改为实际 Linux 路径，数据库和索引建议使用当前副本的默认 `data/`。可用 `HF_HUB_CACHE` 指向已有完整权重缓存；首次下载保持 `HF_HUB_OFFLINE=0`、`TRANSFORMERS_OFFLINE=0`，离线开关只在缓存完整后启用。

### Bash 手动验证

按 [README 本地启动](README.md#本地启动)执行，在浏览器检查推荐、补充条件、实际候选对比及刷新后的消息恢复。2026-10-26，WSL Bash 前后端分离方式已手动验证通过。页面或 `/docs` 返回 200 只证明 HTTP 可达，聊天仍需模型 API、商品库和索引正常。

## Docker

WSL 使用 Docker Desktop 时，启用 `Settings → Resources → WSL Integration` 中的当前发行版，并重新打开终端。先停止本地前后端，再从项目根目录执行：

```bash
docker info
docker compose version
docker compose config --quiet
docker compose up --build
```

Compose 先运行 `init` 导入默认数据、加载嵌入模型并准备索引，再启动后端和前端。默认 CPU 运行；`TORCH_INDEX_URL` 可覆盖 PyTorch 安装源。初始化不调用聊天 API，后端健康检查访问 `/docs`，实际模型可用性由对话验证。

| 项目 | 说明 |
|---|---|
| 访问 | 前端 `localhost:5173`，API `localhost:8000/docs`；端口仅绑定本机 |
| 数据 | `./data` 绑定到 `/app/data`；容器路径不能沿用 `D:/...` 或 `/mnt/d/...` |
| 权重 | `huggingface_cache` 命名卷；本机缓存不会自动进入容器，勿直接沿用本机 `HF_HUB_CACHE` |
| 离线 | 首次保持 `HF_HUB_OFFLINE=0`；卷内权重完整后才可设为 `1` |
| 运行数据 | Linux 容器可能生成 root 属主文件；本地与 Docker 建议使用独立副本或数据目录 |
| Redis | 随 Compose 启动，检索缓存仍由 `REDIS_ENABLED` 控制，默认关闭 |

```bash
docker compose ps -a
docker compose logs -f init backend
docker compose down  # 保留 ./data 和模型缓存卷
```

数据或嵌入模型更新后，停止服务、显式重建索引再启动：

```bash
docker compose down
docker compose run --rm --no-deps init python -m backend.bootstrap --rebuild-index
docker compose up --build
```

下载失败时检查 Docker Desktop、WSL 客户端及构建容器各自的代理配置；构建代理可通过 [Docker CLI 的 proxies 或构建参数](https://docs.docker.com/engine/cli/proxy/)传入。WSL NAT 模式下，Windows 的 `127.0.0.1` 代理不可直接当作 Linux 本机代理。Socket 权限错误先重新打开 WSL 登录终端。

2026-10-25 已完成 WSL + Docker Desktop 的构建、首次权重下载、初始化、三轮对话和重启复用；当前源码待手动复核。历史报告保存在本机验证副本的 `data/wsl_validation/`。

## 其他运行入口

以下命令从仓库根目录执行，Python 使用项目虚拟环境：

| 入口 | 命令与说明 |
|---|---|
| 后端直接执行 | `python -m backend.main` 或 `python backend/main.py`；自带 reload，监听 `0.0.0.0:8000`，服务及目录 API 已验证 |
| 前端构建后预览 | `npm --prefix frontend run build` 后执行 `npm --prefix frontend run preview`；默认 4173，需另启后端，页面及 API 代理已验证 |
| 旧版 Streamlit | `python -m streamlit run src/ui/shopping_app.py`；默认 8501，仅 HTTP 服务与健康检查通过，交互和对话未验 |

验证结果对应执行时的源码，不代表全品类回答质量验收；独立 Linux / macOS 尚未实测。

## 数据与索引

默认目录包含 500 个型号、549 条配置，数据口径及限制见[目录说明](datasets/real_catalog/README.md)。默认双库均不存在时，`backend.bootstrap` 自动导入；部分、空或损坏的数据库明确报错，不覆盖已有数据。

```bash
python -m src.data.prepare_catalog --validate-only  # 只校验 JSONL，无需模型
python -m src.data.prepare_catalog                  # 导入默认双库
python -m backend.bootstrap                         # 准备或复用索引
python -m backend.bootstrap --rebuild-index          # 显式重建，成功后重启后端
```

自定义路径先运行 `python -m src.data.prepare_catalog --product-db data/custom/products.db`，再设置 `.env` 的 `PRODUCT_DB_PATH` 与 `PRODUCT_CHROMA_DIR` 并初始化；评价库固定为商品库同目录的 `product_reviews.db`。

索引按版本构建并在校验成功后切换活动清单，失败保留原索引。数据指纹、模型或集合不匹配时拒绝复用，需显式重建。旧清单缺少内容指纹，使用者须确认数据与模型一致。

维护 JSONL 后，更新清单哈希、数量与指纹，独立复核 `evals/datasets/catalog/` 标注，并运行 `python scripts/audit_catalog_quality.py --release-check`。发布新数据先在新目录准备双库与索引，再切换配置；缓存与过程报告留在忽略目录，不随仓库分发。

## 测试与评测

```bash
python -m pytest tests -q
python -m evals.runner --mode deterministic
npm --prefix frontend test
npm --prefix frontend run build
```

真实目录评测需先准备数据库、模型和索引：

```bash
python -m evals.retrieval_runner
python -m evals.catalog_smoke
python -m evals.catalog_smoke --live --output-dir evals/results/catalog-live
```

| 检查 | 范围 |
|---|---|
| 确定性 Agent | 25 个路由、工具边界、画像和终止案例；不需 API 或权重 |
| 真实目录 RAG | 76 条案例；记录 Recall@5、命中率、来源和约束违规；21 条精确配置须全命中，结构化硬条件、禁入和空结果违规须为零 |
| `catalog_smoke` | 真实数据库、检索、工具与图，使用受控模型验证七品类流程 |
| `catalog_smoke --live` | 真实 API 验证三新品类的多轮场景；会话、画像、Checkpoint 隔离到报告目录，回答仍需人工复核 |

缺少前置条件或数据、标注、索引不一致时，RAG 报告 `blocked`、退出码 `2`，不记为零分。原 30 条契约集使用独立目录，不能在真实目录上直接计分：

```bash
python -m evals.retrieval_runner --dataset-id retrieval-contract-v1 --manifest evals/datasets/contracts/manifest.json --cases evals/retrieval_cases.jsonl
```

## SSE 与运行指标

客户端以 SSE `done.answer` 作为最终回答并保存；存在结构化商品结果时，token 在调用结束和校验完成后发出。Verifier 检查显式商品 ID、配对价格及候选约束，不代表全部规格、体验或推荐理由均已核验。

响应及 JSON 日志提供运行 ID、模型调用、重试、用量、检索和工具指标；专用 telemetry 使用会话短哈希并脱敏凭据。LangSmith 仅在配置 `LANGSMITH_TRACING=true` 和密钥后启用，普通模块日志与外部 tracing 为独立通道。

## 启动排错

| 现象 | 处理 |
|---|---|
| `No module named uvicorn` | 使用当前系统的项目虚拟环境，Windows 直接用 `.venv\Scripts\python.exe` |
| WSL 缺少 Python / Node 或前端平台二进制 | 安装 Linux 环境，检查 PATH；在独立副本执行 `npm --prefix frontend ci` |
| 找不到模型 / 下载失败 | 检查网络、缓存路径和权重完整性；首次不要启用离线模式 |
| 双库或索引无效 | 核对路径和数据，按上方流程显式导入或重建，勿直接覆盖已有库 |
| 页面可见但无法聊天 | 确认 8000 后端运行，查看对应请求日志；HTTP 就绪不等于 Agent 就绪 |
| 401 / 403 / 模型或工具参数错误 | 核对 API 密钥、地址、模型和工具调用支持，修改后重启后端 |
| 前端端口变化 | 以 Vite 输出为准；本地与 Docker 不同时占用 8000 / 5173 |
| 推荐不符合预期 | 检查目录、预算、品类、排除品牌及活动索引；流程通过不代表回答质量达标 |
