# 开发与评测说明

[返回 README](README.md)

## 环境与配置

本地已验证 Python 3.12、Node.js 24。建议在仓库内创建隔离环境，再按 README 安装依赖：

```bash
python -m venv .venv
```

激活方式：Linux / macOS 使用 `source .venv/bin/activate`；Windows PowerShell 使用 `.\.venv\Scripts\Activate.ps1`。如果 PowerShell 的执行策略限制激活，可直接使用 `.\.venv\Scripts\python.exe` 运行 Python 命令，使用 `npm.cmd` 运行前端命令。

将根目录 `.env.example` 复制为 `.env`，配置 `OPENAI_API_KEY`、`OPENAI_BASE_URL` 和 `LLM_MODEL`。该接口使用 OpenAI-compatible 客户端，模型需支持主流程的工具调用。

主要配置在 [`src/config.py`](src/config.py)：

| 配置 | 用途 / 默认值 |
|---|---|
| `PRODUCT_DB_PATH` | 商品目录，默认 `data/products.db` |
| `PRODUCT_CHROMA_DIR` | 商品索引，默认 `data/product_chroma_db` |
| `EMBEDDING_MODEL` | 默认 `BAAI/bge-small-zh-v1.5`，需与构建索引时的模型一致 |
| `AGENT_CHECKPOINT_DB_PATH` | 图状态，默认 `data/agent_checkpoints.db` |
| `MAX_LLM_CALLS` / `MAX_TOTAL_TOKENS` | 每轮默认 6 次调用 / 12000 个已报告 Token |
| `MAX_RUN_SECONDS` / `AGENT_REQUEST_TIMEOUT_SECONDS` | 图运行 / 同步请求默认 55 / 60 秒 |
| `REDIS_ENABLED` | 默认 false；启用后配置 `REDIS_URL`，检索缓存默认 TTL 为 900 秒 |

Token 预算依赖服务商返回的用量；时间和取消检查是协作式控制，不能即时终止已发出的同步模型请求。

## 数据与索引

真实商品数据需自行维护，也可采用模拟数据观察 Agent 行为；后续将补充模拟数据以方便使用。当前没有内置商品数据或生成商品库的脚本；Docker 初始化只检查使用者提供的数据并准备索引。使用默认索引构建命令前，需准备以下 SQLite 文件；字段要求以[索引构建器](src/embeddings/product_embedder.py)和[商品检索器](src/retrieval/product_retriever.py)为准：

| 文件 / 表 | 所需字段 |
|---|---|
| `data/products.db` → `products` | `product_id, name, brand, category, subcategory, price, rating, sales_count, release_date, description, specs` |
| `data/product_reviews.db` → `product_reviews` | `review_id, product_id, sentiment, aspect, content` |

`specs` 为 JSON 对象文本，例如 `{"内存": "16GB", "重量": "1.4kg"}`；商品与评价通过整数 `product_id` 关联，整数 `review_id` 标识评价，二者应在各自表内唯一。`price`、`rating` 为数值，`sales_count` 为整数，其余描述字段使用文本。写入 Chroma 的元数据字段不能为 SQL `NULL`，应采用支持的标量类型。

可参考 [`tests/test_product_retriever.py`](tests/test_product_retriever.py) 中的最小商品夹具理解表结构；测试夹具并非可直接运行完整购物流程的演示数据集。模拟商品与评价应明确标注来源，避免将其作为真实选购信息。

```bash
python -m src.embeddings.product_embedder
```

此命令入口使用上述默认数据路径、`data/product_chroma_db` 和默认嵌入模型；自定义路径或模型需要调用 `build_product_embeddings(...)` 并同步运行时配置，不能仅修改环境变量后假定此入口也已切换。

构建器新建描述、规格、评价三个版本化集合，数量校验通过后才切换 `retrieval_manifest.json`。构建失败保留当前活动索引。数据更新后重新构建，并重启后端以加载活动版本。

主图在搜索、对比、推荐和异议处理阶段触发检索；缓存未命中时执行描述向量、规格向量和 SQLite FTS5 trigram 三路召回，经 RRF 融合后按商品目录权威字段过滤。偏好品牌仅加排序分，排除品牌是硬约束。评价只丰富已选候选，不参与商品主排序；FTS5 不可用时降级为双路向量召回，并记录 `sparse_fallback`。

## Agent 评测

```bash
python -m evals.runner --mode deterministic
```

25 个中文案例覆盖七类阶段、四类主工具、检索触发、画像候选、工具异常和轮次上限。指标包括阶段正确率、工具边界、停止原因、检索行为、非空答复率和延迟分位数。

运行器核对 [`evals/datasets/v1/manifest.json`](evals/datasets/v1/manifest.json) 中的案例数量及 SHA-256，并执行 `agent-deterministic-regression` 门禁；失败返回非零退出码。报告写入已忽略的 `evals/results/`，包含数据集版本、代码 revision、Prompt 哈希及稳定失败代码。

配置真实模型、商品库和索引后，可主动运行 Live 评测，产生模型调用费用：

```bash
python -m evals.runner --mode live --limit 5
python -m evals.runner --mode live --case-id search_explicit
```

各 Live 案例使用独立会话 ID，结束后清理 Checkpoint 和画像。当前评测考察可观测的执行行为，不以精确文本匹配衡量回答，也不据此宣称真实推荐质量。

## RAG 评测

```bash
python -m evals.retrieval_runner
```

30 个案例输出 `Recall@K`、命中率、来源覆盖率，以及禁入商品、约束和空结果违规数。案例绑定特定商品 ID，真实质量评测需要匹配的商品数据；不能直接把任意目录的得分视为项目检索水平。

运行前确认商品库具有 `products` 表，Chroma 目录存在活动描述 / 规格 / 评价集合，版本化索引的清单完整，嵌入模型与索引兼容。旧集合名为 `product_descriptions`、`product_specs` 和 `product_reviews`；版本化集合按活动清单解析。

前置检查在加载嵌入模型之前执行。缺少数据库或索引时报告为 `blocked`、返回退出码 `2`，质量指标不记为零；条件满足后执行 `retrieval-safety` 门禁。仓库不预设线上 Recall 提升，指标以目标数据实测为准。

## SSE 与运行指标

`done.answer` 是图完成后的最终回答，客户端应以它替换预览并保存历史。存在结构化商品检索结果时，token 缓冲到本次调用完成和校验结束后再发出；工具请求前说明被丢弃，校验失败只发出安全回答，其他对话保持即时 token。

现有 verifier 检查显式商品 ID、可配对价格和候选约束。缺少可验证的显式事实时可能返回 `not_applicable`；它不验证所有自然语言规格、评价或优惠信息。

同步响应及 SSE `done` 返回运行 ID、模型调用 / 重试、可用 Token 用量、检索 / 缓存状态、召回命中和工具执行指标。专用 telemetry 使用 JSON 日志、会话短哈希和凭据脱敏，不记录原始对话 / 画像；普通模块日志及外部 tracing 是独立通道。

LangSmith 默认关闭，仅在配置 `LANGSMITH_TRACING=true` 与 `LANGSMITH_API_KEY` 时启用。未配置或初始化失败不影响主链路。

## 启动排错

| 现象 | 检查与处理 |
|---|---|
| 索引构建提示数据库无法打开或表不存在 | 确认已准备上述两个数据库及对应表；默认构建命令使用仓库内固定路径，不读取自定义路径环境变量 |
| 嵌入模型下载或加载失败 | 确认能访问模型源、所需权重已完整缓存；离线环境需提前准备同名模型，首次构建不要设置 `HF_HUB_OFFLINE=1` |
| 页面能打开，但无法创建会话或发送消息 | 先访问 `http://localhost:8000/docs`，确认后端运行在 8000 端口，再查看后端错误日志；前端默认代理指向该端口 |
| 首次对话出现 401、403 或模型不存在 | 核对根目录 `.env` 的密钥、API 地址、模型名及服务商访问权限，修改后重启后端 |
| 模型拒绝工具调用参数 | 确认接口兼容 OpenAI 工具调用，且所选模型支持工具调用 |
| 未返回预期商品 | 核对目录数据、预算 / 品类 / 排除品牌条件和活动索引；商品数据更新后重新构建索引并重启后端 |
| 前端显示的地址不是 5173 端口 | 本地 Vite 可能因端口占用选择其他端口，可释放 5173 后重试；默认 `/api` 代理采用同源访问，只有浏览器直接跨域请求后端时才需配置 CORS |

后端启动成功不代表模型、数据和索引均可用：Agent 在首次对话时初始化。排查聊天错误时以对应请求的后端日志为准。

## Docker

安装 Docker 与 Compose V2，按 README 配置 `.env` 并准备两个商品 / 评价 SQLite 文件后，在仓库根目录执行 `docker compose up --build`。无需安装本机 Python / Node.js，也无需预先手动构建默认索引。

[`backend/bootstrap.py`](backend/bootstrap.py) 作为 `init` 服务运行：只读检查数据库、拒绝占位密钥、加载配置中的嵌入模型、首次构建索引或复用已有索引。初始化不调用模型 API，不生成商品数据，不修改源数据库，也不删除已有索引。无效清单、集合缺失、数量不一致或清单模型不匹配时退出并提示检查 / 显式重建；旧版集合可复用，但其嵌入模型仍需使用者确认一致。

后端通过 `service_completed_successfully` 等待初始化完成，前端通过 `service_healthy` 等待后端 HTTP 就绪（[Compose 启动顺序](https://docs.docker.com/compose/how-tos/startup-order/)）。该 HTTP 检查访问 `/docs`，不调用模型 API，也不代表 API 密钥已验证。前端容器的 `API_PROXY_TARGET` 指向 `http://backend:8000`，本地 Vite 仍默认使用 `http://localhost:8000`。

Compose 使用 `.env` 传入模型配置、运行预算与 tracing 设置；Redis 服务随 Compose 启动，检索缓存是否启用由 `REDIS_ENABLED` 决定，默认 false。服务端口仅绑定 `127.0.0.1`，供本机体验；对外部署需自行配置认证、代理与访问控制。

首次构建需要下载依赖和权重。默认 `HF_HUB_OFFLINE=0` 允许下载，并复用 `huggingface_cache` 命名卷；如需离线使用，需预先将对应模型完整缓存到该卷，再设置 `HF_HUB_OFFLINE=1`。宿主机已有的缓存不会自动出现在容器卷中。

常用操作：

```bash
# 查看服务和初始化日志
docker compose ps -a
docker compose logs init backend

# 停止并移除容器，保留 ./data 与命名卷
docker compose down
```

商品数据更新或需要切换嵌入模型时，先停止服务，再显式构建新索引版本，成功后重新启动：

```bash
docker compose down
docker compose run --rm --no-deps init python -m backend.bootstrap --rebuild-index
docker compose up --build
```

已有索引不会因数据库内容变化而自动重建；上述命令使用运行时配置中的商品路径、Chroma 路径和嵌入模型，评价库位于商品库同目录的 `product_reviews.db`。自定义容器路径必须位于实际挂载的目录内；不要直接把宿主机的绝对路径填入容器配置。

当前开发环境未安装 Docker，已通过本地初始化回归、前端代理与构建检查；尚未完成真实容器镜像构建、首次权重下载和完整对话验收。
