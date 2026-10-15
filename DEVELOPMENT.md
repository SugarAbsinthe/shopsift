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
| `LLM_MAX_OUTPUT_TOKENS` | 单次模型请求输出上限，默认 4096 |
| `MAX_LLM_CALLS` / `MAX_TOTAL_TOKENS` | 每轮默认 10 次调用 / 32768 个已报告 Token |
| `MAX_RUN_SECONDS` / `AGENT_REQUEST_TIMEOUT_SECONDS` | 图运行 / 同步请求默认 120 / 150 秒 |
| `REDIS_ENABLED` | 默认 false；启用后配置 `REDIS_URL`，检索缓存默认 TTL 为 900 秒 |

Token 预算依赖服务商返回的用量；时间和取消检查是协作式控制，不能即时终止已发出的同步模型请求。

## 数据与索引

固定目录 [`datasets/real_catalog/`](datasets/real_catalog/README.md) 包含500个不同型号、549条配置、107条体验适用关联及69个独立摘要。直接更新该目录，历史交由Git管理；清单保留数据批次与指纹，用于核对JSONL、SQLite和Chroma。七类均衡扩充，价格为人民币快照，缺失规格保持未知。数据规模与品牌分布门禁通过，体验和真实多轮对话仍为试点，详见[目录说明](datasets/real_catalog/README.md)及[验收标准](data_acceptance_criteria.md)。数据库、索引、采集缓存和过程报告均为本地产物。
```bash
# 不需要 API 密钥、联网或嵌入模型
python -m src.data.prepare_catalog --validate-only
python -m src.data.prepare_catalog

# 完整运行初始化，读取 .env 的路径与嵌入模型配置
python -m backend.bootstrap
```

默认商品库及同目录评价库如下；核心表契约保持兼容：

| 文件 / 表 | 所需字段 |
|---|---|
| `data/products.db` → `products` | `product_id, name, brand, category, subcategory, price, rating, sales_count, release_date, description, specs` |
| `data/product_reviews.db` → `product_reviews` | `review_id, product_id, sentiment, aspect, content` |

`specs` 为 JSON 对象文本，例如 `{"内存": "16GB", "重量": "1.4kg"}`；整数 `product_id` / `review_id` 随 JSONL 固定保存。商品库附加 `product_provenance`、`catalog_sources`，评价库附加 `review_provenance`，记录稳定配置、型号、字段来源和摘要适用范围。旧库缺少来源表时仍能读取。未知评分、销量及发布日期保持空值，构建 Chroma 时省略空元数据，检索、详情与对比显示“未知”；体验摘要不能转化为平台评分或销量。

`prepare_catalog` 校验格式、稳定标识、来源与摘要引用、数量、覆盖率和 SHA-256，在临时位置构建并校验双库后发布。异常时仅回滚自身未被修改的输出；中断通过导入状态记录恢复，部分数据不进入索引。相同目录重复导入复用已有双库；不同或损坏的已有双库拒绝覆盖和合并。

```bash
python -m src.data.prepare_catalog --dataset-dir datasets/real_catalog --product-db data/custom/products.db
# .env 中设置 PRODUCT_DB_PATH=data/custom/products.db 和 PRODUCT_CHROMA_DIR=data/custom/chroma
python -m backend.bootstrap
```

`backend.bootstrap` 只在默认路径下两个数据库均不存在时自动导入。双库均存在时只读校验并保留；仅一个存在、核心表为空或数据损坏时明确报错。自定义路径须先显式导入；评价库始终为商品库同目录的 `product_reviews.db`。完整初始化要求有效格式的模型配置，但不调用聊天模型 API，实际凭据有效性由对话验证。

构建器新建描述、规格、评价三个版本化集合，数量与目录指纹校验通过后才切换 `retrieval_manifest.json`。构建失败保留当前活动索引。新清单记录双库的逻辑内容 SHA-256；来源、商品或体验数据变化后，初始化、检索器和 RAG 评测拒绝复用过期索引，提示 `python -m backend.bootstrap --rebuild-index`。旧清单继续兼容；它们缺少内容指纹，需使用者确认数据与模型一致。

主图在搜索、对比、推荐和异议处理阶段触发检索。预算、品类、排除品牌及注册规格条件先从SQLite计算合格ID，再应用到每条向量与FTS召回通道，避免全库截断遗漏合格商品。`RetrievalConstraints.spec_filters` 只接受注册字段和 `eq/gte/lte/contains_all` 运算；数值统一单位，未知值不能满足硬要求。FTS不可用时仍保留资格过滤。泛推荐先展示不同 `model_key`；型号足够时前五至少三个型号，精确型号查询和版本对比可显示多个配置。偏好品牌只影响排序，排除品牌是硬条件，体验不参与商品主排序。

品类、别名、关键字段、筛选字段和前端示例共用 `src/data/categories.py`。购买外设时优先识别键盘、鼠标或显示器，避免被MacBook等兼容设备名称带偏。详情、对比和体验工具将实际读取的结构化事实通过内部artifact交给校验器，保留工具名称及参数；成功查到但未在初始召回出现的商品可以核验，虚构ID、错误价格及硬约束仍拦截。模型上下文只保留最近三轮及本轮完整工具事实，旧轮工具正文缩短但完整Checkpoint保留；无合格外设时直接解释条件与未知值，避免编造备选。
规格要求支持跨轮追加、替换与撤销；否定、疑问和二选一表达不转成正向硬条件。旧品类可从明确带单位的内存、存储、屏幕尺寸及明确的显卡类型生成筛选值；最高支持容量、可选配置或多值范围不作为具体配置值。未被现有运算支持的负向规格意图保留为画像文字，不宣称已实施对应硬过滤。显式传入的结构化硬条件优先于查询文本推断。`quality.py`中的早期试点数量规则仅用于历史策略兼容，当前目录按`release.py`及清单中的现行策略评估。使用`audit_catalog_quality.py --release-check --source-cache ...`时，证据缓存缺失或哈希不匹配也会返回失败。

## Agent 评测

```bash
python -m evals.runner --mode deterministic
```

25 个中文案例覆盖七类阶段、四类主工具、检索触发、画像候选、工具异常和轮次上限。指标包括阶段正确率、工具边界、停止原因、检索行为、非空答复率和延迟分位数。

运行器核对 [`evals/datasets/contracts/manifest.json`](evals/datasets/contracts/manifest.json) 中的案例数量及 SHA-256，并执行 `agent-deterministic-regression` 门禁；失败返回非零退出码。报告写入已忽略的 `evals/results/`，包含数据集版本、代码 revision、Prompt 哈希及稳定失败代码。

配置真实模型、商品库和索引后，可主动运行 Live 评测，产生模型调用费用：

```bash
python -m evals.runner --mode live --limit 5
python -m evals.runner --mode live --case-id search_explicit
```

各 Live 案例使用独立会话 ID，结束后清理 Checkpoint 和画像。当前评测考察可观测的执行行为，不以精确文本匹配衡量回答，也不据此宣称真实推荐质量。

维护当前JSONL及清单后，运行 `python -m src.data.prepare_catalog --validate-only` 和 `python scripts/audit_catalog_quality.py --release-check`。按目录事实独立复核 `evals/datasets/catalog/retrieval_cases.jsonl`，同步其清单哈希与目录指纹，再显式导入新双库、重建索引。采集脚本、未采用候选、原文缓存和过程报告保存在被忽略的本地 `data/`，不随仓库发布；商品来源及字段定位保留在正式JSONL中。

## RAG 评测

```bash
python -m evals.retrieval_runner
```

默认运行固定目录 [`evals/datasets/catalog/retrieval_cases.jsonl`](evals/datasets/catalog/retrieval_cases.jsonl) 的76条案例（`retrieval-real-catalog-v4`）：原四类各10条，外设各12条，标注按当前目录事实独立确定，空结果预算为1元。清单绑定目录版本与逻辑哈希，报告记录案例SHA-256、嵌入模型、活动索引、实际Recall@5和命中率。新品硬规格通过结构化谓词验收；旧四类的语义规格通过金标准召回衡量，不表示已全部强制过滤。`source_coverage` 是召回通道覆盖率，`provenance_coverage` 是商品来源覆盖率。`forbidden_ids` 只标注结构化硬条件违规，自然语言规格通过相关集合衡量。

已有数据库不自动覆盖或合并。迁移先导入 `python -m src.data.prepare_catalog --product-db data/catalog_release/products.db`，将`.env`的`PRODUCT_DB_PATH`设为该路径、`PRODUCT_CHROMA_DIR`设为`data/catalog_release/chroma`，运行`python -m backend.bootstrap`并重启服务。其他配置与会话保持原样；评测只使用当前目录，双库或索引不匹配时报告`blocked`。
保留原有 30 条契约评测，显式运行：

```bash
python -m evals.retrieval_runner --dataset-id retrieval-contract-v1 --manifest evals/datasets/contracts/manifest.json --cases evals/retrieval_cases.jsonl
python -m evals.catalog_smoke
```

原契约集的 ID 与真实目录不同，需匹配其原始目录，不能将它在真实目录上的分数作为本轮质量指标。`catalog_smoke` 使用真实 SQLite、Chroma、只读工具和 Agent 图，配合受控本地模型验证七类推荐/对比/体验流程及中文品牌排除，不评估远程模型回答质量。

运行前确认商品库具有 `products` 表，Chroma 目录存在活动描述 / 规格 / 评价集合，版本化索引的清单完整，嵌入模型与索引兼容。旧集合名为 `product_descriptions`、`product_specs` 和 `product_reviews`；版本化集合按活动清单解析。

前置检查在加载嵌入模型之前执行。缺少数据库/索引、目录与标注或索引指纹不一致、嵌入模型不匹配时报告 `blocked`、退出码 `2`，质量指标不记为零。真实集门禁要求21条精确配置全部命中、商品来源完整，结构化硬条件、禁入及空结果违规为零；其余案例报告实际表现及诊断，详见[本版验收结果](datasets/real_catalog/README.md)。25条确定性Agent案例保留独立门禁。

真实服务场景使用 `python -m evals.catalog_smoke --live --output-dir evals/results/catalog-live`，调用配置中的真实模型，经FastAPI聊天接口验证三新品类各三个六轮场景；画像、会话与Checkpoint隔离到报告目录。检查推荐、追加硬条件、排除及撤销品牌、对比与体验工具、未知或空结果，并保存各轮结构化条件。受控模型测试不能替代这项，真实回答仍须人工复核；Docker须在具备Docker的环境验收。

## SSE 与运行指标

`done.answer` 是图完成后的最终回答，客户端应以它替换预览并保存历史。存在结构化商品检索结果时，token 缓冲到本次调用完成和校验结束后再发出；工具请求前说明被丢弃，校验失败只发出安全回答，其他对话保持即时 token。

现有 verifier 检查显式商品 ID、可配对价格和候选约束。缺少可验证的显式事实时可能返回 `not_applicable`；它不验证所有自然语言规格、评价或优惠信息。

同步响应及 SSE `done` 返回运行 ID、模型调用 / 重试、可用 Token 用量、检索 / 缓存状态、召回命中和工具执行指标。专用 telemetry 使用 JSON 日志、会话短哈希和凭据脱敏，不记录原始对话 / 画像；普通模块日志及外部 tracing 是独立通道。

LangSmith 默认关闭，仅在配置 `LANGSMITH_TRACING=true` 与 `LANGSMITH_API_KEY` 时启用。未配置或初始化失败不影响主链路。

## 启动排错

| 现象 | 检查与处理 |
|---|---|
| 数据库缺失、为空或损坏 | 默认双库均不存在会导入内置数据；部分、空或损坏的库需显式修复，初始化不会覆盖；自定义路径先运行 `prepare_catalog --product-db ...` |
| 目录内容或嵌入模型与索引不一致 | 检查当前数据和配置，运行 `python -m backend.bootstrap --rebuild-index` 后重启后端 |
| 嵌入模型下载或加载失败 | 确认能访问模型源、所需权重已完整缓存；离线环境需提前准备同名模型，首次构建不要设置 `HF_HUB_OFFLINE=1` |
| 页面能打开，但无法创建会话或发送消息 | 先访问 `http://localhost:8000/docs`，确认后端运行在 8000 端口，再查看后端错误日志；前端默认代理指向该端口 |
| 首次对话出现 401、403 或模型不存在 | 核对根目录 `.env` 的密钥、API 地址、模型名及服务商访问权限，修改后重启后端 |
| 模型拒绝工具调用参数 | 确认接口兼容 OpenAI 工具调用，且所选模型支持工具调用 |
| 未返回预期商品 | 核对目录数据、预算 / 品类 / 排除品牌条件和活动索引；商品数据更新后重新构建索引并重启后端 |
| 前端显示的地址不是 5173 端口 | 本地 Vite 可能因端口占用选择其他端口，可释放 5173 后重试；默认 `/api` 代理采用同源访问，只有浏览器直接跨域请求后端时才需配置 CORS |

后端启动成功不代表模型、数据和索引均可用：Agent 在首次对话时初始化。排查聊天错误时以对应请求的后端日志为准。

## Docker

安装 Docker 与 Compose V2，按 README 配置 `.env` 后，在仓库根目录执行 `docker compose up --build`。镜像包含当前JSONL，无需手工准备默认双库或索引。

[`backend/bootstrap.py`](backend/bootstrap.py) 作为 `init` 服务运行：拒绝占位密钥，默认双库均缺失时导入内置数据，然后只读校验、加载配置中的嵌入模型、首次构建索引或复用已有索引。初始化不调用聊天模型 API，不重新采集网站数据，不覆盖已有双库，也不删除已有索引。无效清单、过期指纹、集合缺失、数量不一致或清单模型不匹配时退出并提示显式重建；旧版集合保持兼容。

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
