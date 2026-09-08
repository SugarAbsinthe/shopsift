# ShopSift

ShopSift 是基于 LangGraph 的智能购物助手。用户用自然语言描述需求，ShopSift 自动挖掘偏好、检索产品、对比推荐，并在同一会话中持久化图状态与用户画像。

```
 React 前端 ──真实 Token SSE──→ FastAPI ──→ Agent (单例)
                                  ├─ 阶段分析 → 规则 + LLM 七阶段分类
                                  ├─ 商品检索 → 描述/规格向量 + FTS5 + RRF + 硬约束
                                  ├─ 推理决策 → per-stage Prompt 动态注入 + 5 工具
                                  └─ 工具调用 → 工厂模式依赖注入
                                  ├─ 图状态 → LangGraph SQLite Checkpoint
                                  └─ 会话画像 → SQLite KV + 置信度时间衰减
                                  ├─ 可观测性 → JSON 日志 + 运行/Token/工具/检索指标
                                  └─ 评测 → 25 个 Agent 案例 + 30 个检索案例
```

## 快速开始

```bash
# Docker
docker compose up

# 手动
pip install -e ".[dev]"
python -m uvicorn backend.main:app --port 8000
cd frontend && npm install && npm run dev
```

浏览器打开 `http://localhost:5173`。

## 项目结构

```
src/agent/       LangGraph 编排、确定性商品召回、per-stage Prompt、5 工具
src/retrieval/   描述/规格向量、FTS5、RRF 融合与结构化约束
src/profile/     会话画像存储（主链路使用 SQLite，保留语义记忆接口）
src/cache/       Redis RAG 缓存
backend/         FastAPI REST + SSE 流式
frontend/        React + TypeScript
evals/           Agent/RAG 评测契约、案例集与运行器
tests/           单元、路由、状态、流式、API 契约与评测回归
```

## Agent 评测

确定性模式使用真实 LangGraph 编排和本地受控依赖，不访问模型、Redis、
ChromaDB 或外网，可作为日常回归入口：

```bash
python -m evals.runner --mode deterministic
```

内置 25 个中文案例，覆盖七类会话阶段、五类主 Agent 工具、检索触发、画像提取、
工具异常和轮次上限。命令输出阶段正确率、工具边界、停止原因、检索行为、
非空答复率及延迟分位数，JSON/Markdown 报告生成到已忽略的
`evals/results/`。

Live 模式仅在主动配置模型密钥、商品 SQLite 和 ChromaDB 数据后运行，
会产生真实模型调用费用，不作为默认测试门禁：

```bash
python -m evals.runner --mode live --limit 5
python -m evals.runner --mode live --case-id search_explicit
```

每个 Live 案例使用独立会话 ID，并在结束后清理 Checkpoint 和画像数据。
回答质量不使用精确文本匹配；当前指标只评价可客观观测的路由、工具、
检索、终止和非空答复。

## RAG 检索与评测

商品检索每次独立执行描述向量、规格向量和 SQLite FTS5 trigram 三路召回，
通过 RRF 融合名次，随后从商品目录补齐权威字段，并对预算、品类和排除品牌
执行硬过滤。评价只用于丰富最终候选，不参与商品主排序。FTS5 不可用时会
降级为双路向量召回，并在聚合指标中标记。

索引采用版本化集合；只有三类集合写入完成且数量校验通过后，才原子切换
`retrieval_manifest.json`。构建失败不会删除或切换当前活动索引：

```bash
python -m src.embeddings.product_embedder
```

配置本地商品 SQLite、活动 ChromaDB 索引和嵌入模型后，可运行 30 条 RAG
案例，输出 `Recall@K`、命中率、来源覆盖率、禁入商品、约束和空结果违规数：

```bash
python -m evals.retrieval_runner
```

案例集是确定性的质量契约；仓库不附带真实商品数据和索引，因此 README
不预设或宣称线上 Recall 提升，指标应以目标环境实测结果为准。

## 可观测性

同步响应和 SSE `done` 事件统一返回 `run_id`、耗时、LLM 调用/重试、
可用的 Token 用量、检索/缓存状态、各召回路命中与过滤统计、工具请求与执行结果。后端日志为单行
JSON，原始对话和画像不进入日志，常见凭据模式会被脱敏，会话 ID 仅记录
不可逆短哈希。

LangSmith 默认关闭；只有同时配置 `LANGSMITH_TRACING=true` 和
`LANGSMITH_API_KEY` 时才启用。未配置或初始化失败不会影响 Agent 主链路。

## License

MIT
