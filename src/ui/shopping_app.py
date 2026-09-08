"""Streamlit UI for the Shopping Guide Agent.

GPT-style chat layout with:
  - Sidebar: model select + real-time profile panel + stage indicator
  - Chat: messages with Markdown rendering
  - Debug expander: profile snapshot, retrieval query, tool call chain
"""
import json
import os
import sys
from pathlib import Path

# Prevent HuggingFace connection timeout — model is cached locally
os.environ["HF_HUB_OFFLINE"] = "1"

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import config, init_langsmith
from src.db.conversation_store import ConversationStore

init_langsmith()

# ---- Model presets ----
MODEL_PRESETS = {
    "DeepSeek V3": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
    },
    "DeepSeek R1": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-reasoner",
    },
    "Qwen Plus": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
    },
    "Qwen Max": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-max",
    },
    "GLM-4": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4-flash",
    },
    "自定义": {
        "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
    },
}

STAGE_LABELS = {
    "discovery": "发现需求",
    "needs_elicitation": "需求挖掘",
    "search": "产品搜索",
    "comparison": "产品对比",
    "objection_handling": "异议处理",
    "recommendation": "最终推荐",
    "summary": "总结收尾",
}

STAGE_EMOJI = {
    "discovery": "👋",
    "needs_elicitation": "🔍",
    "search": "🛒",
    "comparison": "⚖️",
    "objection_handling": "💬",
    "recommendation": "🎯",
    "summary": "✅",
}

CONV_STORE = ConversationStore(str(BASE_DIR / "data" / "conversations.db"))


# ---- Page config ----
st.set_page_config(page_title="ShopSift · 智能购物助手", page_icon="🛒", layout="wide")


# ---- Session State ----
def init_session():
    defaults = {
        "conv_id": None,
        "agent": None,
        "llm_ok": False,
        "agent_loading": False,
        "selected_preset": "DeepSeek V3",
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "max_tool_rounds": 3,
        "show_debug": False,
        "last_stage": "discovery",
        "last_profile": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()


# ---- Helpers ----
def load_messages(conv_id: str) -> list[dict]:
    db_msgs = CONV_STORE.get_messages(conv_id)
    return [
        {"role": m["role"], "content": m["content"], "details": m.get("details")}
        for m in db_msgs
    ]


def build_chat_history(conv_id: str) -> list:
    from langchain_core.messages import HumanMessage, AIMessage
    msgs = CONV_STORE.get_messages(conv_id)
    history = []
    for m in msgs:
        if m["role"] == "user":
            history.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            history.append(AIMessage(content=m["content"]))
    return history[-20:]


@st.cache_resource
def load_embedding_components():
    """Load ProductRetriever and ProfileStore once (cached)."""
    from src.retrieval.product_retriever import ProductRetriever
    from src.profile.profile_store import ProfileStore

    cache = get_rag_cache()

    retriever = ProductRetriever(
        chroma_dir=config.PRODUCT_CHROMA_DIR,
        embedding_model=config.EMBEDDING_MODEL,
        catalog_db=config.PRODUCT_DB_PATH,
        cache=cache,
    )
    profile_store = ProfileStore(
        db_path=config.PROFILE_DB_PATH,
        chroma_dir=config.PROFILE_CHROMA_DIR,
        embedding_model=config.EMBEDDING_MODEL,
    )
    return retriever, profile_store


def get_rag_cache():
    """Get or create the RAG cache instance (not cached_resource — holds a connection)."""
    from src.cache.rag_cache import RAGCache
    if "rag_cache" not in st.session_state:
        st.session_state.rag_cache = RAGCache(
            redis_url=config.REDIS_URL,
            ttl=config.RAG_CACHE_TTL,
        ) if config.REDIS_ENABLED else None
    return st.session_state.rag_cache


def init_agent():
    """Create or recreate the agent with current settings."""
    if not st.session_state.api_key:
        return False

    st.session_state.agent_loading = True
    try:
        from src.agent.shopping_agent import ShoppingGuideAgent
        from src.config import create_llm

        retriever, profile_store = load_embedding_components()

        preset = MODEL_PRESETS.get(st.session_state.selected_preset, MODEL_PRESETS["自定义"])

        llm = create_llm(
            api_key=st.session_state.api_key,
            base_url=preset["base_url"],
            model=preset["model"],
        )

        st.session_state.agent = ShoppingGuideAgent(
            llm=llm,
            product_retriever=retriever,
            profile_store=profile_store,
            max_tool_rounds=st.session_state.max_tool_rounds,
        )
        st.session_state.llm_ok = True
        st.session_state.agent_loading = False
        return True
    except Exception as e:
        st.session_state.llm_ok = False
        st.session_state.agent_loading = False
        st.error(f"Agent 初始化失败: {e}")
        return False


def render_profile_panel(profile_str: str):
    """Render the user profile as confidence-bar cards."""
    if not profile_str or profile_str == "(暂无画像)":
        st.caption("暂无用户画像")
        st.caption("开始对话后，Agent 会自动提取偏好")
        return

    lines = profile_str.strip().split("\n")
    for line in lines:
        line = line.strip("- ")
        if ": " not in line:
            continue
        key, rest = line.split(": ", 1)
        if " (置信度 " not in rest:
            continue
        value, conf_part = rest.rsplit(" (置信度 ", 1)
        conf_pct = conf_part.rstrip("%)")

        key_label = {
            "budget": "预算", "primary_use": "用途", "preferred_brand": "偏好品牌",
            "mobility": "便携需求", "must_have": "刚需", "exclude_brand": "排除品牌",
            "screen_preference": "屏幕偏好", "battery_requirement": "续航要求",
            "product_category": "产品品类",
        }.get(key, key)

        st.caption(f"**{key_label}**: {value}")
        try:
            pct = int(conf_pct)
            st.progress(pct / 100, text=f"置信度 {pct}%")
        except Exception:
            pass


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.title("🛒 ShopSift")
    st.caption("智能购物助手")

    # --- New Chat ---
    if st.button("＋ 新对话", use_container_width=True):
        new_id = CONV_STORE.create_conversation(
            model=MODEL_PRESETS.get(st.session_state.selected_preset, {}).get("model", "")
        )
        st.session_state.conv_id = new_id
        st.session_state.agent = None
        st.session_state.llm_ok = False
        st.session_state.last_stage = "discovery"
        st.session_state.last_profile = ""
        st.rerun()

    st.divider()

    # --- Stage Indicator ---
    st.subheader("当前阶段")
    stage = st.session_state.get("last_stage", "discovery")
    emoji = STAGE_EMOJI.get(stage, "")
    label = STAGE_LABELS.get(stage, stage)
    st.info(f"{emoji} **{label}**")

    st.divider()

    # --- Profile Panel ---
    st.subheader("用户画像")
    profile = st.session_state.get("last_profile", "")
    render_profile_panel(profile)

    st.divider()

    # --- Model & Settings ---
    with st.expander("模型 & 设置"):
        preset_names = list(MODEL_PRESETS.keys())
        st.selectbox(
            "预设模型",
            preset_names,
            index=preset_names.index(st.session_state.selected_preset)
            if st.session_state.selected_preset in preset_names else 0,
            key="selected_preset",
            on_change=lambda: setattr(st.session_state, "agent", None)
            or setattr(st.session_state, "llm_ok", False),
        )
        preset = MODEL_PRESETS.get(st.session_state.selected_preset, MODEL_PRESETS["自定义"])
        st.caption(f"Model: {preset['model']}")

        st.text_input(
            "API Key",
            key="api_key",
            type="password",
            help="OpenAI-compatible API key",
        )

        st.slider("最大工具轮次", 1, 5, key="max_tool_rounds")
        st.checkbox("显示调试信息", key="show_debug")

    st.divider()

    # --- Agent Status ---
    if st.session_state.llm_ok:
        st.success("Agent 已就绪")
    elif st.session_state.agent_loading:
        st.spinner("Agent 初始化中...")
    else:
        st.warning("Agent 未初始化")
        if st.button("🔌 初始化 Agent", use_container_width=True):
            init_agent()
            st.rerun()

    st.divider()

    # --- History ---
    st.subheader("历史对话")
    conversations = CONV_STORE.list_conversations()
    for conv in conversations:
        cid = conv["id"]
        title = conv["title"] or "新对话"
        is_active = cid == st.session_state.conv_id

        col1, col2 = st.columns([5, 1])
        with col1:
            label = f"{'▸ ' if is_active else ''}{title}"
            if st.button(label, key=f"conv_{cid}", use_container_width=True,
                         type="primary" if is_active else "secondary"):
                st.session_state.conv_id = cid
                st.session_state.agent = None
                st.session_state.llm_ok = False
                st.rerun()
        with col2:
            if st.button("🗑", key=f"del_{cid}", help="删除此对话"):
                CONV_STORE.delete_conversation(cid)
                if st.session_state.conv_id == cid:
                    st.session_state.conv_id = None
                st.rerun()

    if not conversations:
        st.caption("暂无历史对话")


# ============================================================
# Main Chat Area
# ============================================================

st.title("🛒 ShopSift 智能购物助手")
st.caption("告诉我你的需求和预算，我帮你找到最合适的产品。目前产品库以 3C 数码（笔记本）为主，更多品类持续扩展中。")

# Create initial conversation if needed
if not st.session_state.conv_id:
    new_id = CONV_STORE.create_conversation(
        model=MODEL_PRESETS.get(st.session_state.selected_preset, {}).get("model", "")
    )
    st.session_state.conv_id = new_id

conv_id = st.session_state.conv_id
messages = load_messages(conv_id)

# Display messages
for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("details") and st.session_state.show_debug:
            detail = msg["details"]
            if isinstance(detail, str):
                try:
                    detail = json.loads(detail)
                except Exception:
                    pass
            with st.expander("调试信息", expanded=False):
                st.json(detail)

# Quick-start chips for empty conversations
if not messages:
    st.markdown("### 💡 试试这样问我：")
    chips = [
        "预算8000左右，主要打游戏，偶尔出差",
        "我是设计专业学生，推荐什么笔记本？",
        "联想和惠普的游戏本哪个好？",
        "想买个适合编程的轻薄本，续航要好",
    ]
    cols = st.columns(2)
    for i, chip in enumerate(chips):
        with cols[i % 2]:
            if st.button(chip, key=f"chip_{i}", use_container_width=True):
                CONV_STORE.add_message(conv_id, "user", chip)
                st.rerun()

# Chat input
if question := st.chat_input("说说你的需求..."):
    CONV_STORE.add_message(conv_id, "user", question)
    with st.chat_message("user"):
        st.markdown(question)

    # Init agent if needed
    if not st.session_state.llm_ok:
        with st.spinner("Agent 初始化中（首次加载约 5 秒）..."):
            init_agent()

    if not st.session_state.llm_ok:
        st.error("Agent 未初始化 — 请在侧边栏填入 API Key 后点击「初始化 Agent」，或检查模型配置")
        st.stop()

    chat_history = build_chat_history(conv_id)

    with st.chat_message("assistant"):
        try:
            agent = st.session_state.agent
            stage_label = STAGE_LABELS.get(st.session_state.get("last_stage", "discovery"), "分析中")

            with st.spinner(f"Agent 正在处理（{stage_label}）..."):
                result = agent.run(question, conv_id=conv_id, chat_history=chat_history)

            answer = result.get("answer", "未能生成回答")
            st.markdown(answer)

            # Update session state
            st.session_state.last_stage = result.get("stage", "discovery")
            st.session_state.last_profile = result.get("user_profile", "")

            # Debug expander
            with st.expander("🔍 调试详情: 阶段 → 画像 → 检索 → 工具调用"):
                run_id = result.get("run_id", "")
                if run_id:
                    st.caption(
                        f"Run {run_id[:12]} · {result.get('latency_ms', 0)} ms · "
                        f"LLM {result.get('llm_calls', 0)} 次"
                    )

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("导购阶段", STAGE_LABELS.get(result.get("stage", ""), result.get("stage", "?")))
                with col_b:
                    st.metric("工具调用轮次", result.get("tool_rounds", 0))
                with col_c:
                    has_products = bool(result.get("product_context"))
                    st.metric("产品检索", "已触发" if has_products else "未触发")

                tab1, tab2, tab3 = st.tabs(["用户画像", "产品检索结果", "工具调用链"])

                with tab1:
                    profile_str = result.get("user_profile", "")
                    if profile_str and profile_str != "(暂无画像)":
                        st.markdown("### 当前用户画像")
                        st.markdown(profile_str)
                    else:
                        st.caption("暂无画像数据")

                with tab2:
                    ctx = result.get("product_context", "")
                    if ctx:
                        st.markdown(ctx)
                    else:
                        st.caption("本轮未触发产品检索（可能在挖掘需求阶段）")

                with tab3:
                    messages_all = result.get("messages", [])
                    from langchain_core.messages import AIMessage as AIm
                    tool_msgs = [
                        m for m in messages_all
                        if isinstance(m, AIm) and hasattr(m, "tool_calls") and m.tool_calls
                    ]
                    if tool_msgs:
                        for i, tm in enumerate(tool_msgs):
                            for tc in tm.tool_calls:
                                st.markdown(f"**Step {i+1}: `{tc['name']}`**")
                                st.code(str(tc.get("args", {}))[:500], language="json")
                    else:
                        st.caption("本轮无工具调用")

            # Save assistant message
            detail_json = json.dumps({
                "stage": result.get("stage", ""),
                "tool_rounds": result.get("tool_rounds", 0),
                "has_product_context": bool(result.get("product_context")),
                "profile_snapshot": result.get("user_profile", "")[:200],
            }, ensure_ascii=False)
            CONV_STORE.add_message(conv_id, "assistant", answer, details=detail_json)

        except Exception as e:
            import traceback
            error_msg = f"Agent 执行出错: {str(e)}"
            st.error(error_msg)
            with st.expander("错误详情"):
                st.code(traceback.format_exc())
            CONV_STORE.add_message(conv_id, "assistant", error_msg)

    st.rerun()


# ---- Bottom: Quick diagnostics ----
st.divider()
st.caption("🧪 快速诊断 — 验证各组件是否正常")

if st.button("检查组件状态", use_container_width=True):
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("**ChromaDB 产品索引**")
        try:
            import chromadb
            from src.retrieval.index_manifest import resolve_collection_names
            client = chromadb.PersistentClient(path=config.PRODUCT_CHROMA_DIR)
            collection_names, _ = resolve_collection_names(config.PRODUCT_CHROMA_DIR)
            desc_n = client.get_collection(collection_names["descriptions"]).count()
            spec_n = client.get_collection(collection_names["specs"]).count()
            review_n = client.get_collection(collection_names["reviews"]).count()
            st.success(f"✅ {desc_n} 产品 / {spec_n} 规格 / {review_n} 评价")
        except Exception as e:
            st.error(f"❌ {e}")

    with col2:
        st.markdown("**SQLite 产品目录**")
        try:
            import sqlite3
            conn = sqlite3.connect(config.PRODUCT_DB_PATH)
            n = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
            conn.close()
            st.success(f"✅ {n} 款产品")
        except Exception as e:
            st.error(f"❌ {e}")

    with col3:
        st.markdown("**画像存储**")
        try:
            from src.profile.profile_store import ProfileStore
            store = ProfileStore(config.PROFILE_DB_PATH, config.PROFILE_CHROMA_DIR, config.EMBEDDING_MODEL)
            st.success("✅ ProfileStore 可用")
        except Exception as e:
            st.error(f"❌ {e}")

    with col4:
        st.markdown("**Redis 缓存**")
        cache = st.session_state.get("rag_cache")
        if cache is not None and cache._redis is not None:
            try:
                cache._redis.ping()
                st.success("✅ Redis 缓存已连接")
            except Exception:
                st.warning("⚠ Redis 连接丢失")
        elif config.REDIS_ENABLED:
            st.warning("⚠ 已启用但无法连接")
        else:
            st.info("Redis 未启用 (REDIS_ENABLED=false)")
