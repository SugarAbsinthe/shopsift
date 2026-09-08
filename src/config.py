"""Centralized configuration management for shopping guide Agent."""
import os
import logging
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class Config:
    # LLM
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "sk-placeholder")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # Database
    PRODUCT_DB_PATH: str = os.getenv("PRODUCT_DB_PATH", str(BASE_DIR / "data" / "products.db"))
    PROFILE_DB_PATH: str = os.getenv("PROFILE_DB_PATH", str(BASE_DIR / "data" / "profiles.db"))
    CONV_DB_PATH: str = os.getenv("CONV_DB_PATH", str(BASE_DIR / "data" / "conversations.db"))
    AGENT_CHECKPOINT_DB_PATH: str = os.getenv(
        "AGENT_CHECKPOINT_DB_PATH", str(BASE_DIR / "data" / "agent_checkpoints.db")
    )

    # ChromaDB
    PRODUCT_CHROMA_DIR: str = os.getenv("PRODUCT_CHROMA_DIR", str(BASE_DIR / "data" / "product_chroma_db"))
    PROFILE_CHROMA_DIR: str = os.getenv("PROFILE_CHROMA_DIR", str(BASE_DIR / "data" / "profile_chroma_db"))

    # Embedding
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

    # Agent
    MAX_TOOL_ROUNDS: int = 3

    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    REDIS_ENABLED: bool = os.getenv("REDIS_ENABLED", "false").lower() == "true"
    RAG_CACHE_TTL: int = int(os.getenv("RAG_CACHE_TTL", "900"))  # 15 min default

    # Profile
    PROFILE_DECAY_LAMBDA: float = 0.05  # half-life ~14 days

    # LangSmith
    LANGSMITH_TRACING: bool = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")
    LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "shopsift")
    LANGSMITH_ENDPOINT: str = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")


def _clean_additional_kwargs(additional_kwargs: dict) -> dict:
    """Remove keys that cause issues when passed back in agent loops.

    DeepSeek returns 'reasoning_content' in its response. When LangChain
    includes this AIMessage in the next agent iteration, the API complains
    'reasoning_content must be passed back' even though it was passed.
    The safest fix is to strip it from the message after generation.
    """
    strip_keys = {"reasoning_content"}
    return {k: v for k, v in additional_kwargs.items() if k not in strip_keys}


def _patch_generations(generations):
    """Strip problematic additional_kwargs from all generated messages."""
    for gen in generations:
        msg = getattr(gen, "message", None)
        if isinstance(msg, AIMessage) and msg.additional_kwargs:
            msg.additional_kwargs = _clean_additional_kwargs(msg.additional_kwargs)


class DeepSeekChatOpenAI:
    """Mixin that strips reasoning_content from ChatOpenAI responses.

    Applied dynamically at instance creation time — patches _generate
    and _agenerate to clean responses before they enter the agent loop.
    """

    @staticmethod
    def wrap(instance):
        """Wrap a ChatOpenAI instance to clean DeepSeek-specific response fields."""
        base_generate = instance._generate
        base_agenerate = instance._agenerate

        def patched_generate(messages, stop=None, run_manager=None, **kwargs):
            result = base_generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            _patch_generations(getattr(result, "generations", []))
            return result

        async def patched_agenerate(messages, stop=None, run_manager=None, **kwargs):
            result = await base_agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
            _patch_generations(getattr(result, "generations", []))
            return result

        instance._generate = patched_generate
        instance._agenerate = patched_agenerate
        return instance


def create_llm(api_key: str = None, base_url: str = None, model: str = None,
               temperature: float = 0, timeout: int = 60):
    """Create a ChatOpenAI instance configured for the current provider.

    For DeepSeek: monkey-patches _generate/_agenerate to strip
    'reasoning_content' from responses. This prevents the
    'reasoning_content must be passed back to the API' error
    that occurs when LangChain re-sends message history in the
    Agent executor loop.
    """
    from langchain_openai import ChatOpenAI

    api_key = api_key or config.OPENAI_API_KEY
    base_url = base_url or config.OPENAI_BASE_URL
    model = model or config.LLM_MODEL

    llm = ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
        timeout=timeout,
    )

    if "deepseek" in base_url.lower():
        DeepSeekChatOpenAI.wrap(llm)

    return llm


def init_langsmith(environ=None) -> bool:
    """Enable optional tracing only when explicitly configured with a key."""
    target_env = os.environ if environ is None else environ
    if not config.LANGSMITH_TRACING:
        target_env["LANGSMITH_TRACING"] = "false"
        return False

    api_key = config.LANGSMITH_API_KEY or target_env.get("LANGSMITH_API_KEY", "")
    if not api_key:
        target_env["LANGSMITH_TRACING"] = "false"
        logging.getLogger(__name__).warning(
            "LangSmith tracing requested but LANGSMITH_API_KEY is missing; tracing disabled"
        )
        return False

    target_env["LANGSMITH_TRACING"] = "true"
    target_env["LANGSMITH_API_KEY"] = api_key
    target_env["LANGSMITH_PROJECT"] = config.LANGSMITH_PROJECT
    target_env["LANGSMITH_ENDPOINT"] = config.LANGSMITH_ENDPOINT
    return True


config = Config()


def ensure_hf_offline_if_needed() -> bool:
    """Check if model cache exists locally; set offline if so.

    Avoids network timeouts by checking local cache directory first.
    Only attempts online if the model has never been downloaded.
    """
    import logging
    logger = logging.getLogger(__name__)

    if os.environ.get("HF_HUB_OFFLINE") == "1":
        return True

    # Check if the embedding model is already cached locally
    from pathlib import Path
    cache_root = os.environ.get("HF_HOME", os.path.join(str(Path.home()), ".cache", "huggingface"))
    hub_dir = os.path.join(cache_root, "hub")
    if os.path.isdir(hub_dir):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        logger.info("Model cache found at %s — using offline mode", hub_dir)
        return True

    logger.info("No model cache found — online mode")
    return False
