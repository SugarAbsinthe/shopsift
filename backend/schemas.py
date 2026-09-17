"""Pydantic models for FastAPI request/response schemas."""

from typing import Optional

from pydantic import BaseModel, Field


# ---- Chat ----

class ChatMessage(BaseModel):
    """A single chat history message sent from the frontend."""
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    """Request body for POST /api/chat."""
    conv_id: str = Field(..., min_length=1, description="Conversation ID")
    question: str = Field(..., min_length=1, description="User's latest message")
    chat_history: list[ChatMessage] = Field(
        default_factory=list,
        description="Prior conversation messages (role + content)"
    )


class ChatResponse(BaseModel):
    """Response body from POST /api/chat."""
    answer: str
    stage: str
    product_context: str
    user_profile: str
    tool_rounds: int
    agent_rounds: int
    stop_reason: str
    run_id: str
    request_id: str = ""
    latency_ms: int
    llm_calls: int
    llm_latency_ms: int
    llm_retries: int
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    retrieval_triggered: bool
    cache_hit: Optional[bool] = None
    requested_tools: list[str]
    executed_tools: list[str]
    tool_errors: int
    retrieval_stats: dict = Field(default_factory=dict)
    provider: str = "unknown"
    model: str = "unknown"
    prompt_version: str = "unknown"
    pricing_version: str = "unconfigured"
    estimated_cost_usd: Optional[float] = None
    node_latency_ms: dict[str, int] = Field(default_factory=dict)
    failure_counts: dict[str, int] = Field(default_factory=dict)
    fallbacks: list[str] = Field(default_factory=list)
    tool_policy: list[dict[str, str]] = Field(default_factory=list)
    timeouts: int = 0
    cancelled: bool = False
    budget_stop: Optional[str] = None
    checkpoint_restored: bool = False


# ---- Conversations ----

class ConversationItem(BaseModel):
    """A conversation in the history list."""
    id: str
    title: str
    model: str
    created_at: str
    updated_at: str


class CreateConversationRequest(BaseModel):
    """Request body for POST /api/conversations."""
    title: str = "新对话"
    model: str = ""


class AddMessageRequest(BaseModel):
    """Request body for POST /api/conversations/{id}/messages."""
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str
    details: Optional[dict] = None


class MessageItem(BaseModel):
    """A single message in conversation history."""
    id: int
    conv_id: str
    role: str
    content: str
    details: Optional[str] = None
    created_at: str


# ---- Health ----

class ComponentStatus(BaseModel):
    llm: bool = False
    chromadb: bool = False
    redis: bool = False
    database: bool = False


class HealthResponse(BaseModel):
    status: str
    components: ComponentStatus
