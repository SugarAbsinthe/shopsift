"""Test per-stage prompt mapping and template formatting."""
import pytest
from src.agent.shopping_prompts import (
    DISCOVERY_AGENT_PROMPT,
    SEARCH_AGENT_PROMPT,
    COMPARE_AGENT_PROMPT,
    RECOMMEND_AGENT_PROMPT,
    COMMON_STYLE_GUIDE,
    SHOPPING_SYSTEM_PROMPT,
    STAGE_CLASSIFIER_PROMPT,
)

STAGE_PROMPT_MAP = {
    "discovery": DISCOVERY_AGENT_PROMPT,
    "needs_elicitation": DISCOVERY_AGENT_PROMPT,
    "search": SEARCH_AGENT_PROMPT,
    "comparison": COMPARE_AGENT_PROMPT,
    "objection_handling": SEARCH_AGENT_PROMPT,
    "recommendation": RECOMMEND_AGENT_PROMPT,
    "summary": RECOMMEND_AGENT_PROMPT,
}


class TestStagePromptMapping:
    """Verify every stage maps to a valid, formattable prompt."""

    @pytest.mark.parametrize("stage", list(STAGE_PROMPT_MAP.keys()))
    def test_stage_has_prompt(self, stage):
        prompt = STAGE_PROMPT_MAP[stage]
        assert len(prompt) > 100, f"{stage} prompt is too short"

    @pytest.mark.parametrize("stage", list(STAGE_PROMPT_MAP.keys()))
    def test_prompt_accepts_format_variables(self, stage):
        prompt = STAGE_PROMPT_MAP[stage]
        result = prompt.format(
            conv_id="test-001",
            stage=stage,
            user_profile="(暂无画像)",
            product_context="",
        )
        assert len(result) > 0
        assert "test-001" in result
        assert stage in result

    def test_discovery_prompt_does_not_mention_search(self):
        """Discovery stage should not expose search tools to LLM."""
        assert "search_products" not in DISCOVERY_AGENT_PROMPT.lower()

    def test_search_prompt_has_product_context_placeholder(self):
        assert "{product_context}" in SEARCH_AGENT_PROMPT

    def test_active_prompts_do_not_expose_duplicate_search_tool(self):
        for prompt in {
            SHOPPING_SYSTEM_PROMPT,
            DISCOVERY_AGENT_PROMPT,
            SEARCH_AGENT_PROMPT,
            COMPARE_AGENT_PROMPT,
            RECOMMEND_AGENT_PROMPT,
        }:
            assert "search_products" not in prompt

    def test_active_prompts_do_not_delegate_profile_writes_to_the_model(self):
        for prompt in {
            SHOPPING_SYSTEM_PROMPT,
            DISCOVERY_AGENT_PROMPT,
            SEARCH_AGENT_PROMPT,
            COMPARE_AGENT_PROMPT,
            RECOMMEND_AGENT_PROMPT,
        }:
            assert "update_user_profile" not in prompt

    def test_all_stage_prompts_include_style_guide(self):
        for name in ["DISCOVERY", "SEARCH", "COMPARE", "RECOMMEND"]:
            prompt = globals()[f"{name}_AGENT_PROMPT"]
            assert "300 字" in prompt, f"{name} missing style guide"


class TestLegacyPrompts:
    """Old prompts kept for backward compatibility."""

    def test_system_prompt_formattable(self):
        result = SHOPPING_SYSTEM_PROMPT.format(
            conv_id="test", stage="discovery",
            user_profile="", product_context="",
        )
        assert "test" in result

    def test_stage_classifier_formattable(self):
        result = STAGE_CLASSIFIER_PROMPT.format(
            current_stage="discovery",
            user_message="你好",
        )
        assert "discovery" in result
        assert "你好" in result
