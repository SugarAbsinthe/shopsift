"""Deterministic user commands; never infer a destructive profile edit."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from src.retrieval.models import canonical_brand


@dataclass(frozen=True)
class ProfileCommandResult:
    handled: bool
    remaining_text: str
    clarification: str = ""
    session_cleared: tuple[str, ...] = ()


_ALIASES = {
    "预算": "budget", "用途": "primary_use", "品牌偏好": "preferred_brand",
    "移动需求": "mobility", "刚需": "must_have", "排除品牌": "exclude_brand",
    "品牌排除": "exclude_brand", "屏幕偏好": "screen_preference",
    "续航要求": "battery_requirement", "商品品类": "product_category",
}
_KEYS = "budget|primary_use|preferred_brand|mobility|must_have|exclude_brand|screen_preference|battery_requirement|product_category"
_DELETE_RE = re.compile(
    r"(?:删除|忘记|清除|取消|撤销)\s*(?:刚才的)?\s*(?:画像|偏好)?\s*[:：]?\s*("
    + _KEYS + "|" + "|".join(_ALIASES) + ")", re.I,
)
_CANDIDATE_RE = re.compile(
    r"(确认|接受|保留|拒绝|不要|否决|删除|忘记|清除)\s*(?:画像)?(?:候选)?\s*[:：]?\s*"
    r"((?:mem|legacy)_[a-zA-Z0-9]+)"
)
_UNLIMITED_BUDGET_RE = re.compile(
    r"(?:预算(?:不设上限|不限(?:制)?|没有限制|无上限)|"
    r"(?:不再|不|不用|不要)(?:限制|限定|设置|设定)预算|不设预算上限)"
)
_NO_BRAND_PREFERENCE_RE = re.compile(r"(?:品牌|牌子)(?:不限|都行|都可以|无所谓|没有偏好|不限制)")
_SOFT_BUDGET_RE = re.compile(r"预算.{0,6}(?:不卡(?:那么|这么)?死|不(?:用|要)?卡(?:那么|这么)?死|放宽(?:一点|一些)?|放松(?:一点|一些)?|可以灵活)")


def process_profile_commands(conv_id: str, message: str, store, *,
                             on_text: Callable[[str], set[str]] | None = None,
                             on_clear: Callable[[str], None] | None = None) -> ProfileCommandResult:
    """Apply explicit edits, leaving other requirements available for extraction.

    An unspecified budget relaxation asks for a new limit instead of silently
    removing a hard cap. Quoted, hypothetical and negated commands are inert.
    """
    remaining = []
    handled = False
    clarification = ""
    session_cleared = []
    scope_text = re.sub(r"暂时(?=(?:先)?(?:不要|不用|不需要|不急着|不|别)?(?:推荐|讨论|梳理|理清|聊))", "", message)
    session_only = bool(re.search(r"这次|本次|这回|暂时|临时", scope_text))

    def retain(text):
        nonlocal clarification, session_cleared
        remaining.append(text)
        if on_text is not None:
            changed = on_text(text)
            session_cleared = [key for key in session_cleared if key not in changed]
            if "budget" in changed:
                clarification = ""

    for clause in re.split(r"[；;。!?！？\n]|(?<!\d)[，,]|[，,](?!\d)|并(?:且)?(?=删除|忘记|清除|取消|撤销)", message):
        text = clause.strip()
        if not text:
            continue
        if re.search(r"如果|假如|假设|例如|举例|比如|他说|她说|你说|[“「『\"]|(?:不要|别|不(?:要|用|必|需要|想|打算|再)?|无需|无须).{0,6}(?:删除|忘记|清除|取消|撤销)", text):
            remaining.append(text)
            continue
        candidate = _CANDIDATE_RE.search(text)
        deletion = _DELETE_RE.search(text)
        unlimited = _UNLIMITED_BUDGET_RE.search(text)
        no_brand = _NO_BRAND_PREFERENCE_RE.search(text)
        match = candidate or deletion or unlimited or no_brand
        keys = []
        if deletion:
            keys = [_ALIASES.get(deletion[1], deletion[1].lower())]
        elif unlimited:
            keys = ["budget"]
        elif no_brand:
            keys = ["preferred_brand"]
        if not match:
            # Clear only the named active brand, including canonical aliases.
            brand = re.search(r"(?:不一定|不再|不必|不用|不需要)(?:只)?(?:要|选|买|考虑)\s*([A-Za-z ]+?|[\u4e00-\u9fff]+?)(?:的|了)?$", text)
            getter = getattr(store, "get_structured", None)
            profile = getter(conv_id) if callable(getter) else {}
            if brand and isinstance(profile, dict):
                preferred = profile.get("preferred_brand", {}).get("value", "")
                if preferred and canonical_brand(brand[1]) == canonical_brand(preferred):
                    match, keys = brand, ["preferred_brand"]
            if not match and _SOFT_BUDGET_RE.search(text) and not re.search(r"\d|[一二两三四五六七八九](?:千|万)", text):
                if isinstance(profile, dict) and profile.get("budget"):
                    handled = True
                    clarification = "你希望把预算上限调整到多少？确认前会保留原预算，不会自动推荐超预算商品。"
                    continue
        if match:
            handled = True
            try:
                if candidate:
                    verb = candidate[1]
                    action = ("confirm_candidate" if verb in {"确认", "接受", "保留"} else
                              "reject_candidate" if verb in {"拒绝", "不要", "否决"} else "delete_candidate")
                    getattr(store, action)(conv_id, candidate[2])
                else:
                    for key in keys:
                        if session_only:
                            session_cleared.append(key)
                        else:
                            store.delete_profile_key(conv_id, key)
                        if on_clear is not None:
                            on_clear(key)
            except (KeyError, ValueError):
                # Stale or cross-conversation candidate IDs cannot become new
                # positive profile signals by falling through to extraction.
                pass
            text = text[:match.start()] + text[match.end():]
            text = re.sub(r"^(?:算了|其实|现在|先|那就|那)|(?:的|了|吧)$", "", text).strip()
        if text:
            retain(text)
    return ProfileCommandResult(handled, "，".join(remaining) if handled else message,
                                clarification, tuple(dict.fromkeys(session_cleared)))


def handle_profile_command(conv_id: str, message: str, profile_store) -> bool:
    """Compatibility entrypoint for callers interested only in edit handling."""
    return process_profile_commands(conv_id, message, profile_store).handled
