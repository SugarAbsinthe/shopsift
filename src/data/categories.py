"""Shared category definitions and conservative, typed specification filters."""
from __future__ import annotations

import re
import math

LEGACY_CATEGORIES = ("笔记本电脑", "手机", "平板电脑", "无线耳机")
REQUIRED = {
    "笔记本电脑": ("处理器", "内存", "存储", "屏幕尺寸", "显卡", "重量"),
    "手机": ("处理器", "存储", "屏幕尺寸", "重量"),
    "平板电脑": ("处理器", "存储", "屏幕尺寸", "网络版本", "重量"),
    "无线耳机": ("佩戴方式", "连接方式", "耳机单次续航"),
    "显示器": ("屏幕尺寸", "屏幕分辨率", "屏幕类型", "刷新率条件", "接口", "支架调节", "含支架尺寸", "含支架重量"),
    "键盘": ("配列", "键数", "布局", "按键技术", "轴体", "连接方式", "尺寸", "重量", "系统兼容", "供电续航"),
    "鼠标": ("尺寸", "重量", "重量口径", "左右手形态", "连接方式", "传感器类型", "DPI范围", "按键数", "供电续航"),
}
CATEGORIES = tuple(REQUIRED)
ALIASES = {
    "显示器": ("显示器", "外接屏", "monitor"), "键盘": ("键盘", "keyboard"), "鼠标": ("鼠标", "mouse"),
    "手机": ("手机", "iphone", "华为mate", "小米14"),
    "笔记本电脑": ("笔记本", "游戏本", "轻薄本", "macbook", "thinkpad"),
    "平板电脑": ("平板", "ipad"), "无线耳机": ("耳机", "airpods", "降噪耳机"),
}
# Values are only populated from reviewed SKU facts, never marketing prose.
FILTER_FIELDS = {
    "memory_gb": "number", "storage_gb": "number", "dedicated_gpu": "bool",
    "screen_inches": "number", "refresh_hz": "number", "resolution": "string",
    "panel": "string", "ports": "set", "usb_c_pd_w": "number", "stand_height_adjustable": "bool",
    "key_count": "number", "layout": "string", "switch": "string", "switch_technology": "string",
    "connections": "set", "weight_g": "number", "body_weight_g": "number", "dpi_max": "number", "button_count": "number",
    "handedness": "string", "systems": "set", "hot_swap": "bool",
    "hdmi_refresh_hz": "number", "dp_refresh_hz": "number", "usb_c_refresh_hz": "number",
}
ENUM_VALUES = {
    "connections": {"bluetooth", "2.4ghz", "wired"},
    "ports": {"HDMI", "DisplayPort", "USB-C", "USB-A", "Thunderbolt", "RJ-45", "VGA", "DVI", "audio"},
    "systems": {"Windows", "macOS", "Linux", "ChromeOS", "Android", "iOS", "iPadOS", "PS5", "Xbox"},
    "layout": {"ANSI", "ISO", "US-QWERTY", "JIS"},
    "switch_technology": {"scissor", "mechanical", "magnetic", "membrane", "optical"},
    "panel": {"IPS", "IPS Black", "VA", "TN", "OLED"},
    "handedness": {"right", "left", "symmetric", "asymmetric", "ambidextrous"},
}
EXAMPLES = {
    "笔记本电脑": "办公和编程用，推荐几款不同型号的笔记本电脑",
    "手机": "推荐几款手机，比较屏幕、重量和公开价格",
    "平板电脑": "推荐上课和看视频用的平板电脑",
    "无线耳机": "推荐通勤用的无线耳机，比较佩戴方式和降噪",
    "显示器": "推荐办公用的显示器，比较分辨率、接口和支架",
    "键盘": "推荐办公和编程用的键盘，比较布局、连接方式和轴体",
    "鼠标": "推荐几款鼠标，比较重量、手型和连接方式",
}


def port_refresh_at_resolution(text: str, resolution: str) -> float | None:
    """Qualify a port's refresh value only at the recorded native resolution."""
    resolutions = {f'{width}x{height}' for width, height in
                   re.findall(r'(?<!\d)(\d{3,5})\s*[x×*]\s*(\d{3,5})(?!\d)', text, re.I)}
    expected = re.sub(r'\s+', '', resolution).lower().replace('×', 'x').replace('*', 'x')
    frequencies = {float(value) for value in re.findall(r'(\d+(?:\.\d+)?)\s*Hz\b', text, re.I)}
    if resolutions != {expected} or len(frequencies) != 1:
        return None
    frequency = frequencies.pop()
    return frequency if math.isfinite(frequency) and frequency > 0 else None


def category_mentions(text: str):
    """Purchase nouns take precedence over a mentioned host computer."""
    lower = text.casefold()
    for category, aliases in ALIASES.items():
        for alias in aliases:
            if alias in lower:
                yield alias, category
                break
    if "电脑" in lower:
        yield "电脑", "笔记本电脑"


def purchase_category(text: str) -> tuple[str, str]:
    """Find an affirmative category without treating a negated noun as a switch."""
    lower = text.casefold()
    for alias, category in category_mentions(lower):
        for match in re.finditer(re.escape(alias), lower):
            prefix = re.split(r"[,，。;；!?！？\n]|但是|不过|而是|但", lower[:match.start()])[-1]
            suffix = re.split(r"[,，。;；!?！？\n]|但是|不过|而是|但", lower[match.end():])[0]
            if re.search(r"不|没|无需|无须|并非|别|排除|除了", prefix):
                continue
            if re.match(r"\s*(?:不|没|无需|无须|并非)", suffix):
                continue
            return alias, category
    return "", ""


def matches_spec_filters(normalized: dict, filters) -> bool:
    for condition in filters:
        value = normalized.get(condition.field)
        expected = condition.value
        if value is None:
            return False
        kind = FILTER_FIELDS.get(condition.field)
        if kind == "number" and (not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)):
            return False
        if kind == "bool" and not isinstance(value, bool):
            return False
        if kind == "string" and not isinstance(value, str):
            return False
        if condition.op == "contains_all":
            if not isinstance(value, list) or not set(expected) <= set(value):
                return False
        elif condition.op == "eq":
            if value != expected:
                return False
        elif not isinstance(value, (int, float)) or isinstance(value, bool):
            return False
        elif condition.op == "gte" and value < expected:
            return False
        elif condition.op == "lte" and value > expected:
            return False
    return True


def catalog_filter_specs(category: str, specs: dict, declared: dict | None = None) -> dict:
    """Normalize explicit legacy specifications, never names or marketing prose."""
    result = dict(declared or {})
    if category not in LEGACY_CATEGORIES:
        return result
    for source, field in (("内存", "memory_gb"), ("存储", "storage_gb")):
        text = str(specs.get(source, ''))
        if re.search(r'最高|最大|支持|可选|选配|可扩|[≥≤+×*]', text):
            continue
        values = {float(n) * (1024 if unit.lower() == 'tb' else 1)
                  for n, unit in re.findall(r'(\d+(?:\.\d+)?)\s*(GB|TB)(?![A-Za-z])', text, re.I)}
        if len(values) == 1:
            result.setdefault(field, values.pop())
    sizes = {float(n) for n in re.findall(r'(\d+(?:\.\d+)?)\s*(?:英寸|inch|寸)', str(specs.get('屏幕尺寸', '')), re.I)}
    if len(sizes) == 1:
        result.setdefault('screen_inches', sizes.pop())
    gpu_type = str(specs.get('显卡类型', ''))
    integrated = bool(re.search('集成|核芯', gpu_type))
    dedicated = bool(re.search('独立', gpu_type))
    if integrated != dedicated:
        result.setdefault('dedicated_gpu', dedicated)
    return result


_NEGATION = r"(?:不再需要|不再要求|不需要|不要求|不想要|不想用|不考虑|不是|不必|不用|无需|无须|不要|不选|别选|排除|取消|撤销|去掉|放弃|非(?!常))"
_CLAUSE_BREAK = r"[；;，,。!?！？\n]|但是|不过|而是|但|并且|同时|只需要|只要|改为|改成"


def _negative_mention(text: str, start: int, end: int) -> bool:
    prefix = re.split(_CLAUSE_BREAK, text[:start])[-1]
    suffix = re.split(_CLAUSE_BREAK, text[end:])[0]
    return bool(re.search(_NEGATION, prefix) or re.match(r"\s*(?:的)?\s*" + _NEGATION, suffix))


def _spec_mentions(text: str, category: str):
    """Yield predicates with spans so negation stays local to each mention."""
    def matches(pattern, field, convert=lambda m: m[0], op="eq"):
        for match in re.finditer(pattern, text, re.I):
            yield {"field": field, "op": op, "value": convert(match)}, match.start(), match.end()

    if category not in LEGACY_CATEGORIES:
        yield from matches(r"蓝牙|Bluetooth", "connections", lambda m: ["bluetooth"], "contains_all")
        yield from matches(r"2\.4G(?:Hz)?", "connections", lambda m: ["2.4ghz"], "contains_all")
        yield from matches(r"有线(?:连接)?", "connections", lambda m: ["wired"], "contains_all")
    if category in LEGACY_CATEGORIES:
        yield from matches(r"(?:屏幕)?\s*(?:不超过|小于等于|≤)\s*(\d+(?:\.\d+)?)\s*英寸", "screen_inches", lambda m: float(m[1]), "lte")
        for match in re.finditer(r"(至少|不低于|不超过|≤|≥)?\s*(\d+(?:\.\d+)?)\s*(GB|TB)\s*(内存|存储)", text, re.I):
            field = 'memory_gb' if match[4] == '内存' else 'storage_gb'
            op = 'gte' if match[1] in ('至少', '不低于', '≥') else 'lte' if match[1] in ('不超过', '≤') else 'eq'
            value = float(match[2]) * (1024 if match[3].lower() == 'tb' else 1)
            yield {'field': field, 'op': op, 'value': value}, match.start(), match.end()
        yield from matches(r"(?:不要|不需要|不带|没有)独立显卡", "dedicated_gpu", lambda m: False)
    if category == "显示器":
        resolutions = {"4k": "3840x2160", "1080p": "1920x1080", "2k": "2560x1440", "2560x1440": "2560x1440"}
        yield from matches(r"4K|1080P|2K|2560x1440", "resolution", lambda m: resolutions[m[0].lower()])
        for match in re.finditer(r"(\d+)\s*Hz\s*(?:及以上|以上|起)", text, re.I):
            prefix = re.split(_CLAUSE_BREAK, text[:match.start()])[-1]
            ports = list(re.finditer(r"HDMI|DisplayPort|(?<![A-Za-z0-9])DP(?![A-Za-z0-9])|USB-C", prefix, re.I))
            port = ports[-1][0].upper() if ports and len(prefix)-ports[-1].end() <= 12 else ""
            field = {"HDMI": "hdmi_refresh_hz", "DP": "dp_refresh_hz", "DISPLAYPORT": "dp_refresh_hz", "USB-C": "usb_c_refresh_hz"}.get(port, "refresh_hz")
            yield {"field": field, "op": "gte", "value": int(match[1])}, match.start(), match.end()
        yield from matches(r"可升降|升降支架", "stand_height_adjustable", lambda m: True)
        for token, value in ((r"HDMI", "HDMI"), (r"USB-C", "USB-C"), (r"DisplayPort|(?<![A-Za-z0-9])DP(?![A-Za-z0-9])", "DisplayPort")):
            yield from matches(token, "ports", lambda m, v=value: [v], "contains_all")
        yield from matches(r"USB[- ]?C.{0,12}?(?:至少|不低于|≥|>=)\s*(\d+)\s*W", "usb_c_pd_w", lambda m: int(m[1]), "gte")
    elif category == "键盘":
        yield from matches(r"热插拔", "hot_swap", lambda m: True)
        yield from matches(r"磁轴", "switch_technology", lambda m: "magnetic")
        yield from matches(r"机械轴", "switch_technology", lambda m: "mechanical")
        for match in re.finditer(r"(至少|不低于)?\s*(\d+)\s*键\s*(及以上|以上|起)?", text):
            yield {"field": "key_count", "op": "gte" if match[1] or match[3] else "eq", "value": int(match[2])}, match.start(), match.end()
        yield from matches(r"ANSI|ISO", "layout", lambda m: m[0].upper())
    elif category == "鼠标":
        yield from matches(r"(?:不超过|小于等于|≤)\s*(\d+(?:\.\d+)?)\s*(?:g|克)", "weight_g", lambda m: float(m[1]), "lte")
    # Named limits can be removed without repeating their previous numeric value.
    reset_fields = {"重量限制|重量要求": ("weight_g",), "分辨率要求|分辨率限制": ("resolution",),
                    "刷新率要求|刷新率限制": ("refresh_hz", "hdmi_refresh_hz", "dp_refresh_hz", "usb_c_refresh_hz"),
                    "键数要求|键数限制": ("key_count",), "连接方式要求|连接方式限制": ("connections",)}
    for pattern, fields in reset_fields.items():
        for field in fields:
            yield from matches(pattern, field, lambda m: None)


def _requirement_edits(text: str, category: str) -> tuple[list[dict], list[dict]]:
    if category not in CATEGORIES:
        return [], []
    affirmative = re.sub(_NEGATION, "", text)
    if re.search(r"对比|对照|体验|评价|评测|价格|规格|参数", text) and not re.search(r"必须|需要|要求|只买", affirmative):
        return [], []
    if re.search(r"是否|能不能|有没有|支持.+吗|多少|怎么样", text) and not re.search(r"必须|需要|要求|只要|推荐|我想买", affirmative):
        return [], []
    if re.search(r"或者|还是|二选一|或", text):
        return [], []
    additions, removals = [], []
    for predicate, start, end in _spec_mentions(text, category):
        if _negative_mention(text, start, end):
            removals.append(predicate)
        elif predicate["value"] is not None:
            additions.append(predicate)
    return additions, removals


def explicit_spec_filters(text: str, category: str) -> list[dict]:
    """Extract positive hard requirements; unsupported negative intent stays prose."""
    predicates = {}
    for predicate in _requirement_edits(text, category)[0]:
        key = predicate["field"], predicate["op"]
        if predicate["op"] == "contains_all" and key in predicates:
            predicate = {**predicate, "value": list(dict.fromkeys(predicates[key]["value"] + predicate["value"]))}
        predicates[key] = predicate
    return list(predicates.values())


def has_spec_requirement_edits(text: str, category: str) -> bool:
    additions, removals = _requirement_edits(text, category)
    return bool(additions or removals)


def merge_spec_filters(previous: list[dict], current: str, category: str) -> list[dict]:
    """Apply cancellations before refinements, preserving unrelated requirements."""
    predicates = {(p["field"], p["op"]): {**p} for p in previous}
    additions, removals = _requirement_edits(current, category)
    for removed in removals:
        for key, old in list(predicates.items()):
            if old["field"] != removed["field"]:
                continue
            if old["op"] == "contains_all" and removed["value"] is not None:
                values = [v for v in old["value"] if v not in removed["value"]]
                if values:
                    predicates[key] = {**old, "value": values}
                    continue
            elif removed["value"] is not None and old["op"] == removed["op"] == "eq" and old["value"] != removed["value"]:
                continue
            del predicates[key]
    replacing_sets = bool(re.search(r"只需要|只要|改为|改成", current))
    replacing_bounds = bool(re.search(r"改为|改成|改到|换成|调整为|调整到", current))
    replaced = set()
    for predicate in additions:
        field, op = predicate["field"], predicate["op"]
        key = field, op
        if op == "contains_all":
            if replacing_sets and field not in replaced:
                predicates.pop(key, None)
                replaced.add(field)
            if old := predicates.get(key):
                predicate = {**predicate, "value": list(dict.fromkeys(old["value"] + predicate["value"]))}
        else:
            if FILTER_FIELDS.get(field) == "number" and replacing_bounds and field not in replaced:
                for old_key in list(predicates):
                    if old_key[0] == field:
                        del predicates[old_key]
                replaced.add(field)
            for old_key in list(predicates):
                if old_key[0] == field and (op == "eq" or old_key[1] == "eq"):
                    del predicates[old_key]
        predicates[key] = predicate
    return list(predicates.values())


def merge_requirement_text(previous: str, current: str, category: str) -> str:
    """Persist readable requirements, including unsupported negative intent."""
    predicates = merge_spec_filters(explicit_spec_filters(previous, category), current, category)
    fragments = []
    for p in predicates:
        f, v = p["field"], p["value"]
        if f in ('memory_gb', 'storage_gb'):
            qualifier = {'gte': '至少', 'lte': '不超过', 'eq': ''}[p['op']]
            fragments.append(f"{qualifier}{v:g}GB{'内存' if f == 'memory_gb' else '存储'}")
        elif f == 'screen_inches': fragments.append(f'屏幕不超过{v:g}英寸')
        elif f == 'dedicated_gpu': fragments.append('不要独立显卡')
        elif f == "connections": fragments += [{"bluetooth": "蓝牙", "2.4ghz": "2.4G", "wired": "有线"}[x] for x in v]
        elif f == "ports": fragments += v
        elif f == "resolution": fragments.append({"3840x2160": "4K", "1920x1080": "1080P", "2560x1440": "2560x1440"}[v])
        elif f.endswith("refresh_hz"):
            port = {"hdmi_refresh_hz": "HDMI ", "dp_refresh_hz": "DP ", "usb_c_refresh_hz": "USB-C "}.get(f, "")
            fragments.append(f"{port}{v}Hz及以上")
        elif f == "stand_height_adjustable": fragments.append("可升降")
        elif f == "usb_c_pd_w": fragments.append(f"USB-C供电至少{v}W")
        elif f == "key_count": fragments.append(f"{'至少' if p['op'] == 'gte' else ''}{v}键")
        elif f == "hot_swap": fragments.append("热插拔")
        elif f == "switch_technology": fragments.append({"magnetic": "磁轴", "mechanical": "机械轴"}[v])
        elif f == "layout": fragments.append(v)
        elif f == "weight_g": fragments.append(f"重量不超过{v:g}g")
    # Negative specifications have no inequality operator in the public filter
    # contract. Keep them visible to the dialogue rather than invert their meaning.
    for clause in re.split(r"[；;，,。\n]", previous + "；" + current):
        if re.search(_NEGATION, clause) and _requirement_edits(clause, category)[1]:
            # A current affirmative requirement supersedes an earlier negation.
            negatives = _requirement_edits(clause, category)[1]
            if clause in current or not any(p["field"] == n["field"] for p in _requirement_edits(current, category)[0] for n in negatives):
                fragments.append(clause.strip())
    return "；".join(dict.fromkeys(fragments))
