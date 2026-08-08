"""
 查询增强（短期记忆驱动的指代/品牌/价格补全）

基于 session_context 快照对用户问题做轻量增强，把省略/指代/价格倾向补全为完整检索式。

4 条规则（按优先级，第一条命中即返回）：
  ① 切换意图检测：'不要手机/换电脑/看一下冰箱' → 原句直传并标记 switch_intent（上游清空会话）；
  ② 强指代消解：'它/这个/还有/再推荐' 用上一轮关键词补全；
  ③ 品牌品类补全：短句 + 品牌词 → '品牌 + 上一轮小类'；
  ④ 价格补全：含价格倾向词时按上一轮价格区间缩放（便宜→上限减半、贵→下限=0.7*上限）。

门控：confidence 低于 0.6 的实体不参与增强；原问题已完整不增强；
enrich_query 整体 try/except：任何异常返回原问题直通（enriched=False）。
"""
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.offline.etl.category_classifier import CATEGORY_TREE

# 词表
# 品牌词表（规则③ / extractor 共用；匹配忽略大小写）
BRAND_WORDS: List[str] = [
    "iPhone", "Apple", "华为", "荣耀", "小米", "Redmi", "OPPO", "vivo", "一加",
    "三星", "索尼", "戴森", "飞利浦", "松下", "联想", "戴尔", "惠普",
    "海尔", "美的", "格力", "大疆", "Bose", "Beats", "漫步者", "森海塞尔",
    "佳能", "尼康", "富士", "苹果",
]

# 类目词表（大类/中类/小类，按长度降序，最长优先匹配）
CATEGORY_WORDS: List[str] = sorted(
    {w for big, middles in CATEGORY_TREE.items() for w in [big, *middles.keys(), *sum(middles.values(), [])]},
    key=len, reverse=True,
)
# 口语高频品类词（补充类目树外的常见词）
_CATEGORY_EXTRA = ["手机", "电脑", "冰箱", "电视", "耳机", "鞋", "衣", "包", "相机", "空调", "洗衣机"]
_CATEGORY_TERM_RE = "|".join(re.escape(w) for w in (_CATEGORY_EXTRA + CATEGORY_WORDS))

# 规则① 切换意图正则：'不要手机 / 换电脑 / 看一下冰箱' 等
_SWITCH_RE = re.compile(
    r"(了解一下|看一下|改看|看看|看下|不要|换)\s*(" + _CATEGORY_TERM_RE + r")"
)

# 规则② 指代触发词（长词在前避免部分吞噬）
_REFERENCE_RE = re.compile(
    r"(上面那个|刚才那个|还有呢|别的呢|再推荐|类似的|这个|那个|还有|别的|类似|再|它)"
)

# 规则④ 价格倾向词
_PRICE_RE = re.compile(r"(便宜|贵|实惠|预算|价位|多少钱以内|性价比)")

# 常见中文动词（用于「原问题完整」判断）
_VERBS = frozenset("有 是 买 卖 查 看 换 推荐 介绍 了解 找 求 要 想 问 知道 拿 选 挑 给我 帮我 喜欢".split())

_CONF_THRESHOLD = 0.6

# 实体键 → confidence 键映射（extractor 写入的短名，与实体键的后缀不完全一致）
_CONF_KEY_MAP = {
    "last_query_keyword": "keyword",
    "last_mentioned_sku": "sku",
    "last_answer_summary": "summary",
}


def _conf_key(key: str) -> str:
    """把实体键映射为 confidence 中的键名（last_query_keyword → keyword）。"""
    if key in _CONF_KEY_MAP:
        return _CONF_KEY_MAP[key]
    return key[5:] if key.startswith("last_") else key


@dataclass
class EnrichResult:
    """查询增强结果。"""

    query: str            # 最终用于下游（意图识别/二次回调）的问题
    original: str         # 原始问题
    enriched: bool        # 是否发生了增强
    switch_intent: bool   # 是否触发意图切换（上游应清空会话上下文）
    reason: str           # 命中规则说明（调试/日志用）


def _conf_ok(ctx: Dict[str, Any], key: str) -> bool:
    """置信度门控：实体存在且 confidence >= 0.6 才参与增强。

    confidence 的键使用实体短名（如 last_query_keyword → keyword），
    与 extractor 写入的 confidence 结构保持一致。
    """
    conf = ctx.get("confidence") or {}
    val = ctx.get(key)
    if val is None or not bool(val):
        return False
    return float(conf.get(_conf_key(key), 0.0)) >= _CONF_THRESHOLD


def _pick_keyword(ctx: Dict[str, Any]) -> Optional[str]:
    """按优先级挑一个可用的补全关键词（置信度门控）。"""
    for key in ("last_query_keyword", "last_brand", "last_mentioned_sku", "last_answer_summary"):
        if _conf_ok(ctx, key):
            return ctx[key]
    return None


def _clean_reference_query(query: str) -> str:
    """去掉指代触发词与句尾语气词，留下可检索的核心片段（'还有便宜点的吗'→'便宜点的'）。"""
    cleaned = _REFERENCE_RE.sub(" ", query)
    cleaned = re.sub(r"[吗呢吧么]+$", "", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _find_brand(query: str) -> Optional[str]:
    """返回 query 中命中的品牌词（保留原文大小写），未命中返回 None。"""
    low = query.lower()
    for b in BRAND_WORDS:
        if b.lower() in low:
            return b
    return None


def _is_complete(query: str) -> bool:
    """原问题是否已完整：长度>=5 且同时含动词与品类/品牌名词。"""
    if len(query) < 5:
        return False
    if not any(v in query for v in _VERBS):
        return False
    low = query.lower()
    for cw in CATEGORY_WORDS:
        if cw.lower() in low:
            return True
    return _find_brand(query) is not None


def _history_keyword(history: Optional[List[Dict[str, str]]]) -> Optional[str]:
    """ctx 无可用实体时，从 history 最近一条用户消息提取品类词兜底。"""
    if not history:
        return None
    for msg in reversed(history):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        for w in CATEGORY_WORDS:
            if w in content:
                return w
        brand = _find_brand(content)
        if brand:
            return brand
    return None


def _adjust_price(query: str, price_range: Any) -> Optional[str]:
    """按价格倾向词调整上一轮价格区间，返回可追加的价格描述；无法调整返回 None。

    兼容两种快照形态：list [min, max]（extractor 写入）或 dict {"min", "max"}。
    """
    if isinstance(price_range, (list, tuple)) and len(price_range) == 2:
        lower, upper = price_range[0], price_range[1]
    elif isinstance(price_range, dict):
        upper = price_range.get("max")
        lower = price_range.get("min")
    else:
        return None
    upper = float(upper) if upper is not None else None
    lower = float(lower) if lower is not None else None

    if "便宜" in query or "实惠" in query or "性价比" in query:
        if upper is None:
            return None
        upper_new = math.floor(upper * 0.5)
        lower_new = min(lower, upper_new) if lower is not None else None
        if lower_new is not None and lower_new > upper_new:
            lower_new = math.floor(upper * 0.5)
        return f"{int(upper_new)}以内"
    if "贵" in query:
        if upper is None:
            return None
        lower_new = math.floor(upper * 0.7)
        return f"{int(lower_new)}以上"
    return None


def enrich_query(
    query: str,
    ctx: Optional[Dict[str, Any]] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> EnrichResult:
    """
    按 4 条规则对用户问题做记忆增强，返回 EnrichResult。
    整体 try/except 兜底：任何异常返回原问题直通（enriched=False）。
    """
    original = query or ""
    ctx = ctx or {}
    try:
        # 规则① 切换意图：原句直传，上游据此清空会话上下文
        m = _SWITCH_RE.search(original)
        if m:
            return EnrichResult(
                query=original, original=original,
                enriched=False, switch_intent=True,
                reason=f"switch_intent:{m.group(1) or ''}+{m.group(2) or ''}",
            )

        # 完整性门控：原问题已含品牌/品类等可检索实体（如「衬衫 还有现货吗」）
        # 视为全新查询，不再套用上一轮记忆做指代补全（避免拼接出「iPhone 衬衫」）
        if _is_complete(original):
            return EnrichResult(query=original, original=original,
                                enriched=False, switch_intent=False,
                                reason="complete_no_enhance")

        # 规则② 强指代消解：'还有便宜点的吗' → '手机 便宜点的'
        if _REFERENCE_RE.search(original):
            cleaned = _clean_reference_query(original)
            if not cleaned:
                return EnrichResult(query=original, original=original,
                                    enriched=False, switch_intent=False,
                                    reason="reference_no_keyword")
            keyword = _pick_keyword(ctx)
            if not keyword:
                keyword = _history_keyword(history)
            if not keyword:
                return EnrichResult(query=original, original=original,
                                    enriched=False, switch_intent=False,
                                    reason="reference_no_context")
            enhanced = f"{keyword} {cleaned}"
            return EnrichResult(query=enhanced, original=original,
                                enriched=True, switch_intent=False,
                                reason=f"reference+{keyword}")

        # 规则③ 品牌品类补全：短句 + 品牌词 → '品牌 + 上一轮小类'
        if len(original) < 8:
            brand = _find_brand(original)
            if brand and _conf_ok(ctx, "last_category_small"):
                enhanced = f"{brand} {ctx['last_category_small']}"
                return EnrichResult(query=enhanced, original=original,
                                    enriched=True, switch_intent=False,
                                    reason=f"brand+category:{brand}")

        # 规则④ 价格补全：'便宜点的' → '手机 2500以内'（基于上一轮价格区间）
        if _PRICE_RE.search(original):
            price_desc = _adjust_price(original, ctx.get("last_price_range") or {})
            if price_desc:
                base = _pick_keyword(ctx)
                if base:
                    enhanced = f"{base} {price_desc}"
                    return EnrichResult(query=enhanced, original=original,
                                        enriched=True, switch_intent=False,
                                        reason=f"price+{price_desc}")

        # 未命中任何规则：原句直通
        return EnrichResult(query=original, original=original,
                            enriched=False, switch_intent=False,
                            reason="no_enhance")
    except Exception as e:  # 增强失败兜底：原问题直通，绝不阻断主链路
        return EnrichResult(query=original, original=original,
                            enriched=False, switch_intent=False,
                            reason=f"error:{e}")
