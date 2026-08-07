# -*- coding: utf-8 -*-
"""
core/agent/params.py —— 检索参数归一化层（LLM 输出 → 可执行检索参数）
========================================================================
架构定位：LLM 输出的 arguments 是「不可信的原始值」——模型常把整句自然语言
（如"苹果iPhone 15有现货吗"）塞进 product_name，或用字符串表达价格区间
（"3000以内""1万左右"）。本层负责：

  1. 清洗意图词 / 疑问词 / 语气词，提取核心商品实体（品牌 / 型号 / 品类）；
  2. 文本价格 → (min, max) 数值；
  3. 统一输出规范化后的检索参数 dict，供 router 执行 SQL / RAG。

设计原则（业界实践）：
  - LLM 输出永不直通 SQL（Rasa：intent 分类与 entity 抽取分离 + slot 校验）；
  - 清洗 / 校验失败回退原始 query（function calling 回退链）；
  - 检索永远有降级路径（Anthropic：精确 → 模糊 → 品牌/类目 → 反问澄清）。
"""
import re
from typing import Any, Dict, Optional, Tuple

from src.offline.etl.category_classifier import CATEGORY_TREE

# ------------------------------------------------------------
# 词表
# ------------------------------------------------------------
# 意图 / 疑问 / 语气词（检索前剔除；长词优先）。
# 注意：只放多字词——单字词（台/条/件/和/有/要/会/能/点/个/款/只/的/了/吗/呢/吧 等）
# 做纯子串替换会误删商品名关键内容（'台灯'→'灯'、'和牛'→'牛'），
# 一律放到 _SINGLE_STOP（\b 边界匹配，仅独立成词时删除）。
_INTENT_STOP = frozenset(
    """
    推荐 适合 送人 几款 多少钱 多少钱一个 怎么卖 价格多少 多少钱以内
    还有现货吗 还有货吗 还有优惠吗 还有别的 还有没有 还有吗 还有 有什么 有没有货 有现货 有货吗 现货 有货 缺货 没货
    有吗 有没有 有没 有么 在吗 便宜 实惠 性价比 怎么样 如何 多久 几天能到 几天到 哪个 哪款 哪些 什么价位 预算
    是否 是不是 请问 你好 谢谢 我想要 我想 我要 帮我 给我
    可以 能够 没有 一些 几款 类似 这样 那种 这种 一下 一个 一种 一台 一部 一双
    """.split()
)

# 单字意图/语气词：仅独立成词时删除（\b 边界；Python re 的 \w 含中文，
# '台灯' 中 '台' 前后是汉字非边界 → 不会误删；'看台' 同理安全）
_SINGLE_STOP = "台条件款和有要会能点个只吗么呢吧啊呀哦嘛哈哇啦的了货还买送"

# 品牌词表（匹配忽略大小写；与 enricher.BRAND_WORDS 同源，此处独立避免循环依赖）
BRAND_WORDS: list = [
    "iPhone", "Apple", "苹果", "华为", "荣耀", "小米", "Redmi", "OPPO", "vivo", "一加",
    "三星", "索尼", "戴森", "飞利浦", "松下", "联想", "戴尔", "惠普",
    "海尔", "美的", "格力", "大疆", "Bose", "Beats", "漫步者", "森海塞尔",
    "佳能", "尼康", "富士",
]

# 类目词（大类/中类/小类，最长优先），用于识别整句中的品类实体
CATEGORY_WORDS: list = sorted(
    {w for big, middles in CATEGORY_TREE.items() for w in [big, *middles.keys(), *sum(middles.values(), [])]},
    key=len, reverse=True,
)
_CATEGORY_SET = frozenset(CATEGORY_WORDS)

# 型号/标识符模式：品牌词之后的字母数字段（Mate 60 Pro / 15 / G054000 / K50 Ultra）
_MODEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]*(?:[\s\-][A-Za-z0-9]+)*")

# 文本价格模式（支持 万/千 单位与 以内/以下/左右/以上/区间）
_PRICE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(万|千)?\s*(以内|以下|左右|以上)?\s*"
    r"(?:[到至~\-—])\s*(\d+(?:\.\d+)?)\s*(万|千)?"
)
_PRICE_TAIL_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(万|千)?\s*(以内|以下|左右|以上)?$")


def clean_product_name(raw: Optional[str]) -> Optional[str]:
    """
    清洗商品名参数：剔除意图/疑问/语气词，返回核心商品词（品牌/型号/品类）。
    清洗后为空（纯寒暄/纯意图）返回 None——调用方应回退原始 query 或反问澄清。

    例：'苹果iPhone 15有现货吗' → '苹果iPhone 15'
        '衬衫 还有现货吗'     → '衬衫'
        '华为Mate 60 Pro 还有货吗' → '华为Mate 60 Pro'
        '推荐几款手机'        → '手机'
    """
    if not raw:
        return None
    text = str(raw).strip()
    # 1) 剔除意图/疑问/语气词（词表子串，长词优先；词表词存在子串重叠
    #    （如'有现货'与'还有'），迭代删除直至稳定，避免单字残骸）
    stop_words = sorted(_INTENT_STOP, key=len, reverse=True)
    while True:
        prev = text
        for w in stop_words:
            text = text.replace(w, " ")
        if text == prev:
            break
    # 2) 压缩标点与空白
    text = re.sub(r"[，。！？?！、；;：:（）()【】\[\]\"'`~\s]+", " ", text).strip()
    # 3) 残留的单字语气/意图词：仅独立成词时删除（\b 边界，保护'台灯'/'和牛'等商品名）
    text = re.sub(rf"\b[{_SINGLE_STOP}]\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def extract_brand(text: str) -> Optional[str]:
    """返回文本中命中的品牌词（保留原文大小写），未命中返回 None。"""
    if not text:
        return None
    low = text.lower()
    for b in BRAND_WORDS:
        if b.lower() in low:
            return b
    return None


def extract_model(text: str) -> Optional[str]:
    """
    提取型号串（品牌词 + 后续型号段，如 '华为Mate 60 Pro' / '苹果iPhone 15'）。
    仅当品牌词存在且型号段含数字时提取；无法确定返回 None。
    """
    if not text:
        return None
    brand = extract_brand(text)
    if not brand:
        return None
    idx = text.lower().find(brand.lower())
    tail = text[idx + len(brand):].strip()
    m = _MODEL_RE.match(tail)
    if not m:
        return None
    seg = m.group(0).strip()
    # 型号段必须包含数字（避免 'iPhone' 后直接是品类词时误提取）
    if not seg or not re.search(r"\d", seg):
        return None
    # 型号段以数字开头时补空格（'iPhone 15'），否则直接拼接（'华为Mate 60 Pro'）
    return f"{brand} {seg}" if seg[0].isdigit() else f"{brand}{seg}"


def parse_price_text(text: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    """
    解析文本价格 → (min_price, max_price)。

    支持：
      '3000以内/以下' → (None, 3000)
      '3000以上'       → (3000, None)
      '3000-5000'      → (3000, 5000)
      '3000左右'       → (2700, 3300)（±10%）
      '1万'            → (None, 10000)（默认视为预算上限）
    无法解析返回 (None, None)。
    """
    if not text:
        return None, None
    t = str(text).strip().replace("，", ",").replace("元", "").replace("块钱", "")
    unit = {"万": 10000, "千": 1000}

    def _num(v: str, u: Optional[str]) -> float:
        return float(v) * unit.get(u or "", 1)

    # 区间：3000-5000 / 3000到5000 / 3000~5000
    m = _PRICE_RE.search(t)
    if m:
        lo = _num(m.group(1), m.group(2))
        hi = _num(m.group(4), m.group(5))
        return min(lo, hi), max(lo, hi)

    # 单值 + 边界词
    m = _PRICE_TAIL_RE.search(t)
    if m:
        v = _num(m.group(1), m.group(2))
        bound = m.group(3) or ""
        if "以内" in bound or "以下" in bound:
            return None, v
        if "以上" in bound:
            return v, None
        if "左右" in bound:
            return round(v * 0.9, 2), round(v * 1.1, 2)
        # 裸数字：视为预算上限（电商语境多为"预算X元"）
        return None, v
    return None, None


def normalize_arguments(
    tool_name: str,
    arguments: Dict[str, Any],
    raw_query: str,
) -> Dict[str, Any]:
    """
    把 LLM 输出的 arguments 归一化为可执行检索参数（LLM 输出永不直通 SQL）。

    规则：
      - product_name：优先取模型参数并清洗；清洗为空则回退 raw_query 清洗；
        仍为空则置 None（由降级链/反问兜底）；
      - 文本价格（'3000以内'等）→ min_price / max_price 数值；
      - sku_code / category_* / in_stock_only 原样透传（后续 SQL 参数化）。
    """
    args = dict(arguments) if isinstance(arguments, dict) else {}
    norm: Dict[str, Any] = {}

    # ---- sku_code：透传（白名单由 repo 层参数化处理）----
    norm["sku_code"] = args.get("sku_code") or None

    # ---- product_name：清洗（模型参数 → 原始 query 兜底）----
    raw_name = (
        args.get("product_name")
        or args.get("product")
        or args.get("name")
        or args.get("title")
        or args.get("goods")
        or args.get("query")
        or args.get("keyword")
        or None
    )
    cleaned = clean_product_name(raw_name) or clean_product_name(raw_query)
    # 模型用歧义键 'category' 表达品类（如 {'category': '衬衫'}）时，清洗后并入候选
    if cleaned is None:
        cleaned = clean_product_name(args.get("category"))
    norm["product_name"] = cleaned

    # ---- 类目：透传（模型给的类目一般是规范值；若为整句则由降级链处理）----
    norm["category_big"] = args.get("category_big") or args.get("big_category") or None
    norm["category_small"] = args.get("category_small") or args.get("small_category") or None
    norm["category_path"] = args.get("category_path") or args.get("path") or None

    # ---- 价格：数值直接取；字符串走文本解析（LLM 输出兜底，min/max 分别解析避免拼接错误）----
    min_price = args.get("min_price")
    max_price = args.get("max_price")
    lo = hi = None
    if _is_num(min_price):
        lo = float(min_price)
    elif isinstance(min_price, str):
        lo, hi_tmp = parse_price_text(min_price)
        if hi_tmp is not None:
            hi = hi_tmp  # 区间文本（'3000-5000'）填单字段时保留下限/上限
    if _is_num(max_price):
        hi = float(max_price)
    elif isinstance(max_price, str):
        _, hi = parse_price_text(max_price)
    norm["min_price"], norm["max_price"] = lo, hi
    # 模型把价格文本塞进 query/product_name 时的兜底提取
    if lo is None and hi is None:
        lo2, hi2 = parse_price_text(raw_name or raw_query or "")
        if lo2 is not None or hi2 is not None:
            norm["min_price"], norm["max_price"] = lo2, hi2

    # ---- in_stock_only：布尔化 ----
    norm["in_stock_only"] = _to_bool(args.get("in_stock_only"))

    # ---- 工具特定的检索词（RAG / 推荐）----
    if tool_name in ("get_knowledge_base", "product_recommendation"):
        norm["query"] = cleaned or raw_query.strip() or None

    return norm


def _is_num(v: Any) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return True
    if isinstance(v, str):
        try:
            float(v.strip())
            return True
        except ValueError:
            return False
    return False


def _to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v > 0
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "是", "有", "有货")
    return False
