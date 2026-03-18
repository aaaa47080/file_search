"""
memory_lifecycle.py
─────────────────────────────────────────────────────
節點熱度評分（直接移植自 OpenViking openviking/retrieve/memory_lifecycle.py）

公式：
  hotness = sigmoid(log1p(access_count)) × exp(-decay_rate × age_days)

其中：
  decay_rate = ln(2) / half_life_days  （預設 half_life = 7 天）

用途：
  在檢索時，用熱度分數混合語意相關性分數：
  final_score = (1 - alpha) × semantic_score + alpha × hotness_score
  （OpenViking 預設 alpha = 0.2）
"""

import math
import datetime
from typing import Optional


def hotness_score(
    access_count: int,
    last_accessed: Optional[str],
    half_life_days: float = 7.0,
    now: Optional[datetime.datetime] = None,
) -> float:
    """
    計算節點熱度分數（0.0 ~ 1.0）

    Args:
        access_count: 被存取次數
        last_accessed: ISO 格式的最後存取時間（可為 None）
        half_life_days: 衰減半衰期（天），預設 7 天
        now: 當前時間（測試用，生產環境為 None）

    Returns:
        float: 0.0 ~ 1.0 的熱度分數
    """
    # 1. 頻率分數：sigmoid(log1p(count))
    frequency = _sigmoid(math.log1p(access_count))

    # 2. 時間衰減分數
    if last_accessed:
        recency = _time_decay(last_accessed, half_life_days, now)
    else:
        # 從未存取：recency = 0（全新節點）
        recency = 0.0

    return frequency * recency


def blend_scores(
    semantic_score: float,
    access_count: int,
    last_accessed: Optional[str],
    alpha: float = 0.2,
    half_life_days: float = 7.0,
) -> float:
    """
    混合語意分數和熱度分數（OpenViking HierarchicalRetriever 邏輯）

    final = (1 - alpha) × semantic + alpha × hotness

    Args:
        semantic_score: LLM 推理給出的相關性分數（0.0 ~ 1.0）
        access_count: 存取次數
        last_accessed: 最後存取時間
        alpha: 熱度權重（OpenViking 預設 0.2）
    """
    hot = hotness_score(access_count, last_accessed, half_life_days)
    return (1.0 - alpha) * semantic_score + alpha * hot


def score_propagation(
    current_score: float,
    parent_score: float,
    alpha: float = 0.5,
) -> float:
    """
    OpenViking 層級分數傳播公式：
    final = alpha × current + (1 - alpha) × parent

    用於在樹狀結構中，讓父節點的相關性影響子節點分數
    （防止孤立的低相關子節點被過濾）
    """
    return alpha * current_score + (1.0 - alpha) * parent_score


def record_access(node) -> None:
    """更新節點的存取記錄"""
    node.access_count += 1
    node.last_accessed = datetime.datetime.utcnow().isoformat()


# ════════════════════════════════════════════════════════
# 內部工具函式
# ════════════════════════════════════════════════════════

def _sigmoid(x: float) -> float:
    """Sigmoid 函式：1 / (1 + e^(-x))"""
    return 1.0 / (1.0 + math.exp(-x))


def _time_decay(
    last_accessed_iso: str,
    half_life_days: float,
    now: Optional[datetime.datetime] = None,
) -> float:
    """
    指數衰減：exp(-decay_rate × age_days)
    decay_rate = ln(2) / half_life_days
    """
    if now is None:
        now = datetime.datetime.utcnow()

    try:
        last = datetime.datetime.fromisoformat(
            last_accessed_iso.replace("Z", "+00:00")
        )
        # 統一轉為 naive UTC
        if last.tzinfo is not None:
            last = last.replace(tzinfo=None) + last.utcoffset()
        age_days = (now - last).total_seconds() / 86400.0
        age_days = max(0.0, age_days)
    except (ValueError, TypeError, AttributeError):
        return 0.0

    decay_rate = math.log(2) / half_life_days
    return math.exp(-decay_rate * age_days)
