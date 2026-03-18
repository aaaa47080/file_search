"""
utils.py
─────────────────────────────────────────────────────
工具函式（移植自 PageIndex pageindex/utils.py）

核心功能：
  - extract_json()：4 階段 robust JSON 萃取（官方邏輯）
  - retry_llm_call()：10 次重試機制
  - count_tokens()：tiktoken token 計算
  - 樹狀結構工具：get_leaf_nodes()、structure_to_list()
"""

import json
import re
import time
from typing import Optional, Any


# ════════════════════════════════════════════════════════
# JSON 萃取（PageIndex utils.py 的 extract_json）
# ════════════════════════════════════════════════════════

def extract_json(text: str) -> Any:
    """
    4 階段 robust JSON 萃取（直接移植自 PageIndex utils.py）

    Stage 1: 直接解析
    Stage 2: 萃取 {} 或 [] 區塊後解析
    Stage 3: 清理常見問題後再解析
              - Python None → JSON null
              - 移除換行
              - 正規化空白
    Stage 4: 清理尾逗號後最後嘗試
    """
    if not text:
        return {}

    # Stage 1: 直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Stage 2: 萃取 JSON 區塊
    # 先試物件 {}
    obj_match = re.search(r'\{[\s\S]*\}', text)
    if obj_match:
        try:
            return json.loads(obj_match.group(0))
        except json.JSONDecodeError:
            pass

    # 再試陣列 []
    arr_match = re.search(r'\[[\s\S]*\]', text)
    if arr_match:
        try:
            return json.loads(arr_match.group(0))
        except json.JSONDecodeError:
            pass

    # Stage 3: 清理後重試（物件和陣列都試）
    for match in [obj_match, arr_match]:
        if not match:
            continue
        json_content = match.group(0)
        # 修正常見問題
        json_content = json_content.replace('None', 'null')
        json_content = json_content.replace('True', 'true')
        json_content = json_content.replace('False', 'false')
        json_content = json_content.replace('\n', ' ').replace('\r', ' ')
        json_content = ' '.join(json_content.split())  # 正規化空白
        try:
            return json.loads(json_content)
        except json.JSONDecodeError:
            pass

        # Stage 4: 清理尾逗號
        json_content = re.sub(r',\s*([}\]])', r'\1', json_content)
        try:
            return json.loads(json_content)
        except json.JSONDecodeError:
            pass

    return {}


def extract_json_array(text: str) -> list:
    """專門萃取 JSON 陣列，返回 list（失敗時返回空 list）"""
    result = extract_json(text)
    if isinstance(result, list):
        return result
    # 若萃取到物件但有陣列欄位，嘗試常見欄位名
    if isinstance(result, dict):
        for key in ("items", "list", "results", "data", "nodes", "queries", "selections"):
            if key in result and isinstance(result[key], list):
                return result[key]
    return []


# ════════════════════════════════════════════════════════
# LLM 重試機制（PageIndex 官方：10 次重試、1 秒間隔）
# ════════════════════════════════════════════════════════

def retry_llm_call(
    llm_client,
    messages: list[dict],
    max_tokens: int = 1000,
    max_retries: int = 10,
    sleep_seconds: float = 1.0,
    label: str = "",
) -> str:
    """
    帶重試的 LLM 呼叫（PageIndex 官方邏輯：10 次重試）

    失敗原因通常是：rate limit、timeout、API 暫時不可用
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            return llm_client.chat(messages, max_tokens=max_tokens)
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait = sleep_seconds * (1 + attempt * 0.5)  # 漸進式等待
                if label:
                    print(f"    [重試 {attempt+1}/{max_retries}] {label}: {e}（等待 {wait:.1f}s）")
                time.sleep(wait)

    raise RuntimeError(f"LLM 呼叫失敗（{max_retries} 次嘗試後）：{last_error}")


def retry_llm_json(
    llm_client,
    messages: list[dict],
    max_tokens: int = 1000,
    max_retries: int = 10,
    expect_array: bool = False,
    label: str = "",
) -> Any:
    """
    帶重試的 LLM 呼叫 + 自動 JSON 解析
    """
    text = retry_llm_call(llm_client, messages, max_tokens, max_retries, label=label)
    if expect_array:
        return extract_json_array(text)
    return extract_json(text)


# ════════════════════════════════════════════════════════
# Token 計算（PageIndex 官方 count_tokens）
# ════════════════════════════════════════════════════════

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """
    計算文字的 token 數（PageIndex 官方邏輯）
    使用 tiktoken，不可用時 fallback 到字元數估算
    """
    try:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")  # GPT-4 通用
        return len(enc.encode(text))
    except ImportError:
        # Fallback：中文約 1 token/字，英文約 4 chars/token
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars
        return chinese_chars + other_chars // 4


def truncate_to_tokens(text: str, max_tokens: int, model: str = "gpt-4o-mini") -> str:
    """截斷文字到指定 token 數"""
    if count_tokens(text, model) <= max_tokens:
        return text

    # 二分搜尋找截斷點
    lo, hi = 0, len(text)
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if count_tokens(text[:mid], model) <= max_tokens:
            lo = mid
        else:
            hi = mid
    return text[:lo] + "...(截斷)"


# ════════════════════════════════════════════════════════
# 樹狀結構工具（PageIndex 官方 utils）
# ════════════════════════════════════════════════════════

def get_leaf_nodes(node) -> list:
    """
    取得所有葉節點（PageIndex get_leaf_nodes）
    葉節點 = 沒有子節點的節點
    """
    if not node.children:
        return [node]
    leaves = []
    for child in node.children:
        leaves.extend(get_leaf_nodes(child))
    return leaves


def structure_to_list(root) -> list:
    """
    將樹狀結構扁平化為列表（PageIndex structure_to_list）
    深度優先，根節點排第一
    """
    return root.flat_list()


def is_leaf_node(node) -> bool:
    return len(node.children) == 0


def get_parent_structure(structure_code: str) -> Optional[str]:
    """
    取得父節點的結構碼（PageIndex 邏輯）
    "1.2.3" → "1.2"
    "1.2"   → "1"
    "1"     → None
    """
    parts = structure_code.split(".")
    if len(parts) <= 1:
        return None
    return ".".join(parts[:-1])


def structure_depth(structure_code: str) -> int:
    """結構碼的深度（"1.2.3" → 3）"""
    return len(structure_code.split("."))
