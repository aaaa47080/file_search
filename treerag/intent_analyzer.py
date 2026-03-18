"""
intent_analyzer.py
─────────────────────────────────────────────────────
查詢意圖分析（直接移植自 OpenViking openviking/retrieve/intent_analyzer.py）

功能：
  分析對話歷史 + 當前問題 → 生成多個 TypedQuery
  讓檢索器可以從多個角度搜尋，提高召回率

原始邏輯：
  - 輸入：session 摘要 + 最近 5 則訊息 + 當前問題
  - 輸出：QueryPlan（多個 TypedQuery，每個有 intent + priority）
"""

from dataclasses import dataclass
from typing import Optional

from utils import extract_json, retry_llm_call


@dataclass
class TypedQuery:
    """
    OpenViking 風格的結構化查詢
    """
    query: str                          # 查詢文字
    intent: str                         # 意圖描述（e.g. "找財務數字", "找風險因素"）
    priority: int                       # 1-5（5 最高）
    context_type: Optional[str] = None  # "resource" | "memory" | "skill" | None


@dataclass
class QueryPlan:
    """
    意圖分析結果：多個 TypedQuery + 推理說明
    """
    queries: list[TypedQuery]
    session_context: str                # 對話歷史摘要
    reasoning: str                      # LLM 的推理說明

    def top_queries(self, n: int = 3) -> list[TypedQuery]:
        """取前 n 個優先度最高的查詢"""
        return sorted(self.queries, key=lambda q: q.priority, reverse=True)[:n]


class IntentAnalyzer:
    """
    OpenViking IntentAnalyzer 的移植實作

    相較原版的差異：
    - 移除了 VikingDB 相依
    - 使用統一的 LLMClient 介面
    - 加入繁體中文支援
    """

    MAX_RECENT_MESSAGES = 5
    MAX_CONTEXT_CHARS = 30000   # 約 10,000 tokens

    def __init__(self, llm_client):
        self.llm = llm_client

    def analyze(
        self,
        current_message: str,
        recent_messages: list[dict] = None,
        session_summary: str = "",
        context_type: Optional[str] = None,
    ) -> QueryPlan:
        """
        分析查詢意圖，生成多個 TypedQuery

        Args:
            current_message: 當前用戶問題
            recent_messages: 最近的對話歷史 [{"role": "user/assistant", "content": "..."}]
            session_summary: 對話歷史的壓縮摘要
            context_type: 限制查詢類型 ("resource" | None)
        """
        recent = (recent_messages or [])[-self.MAX_RECENT_MESSAGES:]
        summary = self._truncate(session_summary, self.MAX_CONTEXT_CHARS)

        prompt = self._build_prompt(
            current_message=current_message,
            recent_messages=recent,
            session_summary=summary,
            context_type=context_type,
        )

        try:
            response = retry_llm_call(self.llm, [{"role": "user", "content": prompt}], max_tokens=800, label="意圖分析")
            return self._parse_response(response, current_message)
        except Exception as e:
            # fallback：直接用原始問題作為單一查詢
            return QueryPlan(
                queries=[TypedQuery(
                    query=current_message,
                    intent="直接查詢",
                    priority=5,
                    context_type=context_type,
                )],
                session_context=session_summary[:200],
                reasoning=f"意圖分析失敗，使用原始問題：{e}",
            )

    def _build_prompt(
        self,
        current_message: str,
        recent_messages: list[dict],
        session_summary: str,
        context_type: Optional[str],
    ) -> str:
        """建立意圖分析 prompt（仿 OpenViking retrieval.intent_analysis）"""

        # 格式化最近訊息
        if recent_messages:
            msg_lines = []
            for msg in recent_messages:
                role = "用戶" if msg.get("role") == "user" else "助手"
                content = str(msg.get("content", ""))[:300]
                msg_lines.append(f"  [{role}]: {content}")
            recent_str = "\n".join(msg_lines)
        else:
            recent_str = "（無）"

        context_hint = ""
        if context_type:
            context_hint = f"\n限制查詢類型：{context_type}"

        return f"""你是一個查詢意圖分析器。請分析用戶的當前問題，生成多個搜尋查詢以提高召回率。

【對話歷史摘要】
{session_summary or "（無歷史）"}

【最近對話】
{recent_str}

【當前問題】
{current_message}
{context_hint}

請輸出 JSON：
{{
  "reasoning": "分析這個問題需要搜尋什麼資訊的推理（1-2句）",
  "queries": [
    {{
      "query": "搜尋查詢文字（可與原問題不同，更有利於定位）",
      "intent": "這個查詢的意圖（e.g. 找具體數字、找背景說明、找比較分析）",
      "priority": 5
    }},
    {{
      "query": "第二個搜尋角度",
      "intent": "另一個意圖",
      "priority": 3
    }}
  ]
}}

規則：
- 最多 3 個查詢，最少 1 個
- priority 1-5（5 最高）
- 查詢語言應與問題一致
- 若問題明確，1 個查詢即可；若問題模糊或跨主題，生成 2-3 個
- 只輸出 JSON"""

    def _parse_response(self, response: str, original_query: str) -> QueryPlan:
        """解析 LLM 回應（使用 PageIndex 官方 4-stage extract_json）"""
        data = extract_json(response)
        if not isinstance(data, dict):
            return self._fallback_plan(original_query)

        try:
            reasoning = data.get("reasoning", "")
            raw_queries = data.get("queries", [])

            queries = []
            for q in raw_queries:
                if not isinstance(q, dict):
                    continue
                query_text = str(q.get("query", original_query)).strip()
                if not query_text:
                    continue
                queries.append(TypedQuery(
                    query=query_text,
                    intent=str(q.get("intent", "查詢")),
                    priority=int(q.get("priority", 3)),
                ))

            if not queries:
                return self._fallback_plan(original_query)

            return QueryPlan(
                queries=queries,
                session_context="",
                reasoning=reasoning,
            )

        except Exception:
            return self._fallback_plan(original_query)

    def _fallback_plan(self, query: str) -> QueryPlan:
        return QueryPlan(
            queries=[TypedQuery(query=query, intent="直接查詢", priority=5)],
            session_context="",
            reasoning="使用原始查詢",
        )

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        if len(text) > max_chars:
            return text[:max_chars] + "...(截斷)"
        return text
