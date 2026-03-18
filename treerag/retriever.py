"""
retriever.py ── PageIndex 風格的簡化檢索器

核心設計：
  ✅ 單次 LLM Call 完成節點選擇（高效）
  ✅ 用 summary 導航，頁碼定位取出內容
  ✅ 無向量資料庫、無熱度機制、無分數傳播
  ✅ 保留多文件支援

檢索流程：
  User Query
      ↓
  [1] LLM 閱讀文件摘要 → 選 doc_id（若多文件）
      ↓
  [2] LLM 閱讀章節樹（含 summary）→ 選 node_id + relevance_score
      ↓
  [3] 用 start_page/end_page 取出原始內容
      ↓
  [4] 生成回答
"""

from dataclasses import dataclass
from utils import extract_json, retry_llm_call


# ════════════════════════════════════════════════════════
# 資料結構
# ════════════════════════════════════════════════════════

@dataclass
class MatchedNode:
    """匹配的節點"""
    node_id: str
    doc_id: str
    title: str
    start_page: int
    end_page: int
    filename: str
    relevance_score: float      # LLM 給的相關性分數 (0~1)
    content: str = ""           # 取出的原始內容


@dataclass
class RetrievalResult:
    """檢索結果"""
    query: str
    answer: str
    matched_nodes: list[MatchedNode]
    reasoning: str              # LLM 選擇節點的推理說明
    steps: list[str]            # 簡易軌跡

    def render_trace(self) -> str:
        """輸出可讀的檢索軌跡"""
        lines = [
            "=" * 60,
            f"🔍 查詢：{self.query}",
            "=" * 60,
            f"\n🧠 推理：{self.reasoning}",
        ]

        if self.matched_nodes:
            lines.append("\n" + "─" * 60)
            lines.append("📌 引用來源：")
            for node in self.matched_nodes:
                lines.append(
                    f"  [{node.relevance_score:.2f}] [{node.doc_id}] "
                    f"{node.title} (p{node.start_page}-{node.end_page})"
                )

        lines.append("\n" + "=" * 60)
        lines.append("💡 回答：")
        lines.append(self.answer)
        lines.append("=" * 60)
        return "\n".join(lines)


# ════════════════════════════════════════════════════════
# 主檢索器
# ════════════════════════════════════════════════════════

class SimpleRetriever:
    """
    PageIndex 風格的簡化檢索器

    與原 HierarchicalRetriever 的差異：
    - ❌ 移除 IntentAnalyzer（單一查詢即可）
    - ❌ 移除 memory_lifecycle（無熱度、分數傳播）
    - ❌ 移除對話歷史（純檢索，非對話）
    - ✅ 保留多文件支援
    - ✅ 單次 LLM Call 完成節點選擇（高效）
    """

    MAX_TOP_NODES = 5         # 最多取幾個節點
    MAX_DOCS = 3              # 多文件時最多選幾個

    def __init__(self, llm_client, fs_index, pdf_docs: dict):
        """
        Args:
            llm_client: LLMClient
            fs_index: FileSystemIndex
            pdf_docs: {doc_id: PDFDocument}
        """
        self.llm = llm_client
        self.fs = fs_index
        self.pdf_docs = pdf_docs

    def query(self, question: str, verbose: bool = True, target_doc_ids: list[str] = None):
        """
        主查詢入口 ── PageIndex 風格的簡化流程

        Args:
            question: 查詢問題
            verbose: 是否輸出詳細資訊
            target_doc_ids: 指定要搜尋的文件 ID 列表（若為 None 則自動選擇）
        """
        steps = []

        # 若未指定 target_doc_ids，使用所有文件
        if target_doc_ids is None:
            target_doc_ids = list(self.pdf_docs.keys())

        # ── Step 1: 若多文件，先選文件 ──
        if len(target_doc_ids) > 1:
            # 只建構目標文件的 L0 context
            l0_context = self._build_l0_context(target_doc_ids)
            selected_docs = self._select_docs(question, l0_context)
            selected_docs = selected_docs[:self.MAX_DOCS]
            steps.append(f"從 {len(target_doc_ids)} 份文件中選出 {len(selected_docs)} 份：{selected_docs}")
            if verbose:
                print(f"  [Step 1] 相關文件：{selected_docs}")
        else:
            selected_docs = target_doc_ids
            steps.append(f"搜尋文件：{selected_docs}")

        # ── Step 2: 在每份文件中選節點（核心！單次 LLM Call）──
        all_matched: list[MatchedNode] = []

        for doc_id in selected_docs:
            doc_index = self.fs.load_doc_index(doc_id)
            if not doc_index:
                continue

            pdf_doc = self.pdf_docs.get(doc_id)
            filename = doc_index.filename
            tree_view = doc_index.render_tree()

            # 單次 LLM Call 完成節點選擇
            reasoning, node_list = self._select_nodes(question, tree_view)

            for node_data in node_list[:self.MAX_TOP_NODES]:
                node_id = node_data.get("node_id", "")
                score = float(node_data.get("relevance_score", 0.5))

                node = doc_index.get_node(node_id)
                if not node:
                    continue

                # 取出內容
                content = ""
                if pdf_doc:
                    content = pdf_doc.get_range_text(node.start_page, node.end_page)

                all_matched.append(MatchedNode(
                    node_id=node_id,
                    doc_id=doc_id,
                    title=node.title,
                    start_page=node.start_page,
                    end_page=node.end_page,
                    filename=filename,
                    relevance_score=score,
                    content=content,
                ))

            steps.append(f"[{doc_id}] 找到 {len(node_list)} 個節點")
            if verbose:
                print(f"  [Step 2] [{doc_id}] 找到 {len(node_list)} 個節點")

        # 按相關性排序
        all_matched.sort(key=lambda n: n.relevance_score, reverse=True)

        steps.append(f"總共匹配 {len(all_matched)} 個節點")

        # ── Step 3: 生成回答 ──
        if not all_matched:
            return RetrievalResult(
                query=question,
                answer="在知識庫中找不到相關內容。",
                matched_nodes=[],
                reasoning="無匹配節點",
                steps=steps,
            )

        # 組合內容
        context_texts = []
        for node in all_matched[:self.MAX_TOP_NODES]:
            context_texts.append(
                f"【{node.filename}｜{node.title}｜第{node.start_page}-{node.end_page}頁】\n"
                f"{node.content}"
            )

        combined_context = "\n\n" + ("─" * 40 + "\n\n").join(context_texts)
        answer = self._generate_answer(question, combined_context)

        steps.append("已生成回答")

        return RetrievalResult(
            query=question,
            answer=answer,
            matched_nodes=all_matched[:self.MAX_TOP_NODES],
            reasoning=reasoning,
            steps=steps,
        )

    # ─────────────────────────────────────────────
    # Step 1: 文件選擇（多文件時）
    # ─────────────────────────────────────────────

    def _select_docs(self, query: str, l0_context: str) -> list[str]:
        """LLM 選擇相關文件"""
        prompt = f"""請判斷以下問題應該查詢哪些文件。

【問題】
{query}

【可查詢的文件】
{l0_context}

輸出 JSON：
{{
  "reasoning": "選擇理由（1 句）",
  "selections": [
    {{"doc_id": "文件 ID", "score": 0.9, "reason": "為什麼相關"}}
  ]
}}

規則：
- score 範圍 0.0 ~ 1.0（1.0 = 完全相關）
- 只選相關的文件（score >= 0.5）
- 按 score 降冪排序
- 若問題沒有明確指向特定文件，選擇所有文件
- 只輸出 JSON"""

        try:
            resp = retry_llm_call(
                self.llm,
                [{"role": "user", "content": prompt}],
                max_tokens=400,
                label="文件選擇",
            )
            data = extract_json(resp)
            if isinstance(data, dict):
                selections = data.get("selections", [])
                # 過濾 score >= 0.5 且 doc_id 存在
                valid_ids = [
                    s["doc_id"] for s in selections
                    if float(s.get("score", 0)) >= 0.5 and s.get("doc_id") in self.fs.entries
                ]
                if valid_ids:
                    return valid_ids
        except Exception as e:
            pass

        # fallback：返回所有文件
        return list(self.fs.entries.keys())

    def _build_l0_context(self, doc_ids: list[str]) -> str:
        """建構指定文件的 L0 上下文"""
        lines = [f"可查詢的文件共 {len(doc_ids)} 份："]
        for doc_id in doc_ids:
            entry = self.fs.entries.get(doc_id)
            if entry:
                lines.append(
                    f"  [{doc_id}] {entry.filename}（{entry.total_pages}頁）：{entry.doc_summary}"
                )
        return "\n".join(lines)

    # ─────────────────────────────────────────────
    # Step 2: 節點選擇（核心！單次 LLM Call）
    # ─────────────────────────────────────────────

    def _select_nodes(self, query: str, tree_view: str) -> tuple[str, list[dict]]:
        """
        單次 LLM Call 完成節點選擇 ── PageIndex 風格

        Returns:
            (reasoning, node_list)
            node_list: [{"node_id": "...", "relevance_score": 0.9, "reason": "..."}, ...]
        """
        prompt = f"""請在以下文件的章節結構中，找出與問題最相關的章節節點。

【問題】
{query}

【文件章節結構】
{tree_view}

輸出 JSON：
{{
  "reasoning": "選擇理由（2-3 句，說明為什麼這些節點相關）",
  "nodes": [
    {{
      "node_id": "完整的 node_id",
      "relevance_score": 0.9,
      "reason": "選擇原因"
    }}
  ]
}}

規則：
- relevance_score 範圍 0.0 ~ 1.0（1.0 = 完全相關）
- 找出所有可能包含答案的節點（1-5 個）
- node_id 必須完全匹配章節結構中的 [node_id]
- 優先選葉節點（最細層級），但若問題跨越多節也可選父節點
- 只輸出 JSON"""

        try:
            resp = retry_llm_call(
                self.llm,
                [{"role": "user", "content": prompt}],
                max_tokens=600,
                label="節點選擇",
            )
            data = extract_json(resp)
            if isinstance(data, dict):
                reasoning = data.get("reasoning", "")
                nodes = data.get("nodes", [])
                if isinstance(nodes, list) and nodes:
                    return reasoning, nodes
        except Exception as e:
            pass

        # fallback
        return "無法解析，返回空結果", []

    # ─────────────────────────────────────────────
    # Step 3: 生成回答
    # ─────────────────────────────────────────────

    def _generate_answer(self, question: str, context: str) -> str:
        """基於萃取的內容生成最終回答"""
        prompt = f"""請根據以下文件內容回答問題。

【問題】
{question}

【相關文件內容】
{context[:8000]}

回答要求：
1. 只依據提供的內容，不要憑空推測
2. 引用具體頁碼或章節名稱（例如：「根據第 21 頁的財務摘要...」）
3. 若資訊不足，說明哪部分缺乏
4. 用繁體中文回答（除非原文是英文）
5. 若內容包含多個來源，整合成連貫的回答"""

        return retry_llm_call(
            self.llm,
            [{"role": "user", "content": prompt}],
            max_tokens=1500,
            label="生成回答",
        )
