"""
_phase4_summary.py
─────────────────────────────────────────────────────
Phase 4 & 5: Summary generation for nodes and document
"""

from concurrent.futures import ThreadPoolExecutor
from utils import extract_json, retry_llm_call
from tree_models import TreeNode


class _Phase4SummaryMixin:
    """Phase 4+5: 生成節點摘要 + 文件整體摘要"""

    def _generate_summaries(self, root: TreeNode, pdf_doc):
        """
        為葉節點生成 .abstract（一句話）和 .overview（段落）
        父節點的摘要從子節點合成
        """
        all_nodes = root.flat_list()
        leaf_nodes = [n for n in all_nodes if not n.children and n.structure_code != "root"]

        def summarize_leaf(node):
            end = min(node.end_page, pdf_doc.total_pages)
            text = pdf_doc.get_range_text(node.start_page, end)
            if not text.strip():
                node.summary = "此節無文字內容"
                node.overview = ""
                return

            # 長節點用「首段 + 中段 + 末段」三點採樣，避免只看前幾頁
            # 短節點（< 8000 字元）直接使用全文
            FULL_LIMIT = 8000
            if len(text) <= FULL_LIMIT:
                content_for_llm = text
            else:
                head = text[:3500]
                mid_start = len(text) // 2 - 1000
                mid = text[mid_start: mid_start + 2000]
                tail = text[-2000:]
                content_for_llm = (
                    f"{head}\n\n...[中間省略，共 {len(text)} 字元]...\n\n"
                    f"{mid}\n\n...[末段]...\n\n{tail}"
                )

            prompt = f"""請為以下文件章節生成兩層摘要：

章節：{node.title}（第 {node.start_page}-{end} 頁）

內容（{'全文' if len(text) <= FULL_LIMIT else '首 / 中 / 末段採樣'}）：
{content_for_llm}

請輸出 JSON：
{{
  "abstract": "一句話摘要（不超過 60 字）",
  "overview": "2-4 句話的詳細概述（不超過 200 字）"
}}
只輸出 JSON。"""

            try:
                resp = retry_llm_call(self.llm, [{"role": "user", "content": prompt}], max_tokens=400, label=f"摘要:{node.title[:15]}")
                data = extract_json(resp)
                if isinstance(data, dict):
                    node.summary = str(data.get("abstract", ""))[:200]
                    node.overview = str(data.get("overview", ""))[:500]
                else:
                    node.summary = resp.strip()[:200]
                    node.overview = ""
            except Exception:
                node.summary = f"第 {node.start_page}-{end} 頁內容"

        # 並行生成葉節點摘要
        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            list(executor.map(summarize_leaf, leaf_nodes))

        # ── 自底向上：用 LLM 合成父節點摘要（真正的 bottom-up）──
        # reversed(all_nodes) 保證子節點先於父節點處理
        def synthesize_parent(node: TreeNode):
            """用子節點的 abstract 合成父節點摘要（LLM 呼叫）"""
            child_summaries = "\n".join(
                f"- {c.title}（p{c.start_page}-{c.end_page}）：{c.summary}"
                for c in node.children[:8]
                if c.summary
            )
            if not child_summaries:
                node.summary = "包含：" + "、".join(c.title for c in node.children[:5])
                node.overview = ""
                return

            prompt = f"""以下是文件章節「{node.title}」（第 {node.start_page}-{node.end_page} 頁）的各子節摘要：

{child_summaries}

請根據上述子節摘要，為整個章節生成兩層摘要：
{{
  "abstract": "一句話摘要（不超過 60 字，涵蓋整章重點）",
  "overview": "2-4 句話的章節概述（不超過 200 字，說明各子節的關係與整章意涵）"
}}
只輸出 JSON。"""

            try:
                resp = retry_llm_call(
                    self.llm,
                    [{"role": "user", "content": prompt}],
                    max_tokens=400,
                    label=f"章節摘要:{node.title[:15]}",
                )
                data = extract_json(resp)
                if isinstance(data, dict):
                    node.summary = str(data.get("abstract", ""))[:200]
                    node.overview = str(data.get("overview", ""))[:500]
                else:
                    # fallback：用子節點摘要拼接
                    node.summary = "包含：" + "、".join(c.title for c in node.children[:5])
                    node.overview = child_summaries
            except Exception:
                node.summary = "包含：" + "、".join(c.title for c in node.children[:5])
                node.overview = child_summaries

        for node in reversed(all_nodes):
            if node.children and not node.summary and node.structure_code != "root":
                synthesize_parent(node)

    @staticmethod
    def _build_tree_outline(node: "TreeNode", depth: int = 0, max_depth: int = 3) -> list:
        """
        將樹展開成縮排清單，每行含 title + summary。
        對齊 PageIndex generate_doc_description：把整棵樹結構送給 LLM。
        """
        if depth > max_depth:
            return []
        indent = "  " * depth
        summary_part = f"：{node.summary}" if node.summary else ""
        lines = [f"{indent}[{node.structure_code}] {node.title}（p{node.start_page}-{node.end_page}）{summary_part}"]
        for child in node.children:
            lines.extend(_Phase4SummaryMixin._build_tree_outline(child, depth + 1, max_depth))
        return lines

    def _generate_doc_summary(self, root: TreeNode, pdf_doc) -> tuple[str, str]:
        """
        生成整份文件的摘要和概述。

        對齊 PageIndex generate_doc_description：
        把整棵樹（所有節點 title + summary）序列化後送 LLM，
        讓 LLM 看到完整架構再生成文件摘要。
        """
        outline_lines = []
        for child in root.children:
            outline_lines.extend(self._build_tree_outline(child, depth=0))
        tree_outline = "\n".join(outline_lines) if outline_lines else "（無章節結構）"

        if not outline_lines:
            # fallback：樹為空時讀前 3 頁
            tree_outline = pdf_doc.get_range_text(1, min(3, pdf_doc.total_pages))[:2000]

        prompt = f"""請為這份文件生成整體摘要：

文件名稱：{pdf_doc.filename}（共 {pdf_doc.total_pages} 頁）

文件結構與各節摘要：
{tree_outline[:6000]}

輸出 JSON：
{{
  "abstract": "一句話摘要（不超過 80 字，說明整份文件的核心主旨）",
  "overview": "3-5 句話的文件概述（不超過 300 字，說明各章節的關係與文件整體架構）"
}}
只輸出 JSON。"""

        try:
            resp = retry_llm_call(self.llm, [{"role": "user", "content": prompt}], max_tokens=500, label="文件摘要")
            data = extract_json(resp)
            if isinstance(data, dict):
                return str(data.get("abstract", ""))[:300], str(data.get("overview", ""))[:600]
        except Exception:
            pass

        return f"{pdf_doc.filename}（{pdf_doc.total_pages}頁）", ""
