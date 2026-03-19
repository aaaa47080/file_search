"""
_phase4_summary.py
─────────────────────────────────────────────────────
Phase 4 & 5: Summary generation for nodes and document
"""

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING
from utils import extract_json, retry_llm_call
from tree_models import TreeNode

if TYPE_CHECKING:
    from llm_client import LLMClient


class _Phase4SummaryMixin:
    if TYPE_CHECKING:
        llm: "LLMClient"
        MAX_WORKERS: int
    """Phase 4+5: 生成節點摘要 + 文件整體摘要"""

    def _generate_summaries(self, root: TreeNode, pdf_doc):
        """為葉節點生成一句話摘要，父節點摘要從子節點合成"""
        all_nodes = root.flat_list()
        leaf_nodes = [
            n for n in all_nodes if not n.children and n.structure_code != "root"
        ]

        def summarize_leaf(node):
            end = min(node.end_page, pdf_doc.total_pages)
            text = pdf_doc.get_range_text(node.start_page, end)
            if not text.strip():
                node.summary = "此節無文字內容"
                return

            FULL_LIMIT = 8000
            if len(text) <= FULL_LIMIT:
                content_for_llm = text
            else:
                head = text[:3500]
                mid_start = len(text) // 2 - 1000
                mid = text[mid_start : mid_start + 2000]
                tail = text[-2000:]
                content_for_llm = (
                    f"{head}\n\n...[中間省略，共 {len(text)} 字元]...\n\n"
                    f"{mid}\n\n...[末段]...\n\n{tail}"
                )

            prompt = f"""You are given a part of a document, your task is to generate a description of the partial document about what are main points covered in the partial document.

Partial Document Text: {content_for_llm}

Directly return the description, do not include any other text."""

            try:
                resp = retry_llm_call(
                    self.llm,
                    [{"role": "user", "content": prompt}],
                    max_tokens=800,
                    label=f"摘要:{node.title[:15]}",
                )
                node.summary = resp.strip()[:3000]
            except Exception:
                node.summary = f"第 {node.start_page}-{end} 頁內容"

        # 並行生成葉節點摘要
        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            list(executor.map(summarize_leaf, leaf_nodes))

        # 自底向上：用子節點摘要合成父節點摘要
        def synthesize_parent(node: TreeNode):
            child_summaries = "\n".join(
                f"- {c.title}：{c.summary}" for c in node.children[:8] if c.summary
            )
            if not child_summaries:
                node.summary = "包含：" + "、".join(c.title for c in node.children[:5])
                return

            prompt = f"""You are given a part of a document, your task is to generate a description of the partial document about what are main points covered in the partial document.

Partial Document Text: {child_summaries}

Directly return the description, do not include any other text."""

            try:
                resp = retry_llm_call(
                    self.llm,
                    [{"role": "user", "content": prompt}],
                    max_tokens=800,
                    label=f"章節摘要:{node.title[:15]}",
                )
                node.summary = resp.strip()[:3000]
            except Exception:
                node.summary = "包含：" + "、".join(c.title for c in node.children[:5])

        for node in reversed(all_nodes):
            if node.children and not node.summary and node.structure_code != "root":
                synthesize_parent(node)

    @staticmethod
    def _build_tree_outline(
        node: "TreeNode", depth: int = 0, max_depth: int = 3
    ) -> list:
        """將樹展開成縮排清單，每行含 title + summary"""
        if depth > max_depth:
            return []
        indent = "  " * depth
        summary_part = f"：{node.summary}" if node.summary else ""
        lines = [
            f"{indent}[{node.structure_code}] {node.title}（p{node.start_page}-{node.end_page}）{summary_part}"
        ]
        for child in node.children:
            lines.extend(
                _Phase4SummaryMixin._build_tree_outline(child, depth + 1, max_depth)
            )
        return lines

    def _generate_doc_summary(self, root: TreeNode, pdf_doc) -> tuple[str, str]:
        """
        生成整份文件摘要（對齊 PageIndex generate_doc_description）。
        把整棵樹結構送給 LLM 再生成摘要。
        """
        outline_lines = []
        for child in root.children:
            outline_lines.extend(self._build_tree_outline(child, depth=0))
        tree_outline = "\n".join(outline_lines) if outline_lines else "（無章節結構）"

        if not outline_lines:
            tree_outline = pdf_doc.get_range_text(1, min(3, pdf_doc.total_pages))[:2000]

        prompt = f"""Your are an expert in generating descriptions for a document. You are given a structure of a document. Your task is to generate a one-sentence description for the document, which makes it easy to distinguish the document from other documents.

Document Structure: {tree_outline[:6000]}

Directly return the description, do not include any other text."""

        try:
            resp = retry_llm_call(
                self.llm,
                [{"role": "user", "content": prompt}],
                max_tokens=150,
                label="文件摘要",
            )
            return resp.strip()[:300], ""
        except Exception:
            pass

        return f"{pdf_doc.filename}（{pdf_doc.total_pages}頁）", ""
