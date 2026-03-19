"""
tree_models.py
─────────────────────────────────────────────────────
Data models for tree index: TreeNode and DocumentIndex
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TreeNode:
    """樹狀索引節點，含 OpenViking 風格的 URI"""

    node_id: str  # e.g. "tsmc_2025q1://1/1.1"
    title: str
    summary: str  # 一句話摘要
    start_page: int
    end_page: int
    level: int  # 0=root, 1=章, 2=節, ...
    structure_code: str  # PageIndex 風格：e.g. "1.2.3"
    parent_id: Optional[str] = None
    children: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "title": self.title,
            "summary": self.summary,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "level": self.level,
            "structure_code": self.structure_code,
            "parent_id": self.parent_id,
            "children": [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TreeNode":
        node = cls(
            node_id=d["node_id"],
            title=d["title"],
            summary=d.get("summary", ""),
            start_page=d["start_page"],
            end_page=d["end_page"],
            level=d["level"],
            structure_code=d.get("structure_code", ""),
            parent_id=d.get("parent_id"),
        )
        node.children = [cls.from_dict(c) for c in d.get("children", [])]
        return node

    def flat_list(self) -> list["TreeNode"]:
        result = [self]
        for c in self.children:
            result.extend(c.flat_list())
        return result

    def get_children(self) -> list["TreeNode"]:
        return self.children

    def render_brief(self) -> str:
        """用於 LLM 導航的單行顯示"""
        indent = "  " * self.level
        tag = "📂" if self.children else "📄"
        summary_preview = f": {self.summary[:60]}..." if self.summary else ""
        return (
            f"{indent}{tag} [{self.node_id}] {self.title} "
            f"(p{self.start_page}-{self.end_page}){summary_preview}"
        )


@dataclass
class DocumentIndex:
    """完整文件索引：TreeNode 樹 + 元資料"""

    doc_id: str
    filename: str
    filepath: str
    total_pages: int
    doc_summary: str  # 整份文件摘要
    doc_overview: str  # 整份文件概述
    root: TreeNode
    created_at: str = ""
    heading_source: str = ""  # bookmarks | font | llm

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "filepath": self.filepath,
            "total_pages": self.total_pages,
            "doc_summary": self.doc_summary,
            "doc_overview": self.doc_overview,
            "created_at": self.created_at,
            "heading_source": self.heading_source,
            "root": self.root.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DocumentIndex":
        return cls(
            doc_id=d["doc_id"],
            filename=d["filename"],
            filepath=d["filepath"],
            total_pages=d["total_pages"],
            doc_summary=d["doc_summary"],
            doc_overview=d.get("doc_overview", ""),
            created_at=d.get("created_at", ""),
            heading_source=d.get("heading_source", ""),
            root=TreeNode.from_dict(d["root"]),
        )

    def save(self, index_dir: str) -> str:
        Path(index_dir).mkdir(parents=True, exist_ok=True)
        out_path = Path(index_dir) / f"{self.doc_id}.index.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return str(out_path)

    @classmethod
    def load(cls, index_path: str) -> "DocumentIndex":
        with open(index_path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def get_node(self, node_id: str) -> Optional[TreeNode]:
        for n in self.root.flat_list():
            if n.node_id == node_id:
                return n
        return None

    def get_children_of(self, node_id: str) -> list[TreeNode]:
        node = self.get_node(node_id)
        return node.children if node else []

    def render_tree(self) -> str:
        lines = [
            f"📄 [{self.doc_id}] {self.filename} ({self.total_pages}頁)",
            f"   摘要：{self.doc_summary}",
            "",
        ]
        self._render_node(self.root, lines, depth=0)
        return "\n".join(lines)

    def _render_node(self, node: TreeNode, lines: list, depth: int):
        lines.append(node.render_brief())
        for child in node.children:
            self._render_node(child, lines, depth + 1)
