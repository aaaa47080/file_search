"""
filesystem_index.py
─────────────────────────────────────────────────────
跨文件的檔案系統索引（OpenViking L0/L1/L2 三層架構）

L0：整個知識庫的一句話摘要（< 200 tokens）
L1：所有文件的章節樹列表（< 2000 tokens）
L2：指定節點的原始頁面文字（按需載入）
"""

import json
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class FileSystemEntry:
    doc_id: str
    filename: str
    filepath: str
    doc_summary: str
    total_pages: int
    index_path: str
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "filepath": self.filepath,
            "doc_summary": self.doc_summary,
            "total_pages": self.total_pages,
            "index_path": self.index_path,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FileSystemEntry":
        return cls(
            doc_id=d["doc_id"],
            filename=d["filename"],
            filepath=d["filepath"],
            doc_summary=d["doc_summary"],
            total_pages=d["total_pages"],
            index_path=d["index_path"],
            tags=d.get("tags", []),
        )


class FileSystemIndex:
    FS_INDEX_FILE = "filesystem.index.json"

    def __init__(self, index_dir: str):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.fs_index_path = self.index_dir / self.FS_INDEX_FILE
        self.entries: dict[str, FileSystemEntry] = {}
        self._load()

    def _load(self):
        if self.fs_index_path.exists():
            with open(self.fs_index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for e in data.get("entries", []):
                    entry = FileSystemEntry.from_dict(e)
                    self.entries[entry.doc_id] = entry

    def _save(self):
        data = {
            "index_dir": str(self.index_dir),
            "entries": [e.to_dict() for e in self.entries.values()],
        }
        with open(self.fs_index_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def register(self, doc_index) -> FileSystemEntry:
        entry = FileSystemEntry(
            doc_id=doc_index.doc_id,
            filename=doc_index.filename,
            filepath=doc_index.filepath,
            doc_summary=doc_index.doc_summary,
            total_pages=doc_index.total_pages,
            index_path=str(self.index_dir / f"{doc_index.doc_id}.index.json"),
        )
        self.entries[doc_index.doc_id] = entry
        self._save()
        return entry

    def load_doc_index(self, doc_id: str):
        from tree_index import DocumentIndex
        entry = self.entries.get(doc_id)
        if not entry:
            return None
        p = Path(entry.index_path)
        if not p.exists():
            return None
        return DocumentIndex.load(str(p))

    # ─────────────────────────────────────────────
    # 三層式情境
    # ─────────────────────────────────────────────

    def get_L0_context(self) -> str:
        """L0：整個知識庫摘要（< 200 tokens）"""
        if not self.entries:
            return "知識庫為空。"
        lines = [f"知識庫共 {len(self.entries)} 份文件："]
        for e in self.entries.values():
            lines.append(f"  [{e.doc_id}] {e.filename}（{e.total_pages}頁）：{e.doc_summary}")
        return "\n".join(lines)

    def get_L1_context(self, doc_ids: list[str] = None) -> str:
        """L1：章節樹列表（中等 tokens）"""
        targets = doc_ids or list(self.entries.keys())
        parts = []
        for doc_id in targets:
            doc_idx = self.load_doc_index(doc_id)
            if doc_idx:
                parts.append(doc_idx.render_tree())
        return "\n\n".join(parts)

    def get_L2_context(self, doc_id: str, node_id: str, pdf_doc) -> str:
        """L2：節點原始文字（按需，高 tokens）"""
        doc_idx = self.load_doc_index(doc_id)
        if not doc_idx:
            return f"找不到文件 {doc_id}。"
        node = doc_idx.get_node(node_id)
        if not node:
            return f"找不到節點 {node_id}。"
        text = pdf_doc.get_range_text(node.start_page, node.end_page)
        return (
            f"【{doc_idx.filename}｜{node.title}｜"
            f"第 {node.start_page}-{node.end_page} 頁】\n"
            f"{'─' * 40}\n{text}"
        )

    def render_full_view(self) -> str:
        lines = ["=" * 60, "📚 知識庫索引", "=" * 60, "", self.get_L0_context(), ""]
        lines.append("─" * 60)
        lines.append("【L1 詳細結構】\n")
        lines.append(self.get_L1_context())
        return "\n".join(lines)
