"""
tree_index.py
─────────────────────────────────────────────────────
樹狀索引建構（PageIndex page_index.py 的設計精髓 + OpenViking BuildingTree）

PageIndex 核心演算法：
  1. 用 LLM 從目錄頁萃取結構碼列表（e.g. 1.2.3 → depth 3）
  2. 用 LLM 驗證每個章節是否真的出現在指定頁（fuzzy matching）
  3. post_processing：從結構碼建立父子關係樹
  4. 為每個葉節點生成摘要

OpenViking BuildingTree：
  - URI 式節點 ID（doc://filename/ch1/sec2）
  - get_path_to_root()、flat_list()、get_children()
"""

import datetime
from _phase1_structure import _Phase1StructureMixin
from _phase2_verify import _Phase2VerifyMixin
from _phase3_tree import _Phase3TreeMixin
from _phase4_summary import _Phase4SummaryMixin
from tree_models import TreeNode, DocumentIndex


class TreeIndexBuilder(
    _Phase1StructureMixin,
    _Phase2VerifyMixin,
    _Phase3TreeMixin,
    _Phase4SummaryMixin,
):
    """
    基於 PageIndex page_index.py 的樹狀索引建構器

    Phase 1：從書籤 or LLM 取得結構碼列表
    Phase 2：LLM 驗證每個章節的實際頁碼（fuzzy matching）
    Phase 3：post_processing → 建立父子關係樹
    Phase 4：生成節點摘要（.abstract + .overview）
    """

    TOC_SCAN_PAGES = 10               # 掃描前幾頁尋找目錄
    # 並行驗證的執行緒數（PageIndex 用 ThreadPoolExecutor）
    MAX_WORKERS = 4
    # 無 TOC 時全文掃描的批次大小（約 10K tokens，1 token ≈ 4 chars）
    LLM_GROUP_CHARS = 40000
    # 切分閾值（對齊 PageIndex config.yaml max_token_num_each_node=20000）
    # 用字元數估算：20000 tokens × 4 chars/token = 80000 chars
    MAX_CHARS_PER_NODE = 80000

    def __init__(self, llm_client):
        self.llm = llm_client

    def build(self, pdf_doc, doc_id: str = None) -> DocumentIndex:
        doc_id = doc_id or pdf_doc.path.stem.replace(" ", "_").lower()
        print(f"\n🔍 建立索引：{pdf_doc.filename}（{pdf_doc.total_pages}頁）")

        # Phase 1: 取得結構列表
        print("   Phase 1: 萃取文件結構...")
        structure_list = self._get_structure_list(pdf_doc, doc_id)
        print(f"   → 找到 {len(structure_list)} 個章節候選")

        # Phase 2: 驗證頁碼（PageIndex 的 check_title_appearance）
        print("   Phase 2: 驗證章節頁碼...")
        structure_list = self._verify_page_numbers(structure_list, pdf_doc)

        # Phase 2b: 驗證失敗 fallback（accuracy ≤ 50% 或末頁 < 文件一半）
        if not structure_list:
            print("   → 驗證失敗，降級至 LLM 全文掃描...")
            structure_list = self._llm_extract_structure(pdf_doc, doc_id)
            structure_list = self._verify_page_numbers(structure_list, pdf_doc)

        if not structure_list:
            # LLM 全文掃描也失敗 → 再試 Pattern scan
            # Pattern scan 直接從 PDF 實體頁面抓標題行，不依賴 TOC 頁碼，不需驗證
            print("   → LLM 掃描失敗，嘗試 Pattern scan...")
            structure_list = self._scan_heading_patterns(pdf_doc)
            if structure_list:
                for item in structure_list:
                    item.setdefault("verified", True)
                    item.setdefault("appear_start", True)
                print(f"   → Pattern scan 找到 {len(structure_list)} 個章節（頁碼直接來自 PDF）")
            else:
                print("   → Pattern scan 亦失敗，以空結構繼續（單節點樹）")

        # Phase 2c: Gap 5 — 若首章節不從第 1 頁起，插入前言節點
        structure_list = self._add_preface_if_needed(structure_list)

        # Phase 3: 建立樹結構（PageIndex 的 post_processing + list_to_tree）
        print("   Phase 3: 建立樹狀結構...")
        root = self._build_tree(structure_list, doc_id, pdf_doc.total_pages)

        # Phase 3.5: 切分超大葉節點（對齊 PageIndex process_large_node_recursively）
        print(f"   Phase 3.5: 切分超大葉節點（>{self.MAX_PAGES_PER_LEAF}頁的葉節點）...")
        self._split_large_nodes(root, pdf_doc)

        # Phase 4: 生成摘要
        print("   Phase 4: 生成節點摘要...")
        self._generate_summaries(root, pdf_doc)

        # Phase 5: 文件整體摘要
        print("   Phase 5: 生成文件摘要...")
        doc_summary, doc_overview = self._generate_doc_summary(root, pdf_doc)

        node_count = len(root.flat_list())
        print(f"   ✅ 索引完成！{node_count} 個節點")

        return DocumentIndex(
            doc_id=doc_id,
            filename=pdf_doc.filename,
            filepath=pdf_doc.path,
            total_pages=pdf_doc.total_pages,
            doc_summary=doc_summary,
            doc_overview=doc_overview,
            root=root,
            created_at=datetime.datetime.now().isoformat(),
            heading_source=pdf_doc.heading_source,
        )


# Re-export for backward compatibility
__all__ = ["TreeIndexBuilder", "TreeNode", "DocumentIndex"]
