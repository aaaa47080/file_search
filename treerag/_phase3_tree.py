"""
_phase3_tree.py
─────────────────────────────────────────────────────
Phase 3: Tree building and node splitting
"""

from typing import TYPE_CHECKING, Optional, Union
from utils import extract_json, extract_json_array, retry_llm_call
from tree_models import TreeNode

if TYPE_CHECKING:
    from llm_client import LLMClient


class _Phase3TreeMixin:
    if TYPE_CHECKING:
        llm: "LLMClient"
        MAX_CHARS_PER_NODE: int
    """Phase 3: post_processing → 建立父子關係樹 + 切分超大葉節點"""

    # 對齊 PageIndex 的 max_page_num_each_node 預設值
    MAX_PAGES_PER_LEAF = 10

    def _build_tree(
        self, structure_list: list[dict], doc_id: str, total_pages: int
    ) -> TreeNode:
        """
        PageIndex 的 post_processing + list_to_tree 邏輯。

        核心修正：當 LLM 給出 "8.1"、"8.2" 但沒有 "8" 時，
        用「有效層級」取代原始層級，讓孤兒節點自動提升到最近存在的祖先之下，
        確保 end_page 計算和 parent_id 都正確。
        """
        if not structure_list:
            return TreeNode(
                node_id=f"{doc_id}://root",
                title="文件全文",
                summary="",
                start_page=1,
                end_page=total_pages,
                level=0,
                structure_code="root",
            )

        # 排序（按結構碼的數字順序）
        def sort_key(item):
            parts = str(item.get("structure", "0")).split(".")
            return [int(p) if p.isdigit() else 0 for p in parts]

        sorted_list = sorted(structure_list, key=sort_key)
        all_codes = {item.get("structure", "") for item in sorted_list}

        # 取得最近存在的父節點 code（跳過不存在的中間層）
        def get_effective_parent(code: str) -> str:
            parent = _raw_parent(code)
            while parent != "root" and parent not in all_codes:
                parent = _raw_parent(parent)
            return parent

        def _raw_parent(code: str) -> str:
            parts = code.split(".")
            return "root" if len(parts) <= 1 else ".".join(parts[:-1])

        # 預先計算每個節點的有效層級
        eff_levels: dict[str, int] = {}
        for item in sorted_list:
            code = item.get("structure", "")
            ep = get_effective_parent(code)
            eff_levels[code] = 1 if ep == "root" else eff_levels.get(ep, 1) + 1

        # 建立根節點與節點字典
        root = TreeNode(
            node_id=f"{doc_id}://root",
            title="文件全文",
            summary="",
            start_page=1,
            end_page=total_pages,
            level=0,
            structure_code="root",
        )
        nodes: dict[str, TreeNode] = {"root": root}

        for item in sorted_list:
            code = item.get("structure", "")
            if not code:
                continue
            node = TreeNode(
                node_id=f"{doc_id}://{code.replace('.', '/')}",
                title=item.get("title", "未命名"),
                summary="",
                start_page=item.get("physical_index", 1),
                end_page=total_pages,
                level=eff_levels.get(code, 1),
                structure_code=code,
            )
            nodes[code] = node

        # 設定 end_page：用有效層級找下一個同層或上層節點
        for i, item in enumerate(sorted_list):
            code = item.get("structure", "")
            maybe_node = nodes.get(code)
            if maybe_node is None:
                continue
            node = maybe_node
            eff_level = eff_levels.get(code, 1)

            next_page = total_pages
            for j in range(i + 1, len(sorted_list)):
                nxt = sorted_list[j]
                nxt_code = nxt.get("structure", "")
                nxt_eff_level = eff_levels.get(nxt_code, 1)
                if nxt_eff_level <= eff_level:
                    if nxt.get("appear_start", True):
                        next_page = int(nxt.get("physical_index", total_pages)) - 1
                    else:
                        next_page = int(nxt.get("physical_index", total_pages))
                    break
            node.end_page = max(node.start_page, next_page)

        # 建立父子關係（使用有效父節點）
        for code, node in nodes.items():
            if code == "root":
                continue
            parent_code = get_effective_parent(code)
            parent = nodes.get(parent_code, root)
            node.parent_id = parent.node_id
            parent.children.append(node)

        # 修正父節點的頁碼範圍（覆蓋所有子節點）
        self._fix_parent_pages(root)

        return root

    def _get_parent_code(self, code: str) -> str:
        """取得父節點的結構碼（1.2.3 → 1.2，1.2 → 1，1 → root）"""
        parts = code.split(".")
        if len(parts) <= 1:
            return "root"
        return ".".join(parts[:-1])

    def _fix_parent_pages(self, node: TreeNode):
        """遞迴修正父節點的頁碼範圍"""
        if node.children:
            for child in node.children:
                self._fix_parent_pages(child)
            child_start = min(c.start_page for c in node.children)
            child_end = max(c.end_page for c in node.children)
            if node.structure_code != "root":
                node.start_page = min(node.start_page, child_start)
                node.end_page = max(node.end_page, child_end)

    def _split_large_nodes(self, root: TreeNode, pdf_doc):
        """
        遞迴切分葉節點，對齊 PageIndex process_large_node_recursively() 的行為：

          若葉節點頁數 > MAX_PAGES_PER_LEAF（預設 10），
          用 LLM 在該頁碼範圍內再萃取子段落，
          並將原葉節點升格為「父節點」（其子節點才是新葉節點）。

        對齊官方：無 depth 限制，純靠雙重閾值終止（頁數 + 字元數）。
        若 LLM 找到的第一個子段落標題與父節點相同，跳過該項（防止標題重複套娃）。
        """

        def _try_split(node: TreeNode):
            # 非葉節點：繼續往下遞迴
            if node.children:
                for child in list(node.children):
                    _try_split(child)
                return

            # ── 對齊 PageIndex process_large_node_recursively 雙重閾值 ──
            # 官方條件：end - start > max_page_num  AND  token_num >= max_token_num
            # 兩個條件必須同時成立才切分（避免「頁數多但文字少」的表格/圖片頁）
            page_span = node.end_page - node.start_page + 1
            if page_span <= self.MAX_PAGES_PER_LEAF:
                return

            # 提取該節點頁面文字（完整，用於 char 計數；後續採樣送 LLM）
            text = pdf_doc.get_range_text(node.start_page, node.end_page)
            if not text.strip():
                return

            # token 數估算：len(text) chars ÷ 4 ≈ tokens（對齊 MAX_CHARS_PER_NODE）
            if len(text) < self.MAX_CHARS_PER_NODE:
                return  # 頁數多但文字量少，不切分（官方 token_num < max_token_num 條件）

            if len(text) > 8000:
                head = text[:3500]
                tail = text[-2000:]
                sample = f"{head}\n\n...[省略]...\n\n{tail}"
            else:
                sample = text

            prompt = f"""以下是文件章節「{node.title}」（第 {node.start_page}–{node.end_page} 頁）的內容。
請在此頁碼範圍內找出子段落或子章節，每個子段落請給出名稱和起始頁碼。

內容（採樣）：
{sample}

輸出 JSON 陣列（至少 2 個、最多 8 個子段落）：
[
  {{"title": "子段落名稱", "physical_index": 起始頁碼}},
  ...
]

規則：
- physical_index 必須在 {node.start_page} 到 {node.end_page} 之間
- 若此章節結構均一，找不到明確子段落，回傳 []
- 只輸出 JSON，不要其他文字"""

            try:
                resp = retry_llm_call(
                    self.llm,
                    [{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    label=f"切分:{node.title[:15]}",
                )
                sub_items = extract_json_array(resp)
                if not sub_items:
                    data = extract_json(resp)
                    sub_items = data if isinstance(data, list) else []

                if not sub_items or len(sub_items) < 2:
                    return  # LLM 認為無子結構，保留原大葉節點

                # ── 對齊 PageIndex page_index.py:1008 ──
                # 若第一個子段落標題與父節點相同，跳過（防止標題重複套娃）
                if (
                    sub_items
                    and sub_items[0].get("title", "").strip() == node.title.strip()
                ):
                    sub_items = sub_items[1:]
                if len(sub_items) < 2:
                    return

                # 驗證頁碼在父節點範圍內
                valid: list[dict[str, Union[int, str]]] = []
                for item in sub_items:
                    try:
                        page = int(item.get("physical_index", node.start_page))
                    except (ValueError, TypeError):
                        continue
                    if node.start_page <= page <= node.end_page:
                        valid.append(
                            {"title": str(item.get("title", "")).strip(), "page": page}
                        )

                if len(valid) < 2:
                    return

                valid.sort(key=lambda x: int(x["page"]))

                # 從父節點的 node_id 中取出 doc_id
                doc_id = node.node_id.split("://")[0]

                # 建立子節點，並掛在原葉節點下（升格為父節點）
                for i, v in enumerate(valid):
                    sub_start = int(v["page"])
                    sub_end = (
                        int(valid[i + 1]["page"]) - 1
                        if i + 1 < len(valid)
                        else node.end_page
                    )
                    sub_end = max(sub_start, sub_end)

                    sub_code = f"{node.structure_code}.{i + 1}"
                    sub_id = f"{doc_id}://{sub_code.replace('.', '/')}"

                    child = TreeNode(
                        node_id=sub_id,
                        title=str(v["title"]),
                        summary="",
                        start_page=sub_start,
                        end_page=sub_end,
                        level=node.level + 1,
                        structure_code=sub_code,
                        parent_id=node.node_id,
                    )
                    node.children.append(child)

                # ── 對齊 PageIndex process_large_node_recursively:1013 ──
                # 切分後父節點的 end_page 縮短為第一個子節點的起始頁，
                # 讓父節點只覆蓋「子節點前的引言頁」，避免範圍與子節點重疊。
                node.end_page = node.children[0].start_page

                print(
                    f"   → 切分大節點「{node.title}」"
                    f"（{page_span}頁）→ {len(node.children)} 個子節點"
                    f"，parent end_page 縮至 {node.end_page}"
                )

                # 遞迴：子節點可能還是太大
                for child in node.children:
                    _try_split(child)

            except Exception:
                pass  # 切分失敗，保留原大葉節點不變

        for child in root.children:
            _try_split(child)
