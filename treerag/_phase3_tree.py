"""
_phase3_tree.py
─────────────────────────────────────────────────────
Phase 3: Tree building and node splitting
"""

from utils import extract_json, extract_json_array, retry_llm_call
from tree_models import TreeNode


class _Phase3TreeMixin:
    """Phase 3: post_processing → 建立父子關係樹 + 切分超大葉節點"""

    # 對齊 PageIndex 的 max_page_num_each_node 預設值
    MAX_PAGES_PER_LEAF = 10

    # 最大遞迴切分深度（防止在找不到明確子結構時無限套娃）
    MAX_SPLIT_DEPTH = 4

    def _build_tree(
        self, structure_list: list[dict], doc_id: str, total_pages: int
    ) -> TreeNode:
        """
        PageIndex 的 post_processing + list_to_tree 邏輯：
        1. 根據結構碼排序
        2. 為每個節點找父節點（結構碼截去最後一段）
        3. 計算頁碼範圍（start=自己，end=下一個同層或父層的前一頁）
        """
        if not structure_list:
            return TreeNode(
                node_id=f"{doc_id}://root",
                title="文件全文",
                summary="",
                overview="",
                start_page=1,
                end_page=total_pages,
                level=0,
                structure_code="root",
            )

        # 建立根節點
        root = TreeNode(
            node_id=f"{doc_id}://root",
            title="文件全文",
            summary="",
            overview="",
            start_page=1,
            end_page=total_pages,
            level=0,
            structure_code="root",
        )

        # 建立節點字典
        nodes: dict[str, TreeNode] = {"root": root}

        # 排序（按結構碼的數字順序）
        def sort_key(item):
            parts = str(item.get("structure", "0")).split(".")
            return [int(p) if p.isdigit() else 0 for p in parts]

        sorted_list = sorted(structure_list, key=sort_key)

        # 建立所有節點
        for item in sorted_list:
            code = item.get("structure", "")
            if not code:
                continue

            level = code.count(".") + 1
            node_id = f"{doc_id}://{code.replace('.', '/')}"

            node = TreeNode(
                node_id=node_id,
                title=item.get("title", "未命名"),
                summary="",
                overview="",
                start_page=item.get("physical_index", 1),
                end_page=total_pages,  # 暫時設為文件末頁，後面會修正
                level=level,
                structure_code=code,
            )
            nodes[code] = node

        # 設定頁碼範圍（end_page = 下一個同層或上層節點的前一頁）
        for i, item in enumerate(sorted_list):
            code = item.get("structure", "")
            node = nodes.get(code)
            if not node:
                continue

            # 找下一個頁碼
            next_page = total_pages
            for j in range(i + 1, len(sorted_list)):
                next_item = sorted_list[j]
                next_code = next_item.get("structure", "")
                next_level = next_code.count(".") + 1
                # 找到同層或上層的下一個節點
                if next_level <= node.level:
                    # Gap 3b: appear_start 影響邊界
                    # 若下一節標題出現在頁面開頭，前一節不包含該頁；
                    # 若標題在頁面中段（appear_start=False），前一節與下一節共用該頁。
                    if next_item.get("appear_start", True):
                        next_page = next_item.get("physical_index", total_pages) - 1
                    else:
                        next_page = next_item.get("physical_index", total_pages)
                    break
                # 找到任何下一個節點（用於計算葉節點邊界）
                if next_level > node.level and j == i + 1:
                    # 是自己的子節點，end_page 先不設
                    pass

            node.end_page = max(node.start_page, next_page)

        # 建立父子關係（PageIndex list_to_tree 邏輯）
        for code, node in nodes.items():
            if code == "root":
                continue
            parent_code = self._get_parent_code(code)
            parent = nodes.get(parent_code, root)
            node.parent_id = parent.node_id
            parent.children.append(node)

        # 修正父節點的頁碼範圍（應覆蓋所有子節點）
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

        修正項目（對齊 PageIndex page_index.py:1008）：
          1. 若 LLM 找到的第一個子段落標題與父節點相同，跳過該項（防止標題重複套娃）
          2. 加入 MAX_SPLIT_DEPTH 限制，避免切出過深的樹
        """

        def _try_split(node: TreeNode, depth: int = 0):
            # 超過最大深度：停止切分
            if depth >= self.MAX_SPLIT_DEPTH:
                return

            # 非葉節點：繼續往下遞迴
            if node.children:
                for child in list(node.children):
                    _try_split(child, depth + 1)
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
                if sub_items and sub_items[0].get("title", "").strip() == node.title.strip():
                    sub_items = sub_items[1:]
                if len(sub_items) < 2:
                    return

                # 驗證頁碼在父節點範圍內
                valid = []
                for item in sub_items:
                    try:
                        page = int(item.get("physical_index", node.start_page))
                    except (ValueError, TypeError):
                        continue
                    if node.start_page <= page <= node.end_page:
                        valid.append({"title": str(item.get("title", "")).strip(), "page": page})

                if len(valid) < 2:
                    return

                valid.sort(key=lambda x: x["page"])

                # 從父節點的 node_id 中取出 doc_id
                doc_id = node.node_id.split("://")[0]

                # 建立子節點，並掛在原葉節點下（升格為父節點）
                for i, v in enumerate(valid):
                    sub_start = v["page"]
                    sub_end = (
                        valid[i + 1]["page"] - 1
                        if i + 1 < len(valid)
                        else node.end_page
                    )
                    sub_end = max(sub_start, sub_end)

                    sub_code = f"{node.structure_code}.{i + 1}"
                    sub_id = f"{doc_id}://{sub_code.replace('.', '/')}"

                    child = TreeNode(
                        node_id=sub_id,
                        title=v["title"],
                        summary="",
                        overview="",
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
                    f"（{page_span}頁, depth={depth}）→ {len(node.children)} 個子節點"
                    f"，parent end_page 縮至 {node.end_page}"
                )

                # 遞迴：子節點可能還是太大（depth + 1）
                for child in node.children:
                    _try_split(child, depth + 1)

            except Exception:
                pass  # 切分失敗，保留原大葉節點不變

        for child in root.children:
            _try_split(child, depth=0)
