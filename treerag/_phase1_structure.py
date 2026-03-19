"""
_phase1_structure.py
─────────────────────────────────────────────────────
Phase 1: Structure extraction (bookmarks, TOC, LLM full-text scan)

官方 PageIndex 三條路徑：
  1. PDF 書籤
  2a. TOC 含頁碼   → process_toc_with_page_numbers
  2b. TOC 無頁碼   → process_toc_no_page_numbers（掃全文補頁碼）
  3.  無 TOC       → process_no_toc（全文掃描）
"""

import json
import re
from typing import TYPE_CHECKING
from utils import extract_json, extract_json_array, retry_llm_call

if TYPE_CHECKING:
    from llm_client import LLMClient


class _Phase1StructureMixin:
    if TYPE_CHECKING:
        llm: "LLMClient"
        TOC_SCAN_PAGES: int
        LLM_GROUP_CHARS: int
    """Phase 1: 從書籤 or LLM 取得結構碼列表"""

    def _get_structure_list(self, pdf_doc, doc_id: str) -> list[dict]:
        """
        官方 PageIndex 四條路徑（互斥）：
          1. PDF 書籤
          2a. TOC 含頁碼   → _detect_and_parse_toc
          2b. TOC 無頁碼   → _detect_toc_titles + _process_toc_no_page_numbers
          3.  無 TOC       → _llm_extract_structure（全文掃描）
        _toc_detected 決定 Phase 2b 是否允許降級至全文掃描。
        """
        self._toc_detected = False

        # 路徑 1：PDF 書籤
        if pdf_doc.bookmarks:
            self._toc_detected = True
            return self._bookmarks_to_structure(pdf_doc.bookmarks)

        # 路徑 2a：TOC 含頁碼
        toc_result = self._detect_and_parse_toc(pdf_doc)
        if toc_result and len(toc_result) >= 3:
            self._toc_detected = True
            print(f"   → 從目錄頁解析到 {len(toc_result)} 個章節（含頁碼）")
            return toc_result

        # 路徑 2b：TOC 無頁碼（官方 process_toc_no_page_numbers）
        toc_titles = self._detect_toc_titles(pdf_doc)
        if toc_titles and len(toc_titles) >= 3:
            self._toc_detected = True
            print(
                f"   → 偵測到目錄（{len(toc_titles)} 個章節，無頁碼），掃描全文補充頁碼..."
            )
            result = self._process_toc_no_page_numbers(pdf_doc, toc_titles)
            if result and len(result) >= 3:
                return result
            print("   → 無法從全文比對頁碼，改走全文掃描")

        # 路徑 3：無 TOC → 全文掃描（官方 process_no_toc）
        return self._llm_extract_structure(pdf_doc, doc_id)

    def _bookmarks_to_structure(self, bookmarks) -> list[dict]:
        """將 PDF 書籤轉成結構碼列表"""
        result = []
        counters: dict[int, int] = {}  # level → 累計數

        def _process(bms, parent_code=""):
            for bm in bms:
                level = bm.level
                counters[level] = counters.get(level, 0) + 1
                # 重置更深層的計數
                for k in list(counters.keys()):
                    if k > level:
                        counters[k] = 0

                # 建立結構碼
                if level == 0:
                    code = str(counters[0])
                else:
                    parent_parts = parent_code.split(".") if parent_code else []
                    code = ".".join(parent_parts[:level] + [str(counters[level])])

                result.append(
                    {
                        "title": bm.title,
                        "structure": code,
                        "physical_index": bm.page_num,
                        "level": level,
                    }
                )
                _process(bm.children, code)

        _process(bookmarks)
        return result

    def _detect_and_parse_toc(self, pdf_doc) -> list[dict]:
        """
        把前 N 頁直接送 LLM，讓 LLM 只負責抽取 (title, page_number) 配對。
        Structure code 由程式按出現順序分配（1, 2, 3...），避免 LLM 用章號當 code。
        """
        scan_pages = min(self.TOC_SCAN_PAGES, pdf_doc.total_pages)

        page_texts = []
        for i in range(1, scan_pages + 1):
            page = pdf_doc.get_page(i)
            if page and page.text.strip():
                page_texts.append(f"[第 {i} 頁]\n{page.text}")

        if not page_texts:
            return []

        combined = "\n\n".join(page_texts)

        prompt = f"""You are given the first few pages of a document ({pdf_doc.total_pages} pages total).

Task: Determine if there is a table of contents (TOC) that contains BOTH titles AND page numbers. If yes, extract ALL entries with their hierarchy level.

Reply format:
{{
  "has_toc": true or false,
  "entries": [
    {{"title": "Chapter 1 title", "page": 5, "level": 0}},
    {{"title": "Section 1.1 title", "page": 7, "level": 1}},
    {{"title": "Chapter 2 title", "page": 19, "level": 0}},
    ...
  ]
}}

Rules:
- "page" MUST be the actual page number printed next to the title in the TOC (e.g. "Chapter 1 ... 198")
- CRITICAL: Do NOT use section numbers, chapter numbers, or sequence numbers as page values
- If the TOC lists titles WITHOUT page numbers printed next to them, set has_toc to false
- "level" is the hierarchy depth: 0 for top-level chapters, 1 for sections, 2 for sub-sections
- Extract ALL entries in order — do NOT skip any
- Do NOT include the TOC heading itself (e.g. "Contents", "目录") as an entry
- For a {pdf_doc.total_pages}-page document, page numbers in the TOC should typically be larger than 20
- Only output JSON, nothing else.

Document pages:
{combined[:20000]}"""

        try:
            resp = retry_llm_call(
                self.llm,
                [{"role": "user", "content": prompt}],
                max_tokens=3000,
                label="TOC解析",
            )
            data = extract_json(resp)
            if not isinstance(data, dict) or not data.get("has_toc"):
                return []

            entries = data.get("entries", [])
            if not entries:
                return []

            # 程式按出現順序分配 structure code，根據 level 建立層級碼
            result = []
            toc_boundary = self.TOC_SCAN_PAGES
            # 各層計數器
            level_counters: dict[int, int] = {}

            for entry in entries:
                if not isinstance(entry, dict) or not entry.get("title"):
                    continue
                try:
                    page = int(entry.get("page", 0))
                except (TypeError, ValueError):
                    continue
                # 過濾掉頁碼在 TOC 區域內的項目（LLM 把目錄所在頁當成章節頁）
                if page <= toc_boundary:
                    continue

                level = int(entry.get("level", 0))
                level_counters[level] = level_counters.get(level, 0) + 1
                # 重置更深層計數
                for k in list(level_counters.keys()):
                    if k > level:
                        level_counters[k] = 0

                # 建立 structure code
                parts = [str(level_counters.get(i, 1)) for i in range(level + 1)]
                structure = ".".join(parts)

                result.append(
                    {
                        "title": str(entry["title"]).strip(),
                        "structure": structure,
                        "physical_index": page,
                        "level": level,
                    }
                )

            # 驗證單調遞增（頁碼應遞增）
            result = self._filter_monotonic(result)

            # 後驗：若所有頁碼都很小（< 10% 總頁數），
            # 很可能 LLM 把章節號當成頁碼回傳，視為無頁碼目錄
            if result:
                min_reasonable = max(20, pdf_doc.total_pages * 0.10)
                if max(r["physical_index"] for r in result) < min_reasonable:
                    return []

            return result
        except Exception:
            return []

    def _filter_monotonic(self, items: list[dict]) -> list[dict]:
        """過濾頁碼非單調遞增的項目（保留遞增的主幹）"""
        if not items:
            return items
        result = [items[0]]
        for item in items[1:]:
            if item["physical_index"] >= result[-1]["physical_index"]:
                result.append(item)
            # 否則跳過這個頁碼錯誤的項目
        return result

    def _detect_toc_titles(self, pdf_doc) -> list[dict]:
        """
        官方 detect_page_index() + toc_transformer()：
        偵測前 N 頁是否有目錄，若有則萃取章節結構（不要求頁碼）。
        回傳 [{"title": ..., "structure": ...}, ...] 或 []
        """
        scan_pages = min(self.TOC_SCAN_PAGES, pdf_doc.total_pages)
        page_texts = []
        for i in range(1, scan_pages + 1):
            page = pdf_doc.get_page(i)
            if page and page.text.strip():
                page_texts.append(f"[第 {i} 頁]\n{page.text}")
        if not page_texts:
            return []

        combined = "\n\n".join(page_texts)
        prompt = f"""You are given the first few pages of a document ({pdf_doc.total_pages} pages total).

Task:
1. Determine if there is a table of contents (TOC) in these pages.
2. If yes, extract the complete hierarchical chapter structure from the TOC.

The "structure" field is the numeric hierarchy index (e.g., "1", "1.1", "2", "2.1").
Page numbers are optional — leave them out if not present in the TOC.

Reply format:
{{
  "has_toc": true or false,
  "table_of_contents": [
    {{"structure": "1", "title": "Chapter title"}},
    {{"structure": "1.1", "title": "Sub-section title"}},
    ...
  ]
}}

If no TOC exists, return {{"has_toc": false, "table_of_contents": []}}.
Only output JSON, nothing else.

Document pages:
{combined[:6000]}"""

        try:
            resp = retry_llm_call(
                self.llm,
                [{"role": "user", "content": prompt}],
                max_tokens=2000,
                label="TOC結構偵測",
            )
            data = extract_json(resp)
            if not isinstance(data, dict) or not data.get("has_toc"):
                return []
            items = data.get("table_of_contents", [])
            result = []
            for item in items:
                if not isinstance(item, dict) or not item.get("title"):
                    continue
                result.append(
                    {
                        "title": str(item.get("title", "")).strip(),
                        "structure": str(item.get("structure", "")).strip(),
                    }
                )
            return result
        except Exception:
            return []

    def _process_toc_no_page_numbers(
        self, pdf_doc, toc_structure: list[dict]
    ) -> list[dict]:
        """
        官方 process_toc_no_page_numbers() + add_page_number_to_toc()：
        給定無頁碼的目錄結構，逐批掃描全文，讓 LLM 比對每個章節的實際起始頁碼。
        """
        total = pdf_doc.total_pages

        # 建立帶標籤的頁面批次（與 _llm_extract_structure 相同邏輯）
        tagged_pages = []
        for page in pdf_doc.pages:
            p = page.page_num
            tagged_pages.append(
                f"<physical_index_{p}>\n{page.text}\n<physical_index_{p}>\n\n"
            )

        groups: list[str] = []
        current_parts: list[str] = []
        current_len = 0
        for i, tagged in enumerate(tagged_pages):
            current_parts.append(tagged)
            current_len += len(tagged)
            if current_len >= self.LLM_GROUP_CHARS and i + 1 < len(tagged_pages):
                groups.append("".join(current_parts))
                current_parts = [tagged]
                current_len = len(tagged)
        if current_parts:
            groups.append("".join(current_parts))

        if not groups:
            return []

        # structure_code → physical_index（找到後不再覆蓋）
        found: dict[str, int] = {}
        pending = list(toc_structure)

        for group_idx, group in enumerate(groups):
            if not pending:
                break

            pending_json = json.dumps(pending, ensure_ascii=False)
            prompt = f"""You are given a list of chapter/section titles and a portion of a document.

For each title, find the page where the chapter CONTENT actually begins (i.e., the page with the chapter heading followed by body text).
Do NOT match against a table of contents listing — only match actual chapter start pages where the chapter's own text begins.

The document pages use tags like <physical_index_X> ... <physical_index_X> to mark page boundaries.

For each section:
- If its chapter content starts in these pages: set "start" to "yes" and "physical_index" to "<physical_index_X>"
- If it does not start here: set "start" to "no" and "physical_index" to null

Reply with a JSON array (same order as input):
[
  {{
    "structure": "x.x",
    "title": "...",
    "start": "yes" or "no",
    "physical_index": "<physical_index_X>" or null
  }},
  ...
]

Only output JSON, nothing else.

Sections to locate:
{pending_json}

Document pages:
{group}"""

            try:
                resp = retry_llm_call(
                    self.llm,
                    [{"role": "user", "content": prompt}],
                    max_tokens=2000,
                    label=f"TOC頁碼補充({group_idx + 1}/{len(groups)})",
                )
                items = extract_json_array(resp) or []
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    code = str(item.get("structure", ""))
                    if code in found:
                        continue
                    if item.get("start") == "yes":
                        pi = item.get("physical_index")
                        if isinstance(pi, str):
                            m = re.search(r"physical_index_(\d+)", pi)
                            if m:
                                found[code] = int(m.group(1))
                pending = [p for p in pending if p.get("structure") not in found]
            except Exception:
                continue

        # 組合結果：只回傳找到頁碼的章節
        result = []
        for item in toc_structure:
            code = item.get("structure", "")
            if code in found:
                result.append(
                    {
                        "title": item["title"],
                        "structure": code,
                        "physical_index": found[code],
                        "level": code.count("."),
                    }
                )
        return result

    def _llm_extract_structure(self, pdf_doc, doc_id: str) -> list[dict]:
        """
        對齊 PageIndex process_no_toc()：
        全文掃描，每頁加 <physical_index_X> 標籤，
        分批送 LLM，後批帶前一批結構繼續延伸（generate_toc_continue）。

        與舊版差異：
          舊版只抽幾個固定位置的頁面（盲點多）
          新版掃全文、分批、帶上下文串接 → 不會遺漏中段章節
        """
        total = pdf_doc.total_pages

        # 每頁包上 <physical_index_X> 標籤（與 PageIndex 格式一致）
        tagged_pages = []
        for page in pdf_doc.pages:
            p = page.page_num
            tagged_pages.append(
                f"<physical_index_{p}>\n{page.text}\n<physical_index_{p}>\n\n"
            )

        # 切成 LLM_GROUP_CHARS 大小的批次（保留最後一頁作重疊，維持上下文連續）
        groups: list[str] = []
        current_parts: list[str] = []
        current_len = 0
        for i, tagged in enumerate(tagged_pages):
            current_parts.append(tagged)
            current_len += len(tagged)
            if current_len >= self.LLM_GROUP_CHARS and i + 1 < len(tagged_pages):
                groups.append("".join(current_parts))
                current_parts = [tagged]  # 最後一頁作重疊
                current_len = len(tagged)
        if current_parts:
            groups.append("".join(current_parts))

        if not groups:
            return [{"title": "全文", "structure": "1", "physical_index": 1}]

        # 第一批：generate_toc_init
        structure = self._toc_generate_init(groups[0], total)

        # 後續批次：generate_toc_continue（帶前一批結構作為上下文）
        for group in groups[1:]:
            additional = self._toc_generate_continue(structure, group, total)
            if additional:
                structure.extend(additional)

        if not structure:
            return [{"title": "全文", "structure": "1", "physical_index": 1}]

        return self._parse_structure_response(json.dumps(structure), total)

    def _toc_generate_init(self, tagged_text: str, total_pages: int) -> list[dict]:
        """第一批頁面：生成初始結構（對齊 PageIndex generate_toc_init）"""
        # 對齊官方 generate_toc_init：
        # - physical_index 保留 <physical_index_X> 格式（LLM 直接複製標籤，更可靠）
        # - 不加「若無章節則強制生成邏輯區塊」指令（官方不做此事）
        # - tagged_text 已按 LLM_GROUP_CHARS 切批，不二次截斷
        prompt = f"""You are an expert in extracting hierarchical tree structure, your task is to generate the tree structure of the document.

The structure variable is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

For the title, you need to extract the original title from the text, only fix the space inconsistency.

The provided text contains tags like <physical_index_X> and <physical_index_X> to indicate the start and end of page X.

For the physical_index, you need to extract the physical index of the start of the section from the text. Keep the <physical_index_X> format.

The response should be in the following format.
    [
        {{
            "structure": <structure index, "x.x.x"> (string),
            "title": <title of the section, keep the original title>,
            "physical_index": "<physical_index_X> (keep the format)"
        }},
    ],

Directly return the final JSON structure. Do not output anything else.
Given text:
{tagged_text}"""

        try:
            resp = retry_llm_call(
                self.llm,
                [{"role": "user", "content": prompt}],
                max_tokens=2000,
                label="結構初始化",
            )
            items = extract_json_array(resp) or []
            items = [i for i in items if isinstance(i, dict) and "title" in i]
            # 對齊官方 convert_physical_index_to_int：將 "<physical_index_X>" 轉為整數
            for item in items:
                pi = item.get("physical_index")
                if isinstance(pi, str):
                    m = re.search(r"physical_index_(\d+)", pi)
                    item["physical_index"] = int(m.group(1)) if m else None
            # 官方 meta_processor 在生成後立即過濾 None：
            # toc_with_page_number = [item for item in ... if item.get('physical_index') is not None]
            return [i for i in items if i.get("physical_index") is not None]
        except Exception:
            return []

    def _toc_generate_continue(
        self, prev_structure: list[dict], tagged_text: str, total_pages: int
    ) -> list[dict]:
        """後續批次：帶著前批結構繼續延伸（對齊 PageIndex generate_toc_continue）"""
        # 對齊官方 generate_toc_continue：
        # - 送完整前置結構（官方 json.dumps(toc_content) 不截斷）
        # - 順序：先 Given text，後 Previous tree structure（官方相同順序）
        # - physical_index 保留 <physical_index_X> 格式
        prev_json = json.dumps(prev_structure, ensure_ascii=False, indent=2)

        prompt = f"""You are an expert in extracting hierarchical tree structure.
You are given a tree structure of the previous part and the text of the current part.
Your task is to continue the tree structure from the previous part to include the current part.

The structure variable is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

For the title, you need to extract the original title from the text, only fix the space inconsistency.

The provided text contains tags like <physical_index_X> and <physical_index_X> to indicate the start and end of page X.

For the physical_index, you need to extract the physical index of the start of the section from the text. Keep the <physical_index_X> format.

The response should be in the following format.
    [
        {{
            "structure": <structure index, "x.x.x"> (string),
            "title": <title of the section, keep the original title>,
            "physical_index": "<physical_index_X> (keep the format)"
        }},
        ...
    ]

Directly return the additional part of the final JSON structure. Do not output anything else.
Given text:
{tagged_text}
Previous tree structure:
{prev_json}"""

        try:
            resp = retry_llm_call(
                self.llm,
                [{"role": "user", "content": prompt}],
                max_tokens=2000,
                label="結構延伸",
            )
            items = extract_json_array(resp) or []
            items = [i for i in items if isinstance(i, dict) and "title" in i]
            # 對齊官方 convert_physical_index_to_int
            for item in items:
                pi = item.get("physical_index")
                if isinstance(pi, str):
                    m = re.search(r"physical_index_(\d+)", pi)
                    item["physical_index"] = int(m.group(1)) if m else None
            # 過濾 None（官方 meta_processor 生成後立即過濾）
            return [i for i in items if i.get("physical_index") is not None]
        except Exception:
            return []

    def _parse_structure_response(self, response: str, total_pages: int) -> list[dict]:
        """解析 LLM 返回的結構 JSON（使用官方 extract_json 4 階段解析）"""
        items = extract_json_array(response)
        if not items:
            # fallback：嘗試解析物件包陣列的格式
            data = extract_json(response)
            items = data if isinstance(data, list) else []
        if not items:
            return [{"title": "全文", "structure": "1", "physical_index": 1}]
        try:
            # 驗證和清理
            result = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title", "未命名")).strip()
                structure = str(item.get("structure", "1")).strip()
                # physical_index=None（convert_physical_index_to_int 失敗時）→ 跳過
                page_raw = item.get("physical_index")
                if page_raw is None:
                    continue
                try:
                    page = int(page_raw)
                except (TypeError, ValueError):
                    continue
                page = max(1, min(page, total_pages))
                result.append(
                    {
                        "title": title,
                        "structure": structure,
                        "physical_index": page,
                        "level": structure.count("."),
                    }
                )
            return result or [{"title": "全文", "structure": "1", "physical_index": 1}]
        except Exception:
            return [{"title": "全文", "structure": "1", "physical_index": 1}]
