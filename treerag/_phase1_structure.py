"""
_phase1_structure.py
─────────────────────────────────────────────────────
Phase 1: Structure extraction (bookmarks, TOC, patterns, LLM sampling)
"""

import json
import re
from collections import Counter
from utils import extract_json, extract_json_array, retry_llm_call


class _Phase1StructureMixin:
    """Phase 1: 從書籤 or LLM 取得結構碼列表"""

    def _get_structure_list(self, pdf_doc, doc_id: str) -> list[dict]:
        """
        優先順序：
          1. PDF 書籤（最精確）
          2. 偵測目錄頁（TOC）並解析（適合大型文件）
          3. 掃描全文標題行模式（不需 LLM，覆蓋無書籤/無TOC的文件）
          4. LLM 分析採樣頁面（fallback）
        """
        # 優先：PDF 書籤
        if pdf_doc.bookmarks:
            return self._bookmarks_to_structure(pdf_doc.bookmarks)

        # 次優：偵測並解析 TOC 頁面
        toc_result = self._detect_and_parse_toc(pdf_doc)
        if toc_result and len(toc_result) >= 3:
            print(f"   → 從目錄頁解析到 {len(toc_result)} 個章節")
            return toc_result

        # 新增：掃描全文所有頁面的標題行模式（不需 LLM）
        pattern_result = self._scan_heading_patterns(pdf_doc)
        if pattern_result and len(pattern_result) >= 3:
            print(f"   → 從標題行模式掃描到 {len(pattern_result)} 個章節")
            pdf_doc.heading_source = "pattern"
            return pattern_result

        # Fallback：LLM 分析採樣頁面
        return self._llm_extract_structure(pdf_doc, doc_id)

    def _bookmarks_to_structure(self, bookmarks) -> list[dict]:
        """將 PDF 書籤轉成結構碼列表"""
        result = []
        counters = {}  # level → 累計數

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

                result.append({
                    "title": bm.title,
                    "structure": code,
                    "physical_index": bm.page_num,
                    "level": level,
                })
                _process(bm.children, code)

        _process(bookmarks)
        return result

    def _detect_and_parse_toc(self, pdf_doc) -> list[dict]:
        """
        Step 1：掃描前幾頁，偵測是否有目錄（Table of Contents）
        Step 2：若找到，用 LLM 解析目錄 → 完整章節列表

        這是處理大型文件（>100頁）的關鍵：
        目錄頁通常列出所有章節和頁碼，比採樣正文更準確
        """
        scan_pages = min(self.TOC_SCAN_PAGES, pdf_doc.total_pages)
        toc_page_texts = []

        # 偵測目錄頁關鍵字（包含有空格的中文變體）
        toc_keywords = [
            "table of contents", "contents", "目錄", "目 錄",
            "content", "index", "目次", "索引",
        ]
        toc_pages_found = []

        for i in range(1, scan_pages + 1):
            page = pdf_doc.get_page(i)
            if not page or not page.text.strip():
                continue  # 跳過空頁（掃描圖片頁）

            # 去除空格後比對關鍵字（處理「目 錄」→「目錄」）
            text_lower = page.text.lower()
            text_nospace = re.sub(r'\s+', '', text_lower)
            has_keyword = (
                any(kw in text_lower for kw in toc_keywords) or
                any(re.sub(r'\s+', '', kw) in text_nospace for kw in toc_keywords)
            )

            # 目錄頁特徵：多行且含有頁碼（行末有 2+ 位數字，排除單頁碼如 "- 1 -"）
            lines = [l.strip() for l in page.text.split("\n") if l.strip()]
            lines_with_num = sum(
                1 for l in lines
                # 行末有 ≥2 位數字，或有「....頁碼」型式，且行本身有實質內容
                if re.search(r'[^\d]\d{1,3}\s*$', l) and len(l) > 8
                and not re.fullmatch(r'[-─~～\s\d]+', l)  # 排除純頁碼行
            )
            has_page_num_pattern = len(lines) > 5 and lines_with_num / len(lines) > 0.35

            if has_keyword or has_page_num_pattern:
                toc_pages_found.append(i)
                toc_page_texts.append(f"[第 {i} 頁]\n{page.text}")

        if not toc_pages_found:
            return []

        # 合併目錄頁文字
        toc_text = "\n\n".join(toc_page_texts)

        prompt = f"""這是一份文件的目錄頁面（共 {pdf_doc.total_pages} 頁）。
請從中萃取完整的章節結構，包含所有章節的名稱和頁碼。

目錄內容：
{toc_text[:5000]}

請輸出 JSON 陣列，列出所有章節：
[
  {{"title": "章節名稱", "structure": "1", "physical_index": 頁碼}},
  {{"title": "子節名稱", "structure": "1.1", "physical_index": 頁碼}}
]

規則：
- structure 用數字層級（1, 1.1, 1.1.1, 2, 2.1 ...）
- physical_index 就是目錄中標示的頁碼數字
- 保留所有層級的章節，不要省略
- 若目錄中無層級，統一用 1, 2, 3, 4 ...
- 只輸出 JSON，不要其他文字"""

        try:
            resp = retry_llm_call(self.llm, [{"role": "user", "content": prompt}], max_tokens=2000, label="TOC解析")
            result = self._parse_structure_response(resp, pdf_doc.total_pages)
            if result:
                # 偵測邏輯頁碼 vs 物理頁碼偏移（財報封面/版權頁造成的偏移）
                offset = self._detect_page_offset(result, pdf_doc)
                if offset != 0:
                    print(f"   → 偵測到頁碼偏移量：{offset:+d}，自動校正")
                    for item in result:
                        item["physical_index"] = max(1, min(
                            item["physical_index"] + offset,
                            pdf_doc.total_pages,
                        ))
            return result
        except Exception:
            return []

    def _detect_page_offset(self, structure_list: list[dict], pdf_doc) -> int:
        """
        偵測 TOC 邏輯頁碼與 PDF 物理頁碼之間的偏移量。

        財報、教科書等文件常見情況：
          TOC 寫「合併資產負債表 ... 1」（邏輯第 1 頁）
          但 PDF 物理第 1 頁是封面，實際內容在物理第 5 頁
          → 偏移量 = +4

        策略：對前幾個章節，在 ±20 頁範圍內搜尋標題實際出現的頁碼，
        計算最常見的偏移量（中位數）並回傳。
        不到 2 個章節找到對應頁碼時回傳 0（不調整）。
        """
        total = pdf_doc.total_pages
        offsets = []

        for item in structure_list[:6]:
            title = item.get("title", "").strip()
            logical_page = item.get("physical_index", 1)
            if not title or len(title) < 2:
                continue

            # 去除所有空白後比對（應對 "合 併 資 產 負 債 表" 這類間距字）
            norm_title = re.sub(r'\s+', '', title).lower()

            # 在邏輯頁 ±20 範圍內搜尋
            search_start = max(1, logical_page - 5)
            search_end = min(total, logical_page + 20)
            for phys_page in range(search_start, search_end + 1):
                page = pdf_doc.get_page(phys_page)
                if not page:
                    continue
                norm_text = re.sub(r'\s+', '', page.text).lower()
                if norm_title in norm_text:
                    offsets.append(phys_page - logical_page)
                    break

        if len(offsets) < 2:
            return 0  # 樣本不足，不調整

        # 回傳最常見的偏移值
        from collections import Counter
        return Counter(offsets).most_common(1)[0][0]

    def _scan_heading_patterns(self, pdf_doc) -> list[dict]:
        """
        掃描全文所有頁面，以正規表達式偵測標題行。
        不需要 LLM，速度快，適合書籤/TOC 均缺失的文件。

        偵測模式（按優先序）：
          - 中文章節：第N章 / 第N節 / 第N篇 / 第N部分
          - 英文章節：Chapter N / Section N / Part N / Appendix N
          - 數字編號：1. Title / 1.1 Title / 1.1.1 Title（後接文字）
          - 羅馬數字：I. Introduction / II. Methods ...

        過濾頁首頁尾（出現頁數 > 30% 的行）
        若找到 ≥ 3 個獨特標題，回傳結構碼列表；否則回傳 []
        """
        total = pdf_doc.total_pages
        if total == 0:
            return []

        # 四種標題模式（依優先序）
        PATTERNS = [
            # 0: 中文大章節
            re.compile(
                r'^第[一二三四五六七八九十百千零\d]+[章節篇部分][\s\u3000\uff1a：:]*(.{0,50})$'
            ),
            # 1: 英文章節（不分大小寫）
            re.compile(
                r'^(?:Chapter|Section|Part|Appendix|Article)\s+[\dIVXivx]+[\s\.\:：]*(.{0,60})$',
                re.IGNORECASE,
            ),
            # 2: 數字編號  1. / 1.1 / 1.1.1  後接非數字文字
            re.compile(
                r'^(\d{1,2}(?:\.\d{1,2}){0,2})[.\s、]\s*([^\d\s].{1,70})$'
            ),
            # 3: 羅馬數字章節  I. / II. / III. ...
            re.compile(
                r'^(I{1,3}|IV|VI{0,3}|IX|X{1,3})\.\s+[A-Z\u4e00-\u9fff].{1,60}$'
            ),
        ]

        # 收集：(page_num, line, level)
        line_first_page: dict[str, int] = {}
        line_page_count: Counter = Counter()
        line_level: dict[str, int] = {}

        for page in pdf_doc.pages:
            seen_this_page: set[str] = set()
            for raw_line in page.text.split("\n"):
                line = raw_line.strip()
                if not line or len(line) > 120:
                    continue

                for pat_idx, pattern in enumerate(PATTERNS):
                    m = pattern.match(line)
                    if not m:
                        continue

                    # 判斷層級
                    if pat_idx in (0, 1, 3):
                        level = 0
                    else:  # pat_idx == 2: 數字編號
                        num_part = m.group(1)
                        level = num_part.count(".")

                    if line not in seen_this_page:
                        seen_this_page.add(line)
                        line_page_count[line] += 1
                        if line not in line_first_page:
                            line_first_page[line] = page.page_num
                            line_level[line] = level
                    break  # 每行只匹配第一個符合的模式

        if not line_first_page:
            return []

        # 過濾頁首頁尾（出現頁數 > 30% 的行）
        threshold = total * 0.3
        headings = [
            (line_first_page[ln], ln, line_level[ln])
            for ln in line_first_page
            if line_page_count[ln] <= threshold
        ]
        headings.sort(key=lambda x: x[0])

        if len(headings) < 3:
            return []

        # 指派結構碼（純啟發式，不需 LLM）
        counters: dict[int, int] = {}
        result = []

        for page_num, title, level in headings:
            # 重置比當前更深的層級計數
            for k in list(counters.keys()):
                if k > level:
                    counters[k] = 0
            counters[level] = counters.get(level, 0) + 1

            code_parts = [str(max(1, counters.get(i, 0))) for i in range(level + 1)]
            code = ".".join(code_parts)

            result.append({
                "title": title,
                "structure": code,
                "physical_index": page_num,
                "level": level,
            })

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
                current_parts = [tagged]   # 最後一頁作重疊
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
                result.append({
                    "title": title,
                    "structure": structure,
                    "physical_index": page,
                    "level": structure.count("."),
                })
            return result or [{"title": "全文", "structure": "1", "physical_index": 1}]
        except Exception:
            return [{"title": "全文", "structure": "1", "physical_index": 1}]
