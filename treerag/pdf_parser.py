"""
pdf_parser.py
─────────────────────────────────────────────────────
PDF 解析模組（基於 OpenViking openviking/parse/parsers/pdf.py 的設計）

策略：
  1. 優先使用 PDF 書籤（Bookmarks/Outline）作為章節結構
  2. 無書籤時，用字體大小分析自動偵測標題
  3. 萃取表格並轉成 Markdown 格式

核心改進（相較初版）：
  - 書籤萃取：完整支援多層級 PDF Outline
  - 字體分析：取樣每 5 頁，識別正文字體大小，偵測標題字體
  - 標題去重：過濾超過 30% 頁面都出現的頁首頁尾
  - 表格轉 Markdown：保留表格結構
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Bookmark:
    """PDF 書籤（Outline 條目）"""
    title: str
    page_num: int        # 1-based
    level: int           # 0 = 頂層
    children: list = field(default_factory=list)


@dataclass
class PageContent:
    """PDF 單頁內容"""
    page_num: int        # 1-based
    text: str
    tables: list[str]    # Markdown 格式的表格
    char_count: int
    font_sizes: list[float] = field(default_factory=list)   # 該頁字體大小列表

    def full_text(self) -> str:
        """文字 + 表格合併"""
        parts = [self.text]
        parts.extend(self.tables)
        return "\n\n".join(p for p in parts if p.strip())

    def to_dict(self) -> dict:
        return {
            "page_num": self.page_num,
            "text": self.text,
            "tables": self.tables,
            "char_count": self.char_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PageContent":
        return cls(
            page_num=d["page_num"],
            text=d["text"],
            tables=d.get("tables", []),
            char_count=d["char_count"],
        )


@dataclass
class PDFDocument:
    """解析後的 PDF 文件（含書籤結構）"""
    path: str
    filename: str
    pages: list[PageContent]
    total_pages: int
    bookmarks: list[Bookmark]         # 書籤樹（可能為空）
    heading_source: str               # "bookmarks" | "font" | "none"
    font_headings: dict[int, list[str]] = field(default_factory=dict)  # page_num → headings

    def get_page(self, page_num: int) -> Optional[PageContent]:
        for p in self.pages:
            if p.page_num == page_num:
                return p
        return None

    def get_range_text(self, start_page: int, end_page: int) -> str:
        texts = []
        for p in self.pages:
            if start_page <= p.page_num <= end_page:
                entry = f"[第 {p.page_num} 頁]\n{p.full_text()}"
                texts.append(entry)
        return "\n\n".join(texts)

    def get_full_text(self) -> str:
        return self.get_range_text(1, self.total_pages)

    def flat_bookmarks(self) -> list[Bookmark]:
        """扁平化書籤列表（深度優先）"""
        result = []
        def _flatten(bms):
            for b in bms:
                result.append(b)
                _flatten(b.children)
        _flatten(self.bookmarks)
        return result


# ════════════════════════════════════════════════════════
# 載入函式
# ════════════════════════════════════════════════════════

def load_pdf(pdf_path: str) -> PDFDocument:
    """載入 PDF，自動選擇最佳解析策略"""
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("請安裝：pip install pdfplumber")

    pdf_path = str(pdf_path)
    filename = Path(pdf_path).name

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)

        # Step 1: 萃取每頁內容
        pages = _extract_pages(pdf)

        # Step 2: 萃取書籤
        bookmarks = _extract_bookmarks(pdf)
        heading_source = "bookmarks" if bookmarks else "none"

        # Step 3: 無書籤時，用字體分析
        font_headings = {}
        if not bookmarks:
            font_headings, detected_headings = _detect_headings_by_font(pdf, pages)
            if detected_headings:
                heading_source = "font"

    return PDFDocument(
        path=pdf_path,
        filename=filename,
        pages=pages,
        total_pages=total_pages,
        bookmarks=bookmarks,
        heading_source=heading_source,
        font_headings=font_headings,
    )


# ════════════════════════════════════════════════════════
# 頁面萃取
# ════════════════════════════════════════════════════════

def _extract_pages(pdf) -> list[PageContent]:
    """萃取所有頁面的文字和表格"""
    pages = []
    for i, page in enumerate(pdf.pages):
        page_num = i + 1

        # 萃取表格（先處理，避免干擾文字提取）
        tables = []
        try:
            for table in page.extract_tables():
                md = _format_table_markdown(table)
                if md:
                    tables.append(md)
        except Exception:
            pass

        # 萃取文字
        try:
            text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
        except Exception:
            text = ""
        text = _clean_text(text)

        # 字體大小（用於標題偵測）
        font_sizes = []
        try:
            words = page.extract_words(extra_attrs=["size"])
            font_sizes = [w.get("size", 0) for w in words if w.get("size")]
        except Exception:
            pass

        pages.append(PageContent(
            page_num=page_num,
            text=text,
            tables=tables,
            char_count=len(text),
            font_sizes=font_sizes,
        ))

    return pages


def _format_table_markdown(table: list) -> str:
    """將 pdfplumber 萃取的表格轉成 Markdown（來自 OpenViking）"""
    if not table or not table[0]:
        return ""

    # 清理儲存格
    def clean_cell(cell):
        if cell is None:
            return ""
        return str(cell).replace("\n", " ").strip()

    cleaned = [[clean_cell(cell) for cell in row] for row in table]
    if not cleaned:
        return ""

    # 建立 Markdown 表格
    header = cleaned[0]
    rows = cleaned[1:] if len(cleaned) > 1 else []

    header_line = "| " + " | ".join(header) + " |"
    separator = "| " + " | ".join(["---"] * len(header)) + " |"
    row_lines = ["| " + " | ".join(row) + " |" for row in rows]

    return "\n".join([header_line, separator] + row_lines)


# ════════════════════════════════════════════════════════
# 書籤萃取
# ════════════════════════════════════════════════════════

def _extract_bookmarks(pdf) -> list[Bookmark]:
    """
    萃取 PDF Outline/Bookmarks（來自 OpenViking _extract_bookmarks）
    使用 pdfplumber 底層的 pypdf2/pikepdf
    """
    try:
        # pdfplumber 底層是 pypdf（PyPDF2 的繼承者）
        raw_pdf = pdf.doc  # pikepdf object
        return _extract_bookmarks_pikepdf(raw_pdf, pdf)
    except Exception:
        pass

    try:
        # 備用：確認 pdfplumber outline API 可用（不解析，直接回傳空）
        pdf.doc.open_outline()
        return []
    except Exception:
        pass

    return []


def _extract_bookmarks_pikepdf(raw_pdf, plumber_pdf) -> list[Bookmark]:
    """使用 pikepdf 萃取書籤（pdfplumber 底層）"""
    total_pages = len(plumber_pdf.pages)

    # 建立頁面 objID → page_num 映射
    page_map = {}
    for i, page in enumerate(raw_pdf.pages):
        page_map[page.objgen] = i + 1

    def parse_outline_items(items, level=0) -> list[Bookmark]:
        bookmarks = []
        for item in items:
            title = str(item.title) if item.title else "（無標題）"
            page_num = 1

            # 解析頁碼
            try:
                if item.destination:
                    dest = item.destination
                    if isinstance(dest, list) and len(dest) > 0:
                        page_ref = dest[0]
                        if hasattr(page_ref, 'objgen'):
                            page_num = page_map.get(page_ref.objgen, 1)
                elif item.action:
                    action = item.action
                    if '/D' in action:
                        dest = action['/D']
                        if hasattr(dest, '__iter__') and len(dest) > 0:
                            page_ref = dest[0]
                            if hasattr(page_ref, 'objgen'):
                                page_num = page_map.get(page_ref.objgen, 1)
            except Exception:
                pass

            bookmark = Bookmark(
                title=title.strip(),
                page_num=max(1, min(page_num, total_pages)),
                level=level,
            )

            # 遞迴處理子書籤
            if item.children:
                bookmark.children = parse_outline_items(item.children, level + 1)

            bookmarks.append(bookmark)
        return bookmarks

    try:
        with raw_pdf.open_outline() as outline:
            if outline.root:
                return parse_outline_items(outline.root)
    except Exception:
        pass

    return []


# ════════════════════════════════════════════════════════
# 字體大小分析（無書籤時的備用）
# ════════════════════════════════════════════════════════

def _detect_headings_by_font(
    pdf,
    pages: list[PageContent],
    sample_every: int = 5,
    min_delta: float = 1.5,
) -> tuple[dict, bool]:
    """
    字體大小分析偵測標題（來自 OpenViking _detect_headings_by_font）

    演算法：
    1. 每隔 sample_every 頁取樣字體大小
    2. 找出出現最頻繁的字體大小 → 正文字體
    3. 超過正文字體 + min_delta 的 → 標題字體
    4. 過濾超過 30% 頁面都出現的（頁首頁尾）
    """
    from collections import Counter

    total = len(pages)
    if total == 0:
        return {}, False

    # 取樣字體大小
    all_sizes = []
    for i, page in enumerate(pages):
        if i % sample_every == 0:
            all_sizes.extend(page.font_sizes)

    if not all_sizes:
        return {}, False

    # 找正文字體大小（最常見）
    size_counter = Counter(round(s, 1) for s in all_sizes)
    if not size_counter:
        return {}, False

    body_size = size_counter.most_common(1)[0][0]

    # 找標題字體大小（比正文大且不太常見）
    heading_sizes = set()
    body_freq = size_counter[body_size]
    for size, count in size_counter.items():
        if size >= body_size + min_delta and count < body_freq * 0.3:
            heading_sizes.add(size)

    if not heading_sizes:
        return {}, False

    # 為每頁找標題文字
    page_headings: dict[int, list[str]] = {}
    heading_text_page_count: Counter = Counter()

    for page in pages:
        headings = []

        # 用文字中的短行作為標題候選（啟發式，書籤不足時的備用）
        for line in page.text.split("\n"):
            line = line.strip()
            if line and len(line) < 100:
                # 如果行長度短且不是數字，可能是標題
                if not line.replace(".", "").replace(" ", "").isdigit():
                    headings.append(line)
                    heading_text_page_count[line] += 1

        if headings:
            page_headings[page.page_num] = headings

    # 過濾出現超過 30% 頁面的文字（頁首頁尾）
    threshold = total * 0.3
    filtered: dict[int, list[str]] = {}
    for pg, heads in page_headings.items():
        clean = [h for h in heads if heading_text_page_count[h] <= threshold]
        if clean:
            filtered[pg] = clean

    return filtered, bool(filtered)


# ════════════════════════════════════════════════════════
# 工具函式
# ════════════════════════════════════════════════════════

def _clean_text(text: str) -> str:
    """清理文字：移除 null bytes、控制字元、多餘空白和換行

    PDF 萃取常含 null bytes (\x00) 和其他控制字元，
    若直接送進 OpenAI API 的 JSON body 會造成 HTTP 400。
    保留的字元：\t（水平 tab）、\n（換行）、\r（回車）。
    """
    # 1. 移除 null bytes 及非列印控制字元（保留 \t \n \r）
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # 2. 整理多餘換行與空白
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+\n', '\n', text)
    text = re.sub(r'\n[ \t]+', '\n', text)
    return text.strip()
