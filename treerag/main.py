"""
main.py ── TreeRAG 檢索系統入口（PageIndex 風格）

核心設計：
  ✅ 專注文件檢索，非對話助手
  ✅ 簡化 CLI 介面，方便驗證
  ✅ 保留 trace 輸出，可觀察檢索過程

使用方式：
  # 建立索引 + 互動模式
  python main.py --pdf file.pdf

  # 單次查詢（方便自動化）
  python main.py --pdf file.pdf --query "Q1 營收是多少？"

  # 多文件查詢
  python main.py --pdf1 TSMC.pdf --pdf2 form10-k.pdf --query "比較兩公司營收"

  # 強制重建索引
  python main.py --pdf file.pdf --reindex

環境變數：
  export OPENAI_API_KEY=sk-...
  export ANTHROPIC_API_KEY=sk-ant-...
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from pdf_parser import load_pdf
from tree_index import TreeIndexBuilder
from filesystem_index import FileSystemIndex
from retriever import SimpleRetriever
from llm_client import create_llm


# ════════════════════════════════════════════════════════
# 知識庫建立
# ════════════════════════════════════════════════════════


def setup_knowledge_base(
    pdf_paths: list[str],
    index_dir: str,
    llm,
    force_reindex: bool = False,
) -> tuple[FileSystemIndex, dict]:
    """
    建立或載入知識庫
    Returns: (FileSystemIndex, {doc_id: PDFDocument})
    """
    fs_index = FileSystemIndex(index_dir)
    pdf_docs = {}
    builder = TreeIndexBuilder(llm)

    for pdf_path in pdf_paths:
        if not Path(pdf_path).exists():
            print(f"⚠️  找不到：{pdf_path}")
            continue

        stem = Path(pdf_path).stem
        safe = "".join(
            c if c.isascii() and (c.isalnum() or c in "_-") else "_"
            for c in stem.replace(" ", "_").lower()
        ).strip("_-")
        # 若 ASCII 部分太短（大量中文被過濾），補上檔名的 MD5 前綴
        import hashlib

        if len(safe) < max(4, len(stem) * 0.3):
            hash6 = hashlib.md5(stem.encode()).hexdigest()[:6]
            doc_id = f"{hash6}_{safe}".strip("_-") if safe else hash6
        else:
            doc_id = safe or "doc"

        index_path = Path(index_dir) / f"{doc_id}.index.json"

        print(f"\n📖 載入：{Path(pdf_path).name}")
        pdf_doc = load_pdf(pdf_path)
        pdf_docs[doc_id] = pdf_doc
        print(f"   頁數：{pdf_doc.total_pages}  書籤來源：{pdf_doc.heading_source}")

        if index_path.exists() and not force_reindex:
            print("   ✅ 索引已存在，直接載入")
            if doc_id not in fs_index.entries:
                from tree_index import DocumentIndex

                doc_index = DocumentIndex.load(str(index_path))
                fs_index.register(doc_index)
            continue

        doc_index = builder.build(pdf_doc, doc_id=doc_id)
        doc_index.save(index_dir)
        fs_index.register(doc_index)
        print("   💾 索引儲存完成")

    return fs_index, pdf_docs


# ════════════════════════════════════════════════════════
# 互動模式（簡化版）
# ════════════════════════════════════════════════════════


def interactive_mode(retriever: SimpleRetriever, fs_index: FileSystemIndex):
    """簡化互動模式 - 專注檢索驗證"""
    print("\n" + "=" * 60)
    print("🤖 TreeRAG 檢索模式（PageIndex 風格）")
    print("=" * 60)
    print("指令：  [問題] → 查詢  |  show → 知識庫結構  |  quit → 離開")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n❓ 查詢：").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 再見！")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("👋 再見！")
            break
        if user_input.lower() == "show":
            print("\n" + fs_index.render_full_view())
            continue

        print("\n⏳ 檢索中...")
        try:
            result = retriever.query(user_input, verbose=True)
            print("\n" + result.render_trace())
        except Exception as e:
            print(f"\n❌ 錯誤：{e}")
            import traceback

            traceback.print_exc()


# ════════════════════════════════════════════════════════
# CLI 入口
# ════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="TreeRAG：PageIndex 風格的簡化文件檢索系統",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
  python main.py --pdf TSMC.pdf
  python main.py --pdf1 TSMC.pdf --pdf2 form10-k.pdf
  python main.py --pdf TSMC.pdf --query "Q1 營收是多少？"
  python main.py --pdf TSMC.pdf --reindex
        """,
    )
    parser.add_argument("--pdf", help="單一 PDF 檔案")
    parser.add_argument("--pdf1", help="第一個 PDF 檔案")
    parser.add_argument("--pdf2", help="第二個 PDF 檔案")
    parser.add_argument("--pdfs", nargs="+", help="多個 PDF 檔案")
    parser.add_argument("--provider", choices=["openai", "anthropic", "ollama"])
    parser.add_argument("--model")
    parser.add_argument("--api-key")
    parser.add_argument("--index-dir", default=".treerag_index")
    parser.add_argument("--reindex", action="store_true", help="強制重建索引")
    parser.add_argument("--query", help="查詢問題（若未指定則進入互動模式）")

    args = parser.parse_args()

    # Resolve index_dir to absolute path
    args.index_dir = str(Path(args.index_dir).resolve())

    # 收集 PDF 路徑
    pdf_paths = []
    if args.pdf:
        pdf_paths.append(args.pdf)
    if args.pdf1:
        pdf_paths.append(args.pdf1)
    if args.pdf2:
        pdf_paths.append(args.pdf2)
    if args.pdfs:
        pdf_paths.extend(args.pdfs)

    if not pdf_paths:
        print("❌ 請指定至少一個 PDF 檔案（--pdf 或 --pdf1）")
        sys.exit(1)

    # 初始化 LLM
    print("🔧 初始化 LLM...")
    llm_kwargs = {}
    if args.model:
        llm_kwargs["model"] = args.model
    elif args.provider == "openai":
        # 預設用 gpt-4o-mini
        llm_kwargs["model"] = "gpt-5.4-mini"
    if args.api_key:
        llm_kwargs["api_key"] = args.api_key
    try:
        llm = create_llm(args.provider, **llm_kwargs)
        print(f"   使用：{llm.provider} / {llm.model}")
    except Exception as e:
        print(f"❌ LLM 初始化失敗：{e}")
        print(
            "   請設定：export OPENAI_API_KEY=sk-... 或 export ANTHROPIC_API_KEY=sk-ant-..."
        )
        sys.exit(1)

    # 建立知識庫
    print("\n📚 建立知識庫...")
    fs_index, pdf_docs = setup_knowledge_base(
        pdf_paths=pdf_paths,
        index_dir=args.index_dir,
        llm=llm,
        force_reindex=args.reindex,
    )

    if not pdf_docs:
        print("❌ 沒有成功載入任何 PDF")
        sys.exit(1)

    print(f"\n✅ 知識庫就緒：{len(pdf_docs)} 份文件")
    print(fs_index.get_L0_context())

    # 建立檢索器
    retriever = SimpleRetriever(
        llm_client=llm,
        fs_index=fs_index,
        pdf_docs=pdf_docs,
    )

    # 執行
    if args.query:
        print(f"\n⏳ 查詢：{args.query}")
        result = retriever.query(args.query, verbose=True)
        print("\n" + result.render_trace())
    else:
        interactive_mode(retriever, fs_index)


if __name__ == "__main__":
    import argparse

    main()
