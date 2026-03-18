# TreeRAG：整合 OpenViking + PageIndex 的無向量 PDF 問答系統

## 架構概念

```
【OpenViking 概念 → 跨文件層】
  檔案系統索引（FileSystemIndex）
    L0：知識庫整體摘要
    L1：所有文件的章節樹列表
    L2：指定章節的原文內容

【PageIndex 概念 → 文件內層】
  文件樹狀索引（DocumentIndex / TreeNode）
    每份 PDF → 章節樹（JSON 儲存）
    LLM 推理導航，不用相似度計算
```

## 查詢流程（無向量資料庫）

```
使用者提問
    ↓
Step 1: LLM 看 L0（文件摘要列表）→ 決定去哪個文件
    ↓
Step 2: LLM 看 L1（章節樹結構）→ 決定去哪個章節
    ↓
Step 3: 直接載入該章節的原始頁面文字（L2）
    ↓
Step 4: LLM 根據原文回答問題 + 附引用頁碼
```

## 安裝

```bash
pip install pdfplumber openai --break-system-packages
# 或 Anthropic
pip install pdfplumber anthropic --break-system-packages
```

## 使用方式

```bash
# 設定 API Key
export OPENAI_API_KEY=sk-...
# 或
export ANTHROPIC_API_KEY=sk-ant-...

# 互動問答（指定兩個 PDF）
python main.py --pdf1 文件A.pdf --pdf2 文件B.pdf

# 使用 Claude
python main.py --pdf1 a.pdf --pdf2 b.pdf --provider anthropic

# 使用本地 Ollama
python main.py --pdf1 a.pdf --pdf2 b.pdf --provider ollama --model llama3.2

# 直接單次查詢
python main.py --pdf1 a.pdf --pdf2 b.pdf --query "這份文件的主要結論是什麼？"

# 強制重建索引
python main.py --pdf1 a.pdf --pdf2 b.pdf --reindex
```

## 檔案結構

```
treerag/
├── main.py              # 主程式 + CLI 入口
├── pdf_parser.py        # PDF 解析，萃取每頁文字
├── tree_index.py        # 樹狀索引資料結構 + LLM 建索引
├── filesystem_index.py  # 跨文件的檔案系統索引（L0/L1/L2）
├── retriever.py         # 兩層推理式檢索引擎
├── llm_client.py        # LLM 包裝層（OpenAI/Anthropic/Ollama）
├── requirements.txt
└── README.md
```

## 索引儲存格式

索引以 JSON 儲存在 `.treerag_index/` 目錄：

```
.treerag_index/
├── filesystem.index.json    # 跨文件的入口索引
├── document_a.index.json    # 文件 A 的完整章節樹
└── document_b.index.json    # 文件 B 的完整章節樹
```

## 設計優勢

| 特性 | 說明 |
|------|------|
| 零向量資料庫 | 不需要 Pinecone / ChromaDB 等 |
| 完全可觀察 | 每步推理過程都記錄，可追蹤 |
| 輕量儲存 | 索引是普通 JSON 檔案 |
| 精準兩層 | 先找對文件，再找對段落 |
| 多 LLM 支援 | OpenAI / Anthropic / Ollama |
