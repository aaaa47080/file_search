#!/bin/bash
# TreeRAG 一鍵環境設定腳本
# 用法：bash setup.sh
# 注意：請在 treerag/ 資料夾內執行

set -e

echo "======================================"
echo "  TreeRAG 環境設定"
echo "======================================"

# 確認在正確目錄
if [ ! -f "main.py" ]; then
  echo "❌ 請先 cd 到 treerag/ 資料夾再執行此腳本"
  echo "   例如：cd ~/Downloads/treerag && bash setup.sh"
  exit 1
fi

# 刪除舊的 .venv（可能是從別的機器複製過來的）
if [ -d ".venv" ]; then
  echo "🗑️  刪除舊的 .venv（重新在本機建立）..."
  rm -rf .venv
fi

# 建立新的 .venv
echo "📦 建立虛擬環境 .venv ..."
python3 -m venv .venv

# 安裝套件
echo "📥 安裝依賴套件（約需 1-2 分鐘）..."
.venv/bin/pip install --upgrade pip --quiet
.venv/bin/pip install -r requirements.txt --quiet

echo ""
echo "======================================"
echo "✅ 環境設定完成！"
echo ""
echo "接下來執行："
echo ""
echo "  source .venv/bin/activate"
echo ""
echo "  python main.py \\"
echo "    --pdf1 \"TSMC 2025Q1 Consolidated Financial Statements_C.pdf\" \\"
echo "    --pdf2 \"form10-k.pdf\" \\"
echo "    --provider openai \\"
echo "    --model gpt-4o-mini \\"
echo "    --api-key \"sk-...\""
echo "======================================"
