"""
_phase2_verify.py
─────────────────────────────────────────────────────
Phase 2: Verification of page numbers (PageIndex check_title_appearance)
"""

import re
from concurrent.futures import ThreadPoolExecutor
from utils import extract_json, retry_llm_call


class _Phase2VerifyMixin:
    """Phase 2: LLM 驗證每個章節的實際頁碼（fuzzy matching）"""

    def _verify_page_numbers(self, structure_list: list[dict], pdf_doc) -> list[dict]:
        """
        Phase 2：驗證頁碼，對齊 PageIndex meta_processor 三層邏輯：

        1. 初步驗證（check_title_appearance）— 標題是否出現在指定頁
        2. Gap 4：若最後節點頁碼 < 總頁數一半 → 結構嚴重不完整，回傳 [] 觸發降級
        3. Gap 2：正確率 > 60% → 修復錯誤項目（最多 3 輪）
                  正確率 ≤ 60% → 回傳 [] 觸發降級
        4. Gap 3：appear_start 第二輪驗證 — 標題是否在頁面開頭出現
                  影響 _build_tree 的 end_page 邊界計算
        """
        if not structure_list:
            return structure_list

        # 書籤來源不需要驗證（已確認）
        if pdf_doc.heading_source == "bookmarks":
            for item in structure_list:
                item["verified"] = True
                item["appear_start"] = True
            return structure_list

        total_pages = pdf_doc.total_pages

        # ── 先剔除超出頁碼範圍的項目（截斷 PDF 問題）──
        for item in structure_list:
            if item.get("physical_index", 1) > total_pages:
                item["physical_index"] = None  # 對齊 PageIndex validate_and_truncate

        valid_list = [i for i in structure_list if i.get("physical_index") is not None]
        if not valid_list:
            return []

        # ── 第一輪：check_title_appearance（並行）──
        def verify_one(item):
            page_num = item["physical_index"]
            page = pdf_doc.get_page(page_num)
            if not page:
                item["verified"] = False
                return item

            # 對齊官方 check_title_appearance：
            # - 模糊匹配，忽略空格差異
            # - 加入 "thinking" 欄位（chain-of-thought，提升準確率）
            # - 回傳 "answer": "yes"/"no"（與官方完全一致）
            # - 送完整頁面文字，不截斷
            # Normalise whitespace in both title and page text to eliminate
            # spaced-character mismatches common in Chinese financial PDFs
            # (e.g. TOC stores "合併資產負債表" but PDF text has "合 併 資 產 負 債 表")
            norm_title = re.sub(r'\s+', ' ', item['title']).strip()
            norm_page_text = re.sub(r'\s+', ' ', page.text).strip()

            prompt = f"""Your job is to check if the given section appears or starts in the given page_text.

Note: do fuzzy matching, ignore any space inconsistency in the page_text.

The given section title is {norm_title}.
The given page_text is {norm_page_text}.

Reply format:
{{
    "thinking": <why do you think the section appears or starts in the page_text>,
    "answer": "yes or no" (yes if the section appears or starts in the page_text, no otherwise)
}}
Directly return the final JSON structure. Do not output anything else."""

            try:
                resp = retry_llm_call(
                    self.llm, [{"role": "user", "content": prompt}],
                    max_tokens=200, label=f"驗證:{item['title'][:15]}"
                )
                data = extract_json(resp)
                if isinstance(data, dict):
                    answer = str(data.get("answer", "no")).lower().strip()
                    item["verified"] = answer == "yes"
                else:
                    item["verified"] = True  # 解析失敗預設通過
            except Exception:
                item["verified"] = True
            return item

        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            verified = list(executor.map(verify_one, valid_list))

        # 恢復原始順序
        order = {item.get("structure", ""): i for i, item in enumerate(structure_list)}
        verified.sort(key=lambda x: order.get(x.get("structure", ""), 0))

        # ── Gap 4：早期放棄檢查 ──
        # 若結構的最後一個頁碼 < 文件一半 → 結構嚴重殘缺，觸發降級
        last_page = max((v.get("physical_index", 1) for v in verified), default=1)
        if last_page < total_pages / 2:
            print(f"   ⚠ 結構僅覆蓋到第 {last_page} 頁（總 {total_pages} 頁），觸發降級")
            return []

        # ── Gap 2：正確率計算與修復循環 ──
        correct = [v for v in verified if v.get("verified", True)]
        accuracy = len(correct) / len(verified) if verified else 0
        print(f"   → 驗證正確率：{accuracy*100:.0f}% （{len(correct)}/{len(verified)}）")

        # Threshold lowered from 0.6 → 0.5: only degrade to single-node when
        # strictly fewer than half the chapters can be located (more lenient for
        # Chinese financial PDFs where font-spaced text causes marginal failures)
        if accuracy <= 0.5:
            print("   ⚠ 正確率 ≤ 50%，觸發降級")
            return []

        if accuracy < 1.0:
            # 有錯誤項目，嘗試修復（最多 3 輪）
            for attempt in range(3):
                incorrect = [v for v in verified if not v.get("verified", True)]
                if not incorrect:
                    break
                print(f"   → 修復第 {attempt+1} 輪，{len(incorrect)} 個錯誤項目...")
                self._repair_incorrect_items(verified, pdf_doc)
                # 重新計算是否還有未修復的
                still_bad = [v for v in verified if not v.get("verified", True)]
                if len(still_bad) == len(incorrect):
                    break  # 修復無進展，停止

        # ── Gap 3：appear_start 第二輪驗證（並行）──
        # 判斷標題是否在頁面「開頭」出現，影響前一節的 end_page
        verified_ok = [v for v in verified if v.get("verified", True)]

        def check_appear_start(item):
            # 官方：physical_index=None 的項目直接設 appear_start=False（'no'），
            # 不需要 LLM 查詢（這些項目已被 validate_and_truncate 排除）
            if item.get("physical_index") is None:
                item["appear_start"] = False
                return item

            page_num = item["physical_index"]
            page = pdf_doc.get_page(page_num)
            if not page:
                item["appear_start"] = False  # 官方：無法取頁時設 'no'
                return item

            # 官方 check_title_appearance_in_start 判斷標準：
            # 標題是否是頁面「第一個內容」（前面沒有其他段落文字）
            # 送完整頁面文字，不截斷（官方不截斷）
            prompt = f"""你將收到一個章節標題和對應的頁面文字。
請判斷該章節標題是否是頁面文字的「第一個內容」。
若標題前面還有其他段落或文字，回答 no。
若標題是頁面的第一件事，回答 yes。

注意：模糊匹配，忽略空格差異。

章節標題：{item['title']}
頁面文字：
{page.text}

回答格式 JSON：{{"thinking": "...", "start_begin": "yes"}} 或 {{"thinking": "...", "start_begin": "no"}}
只輸出 JSON，不要其他文字。"""

            try:
                resp = retry_llm_call(
                    self.llm, [{"role": "user", "content": prompt}],
                    max_tokens=150, label=f"起始:{item['title'][:15]}"
                )
                data = extract_json(resp)
                if isinstance(data, dict):
                    # 官方存 "yes"/"no" 字串；我們轉為 bool 供後續比較
                    start_begin = data.get("start_begin", "no")
                    item["appear_start"] = (str(start_begin).lower() == "yes")
                else:
                    item["appear_start"] = False
            except Exception:
                item["appear_start"] = False
            return item

        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            verified_ok = list(executor.map(check_appear_start, verified_ok))

        return verified_ok

    def _repair_incorrect_items(self, verified: list[dict], pdf_doc):
        """
        修復驗證失敗的項目（對齊 PageIndex fix_incorrect_toc）：
        找到每個失敗項目的「前後最近正確項目」，
        在那個頁碼範圍內重新用 LLM 找正確頁碼，並重新驗證。
        """
        total_pages = pdf_doc.total_pages
        correct_indices = {
            i: v["physical_index"]
            for i, v in enumerate(verified)
            if v.get("verified", True)
        }

        def fix_one(idx_item):
            idx, item = idx_item

            # 找前後最近正確項目的頁碼邊界
            prev_page = 1
            for i in range(idx - 1, -1, -1):
                if i in correct_indices:
                    prev_page = correct_indices[i]
                    break

            next_page = total_pages
            for i in range(idx + 1, len(verified)):
                if i in correct_indices:
                    next_page = correct_indices[i]
                    break

            # 組合搜尋範圍的頁面文字（對齊官方 single_toc_item_index_fixer：
            # 用 <physical_index_X> 標籤包每頁完整文字，不截斷單頁文字）
            range_parts = []
            for p in range(prev_page, min(next_page + 1, total_pages + 1)):
                page = pdf_doc.get_page(p)
                if page:
                    range_parts.append(
                        f"<physical_index_{p}>\n{page.text}\n<physical_index_{p}>\n\n"
                    )
            if not range_parts:
                return

            range_text = "".join(range_parts)

            # 對齊官方 single_toc_item_index_fixer prompt 結構：
            # - 先描述任務，再標示 Section Title，再給 Document pages
            # - 輸出 <physical_index_X> 格式（更可靠，LLM 直接複製標籤）
            prompt = f"""You are given a section title and several pages of a document, your job is to find the physical index of the start page of the section in the partial document.

The provided pages contains tags like <physical_index_X> and <physical_index_X> to indicate the physical location of the page X.

Reply in a JSON format:
{{
    "thinking": <explain which page, started and closed by <physical_index_X>, contains the start of this section>,
    "physical_index": "<physical_index_X>" (keep the format)
}}
Directly return the final JSON structure. Do not output anything else.
Section Title:
{item['title']}
Document pages:
{range_text}"""

            try:
                resp = retry_llm_call(
                    self.llm, [{"role": "user", "content": prompt}],
                    max_tokens=200, label=f"修復:{item['title'][:15]}"
                )
                data = extract_json(resp)
                if not isinstance(data, dict):
                    return
                raw_pi = data.get("physical_index")
                if raw_pi is None:
                    return
                # 官方 convert_physical_index_to_int：從 "<physical_index_X>" 提取整數，
                # 或直接是整數均相容。
                if isinstance(raw_pi, int):
                    new_page = raw_pi
                else:
                    m = re.search(r"physical_index_(\d+)", str(raw_pi))
                    new_page = int(m.group(1)) if m else None
                if new_page is None:
                    return
                if not (prev_page <= new_page <= next_page):
                    return

                # 重新驗證新頁碼
                page = pdf_doc.get_page(new_page)
                if not page:
                    return

                # 對齊官方 fix_incorrect_toc 再驗步驟：
                # 呼叫與 check_title_appearance 相同格式（thinking + answer yes/no）
                v_prompt = f"""Your job is to check if the given section appears or starts in the given page_text.

Note: do fuzzy matching, ignore any space inconsistency in the page_text.

The given section title is {item['title']}.
The given page_text is {page.text}.

Reply format:
{{
    "thinking": <why do you think the section appears or starts in the page_text>,
    "answer": "yes or no" (yes if the section appears or starts in the page_text, no otherwise)
}}
Directly return the final JSON structure. Do not output anything else."""

                v_resp = retry_llm_call(
                    self.llm, [{"role": "user", "content": v_prompt}],
                    max_tokens=200, label=f"再驗:{item['title'][:15]}"
                )
                v_data = extract_json(v_resp)
                if isinstance(v_data, dict):
                    answer = str(v_data.get("answer", "no")).lower().strip()
                    if answer == "yes":
                        item["physical_index"] = new_page
                        item["verified"] = True
            except Exception:
                pass

        incorrect = [(i, v) for i, v in enumerate(verified) if not v.get("verified", True)]
        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            list(executor.map(fix_one, incorrect))

    def _add_preface_if_needed(self, structure_list: list[dict]) -> list[dict]:
        """
        Gap 5：對齊 PageIndex add_preface_if_needed。
        若結構列表中第一個章節的起始頁不是第 1 頁，
        則自動在最前面插入一個「Preface / 前言」節點，
        覆蓋第 1 頁到第一個章節前一頁的範圍。
        """
        if not structure_list:
            return structure_list
        first_page = structure_list[0].get("physical_index", 1)
        if first_page <= 1:
            return structure_list  # 第一節就從第 1 頁開始，無需補序章
        # 插入虛擬前言節點（level=0，structure code="0"）
        preface = {
            "title": "Preface",
            "structure": "0",
            "physical_index": 1,
            "level": 0,
            "appear_start": True,
            "verified": True,
        }
        print(f"   → 自動插入前言節點（第 1-{first_page - 1} 頁）")
        return [preface] + structure_list
