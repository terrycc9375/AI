# 📝 112511071 RAG 解決方案技術報告 (Technical Report)

這份報告詳細說明了我們目前的 RAG (Retrieval-Augmented Generation) 系統架構、性能優化策略以及核心技術細節，方便進行學術討論與同儕評閱。

---

## 🏗️ 系統總覽 (System Overview)
本方案採用了 **混合搜尋 (Hybrid Search)** 與 **重排序 (Reranking)** 為核心的兩階段檢索架構，並搭配本地端部署的 Llama 3.2 3B 指令微調模型進行生成。

### 技術棧 (Tech Stack)
*   **LLM**: `meta-llama/Llama-3.2-3B-Instruct` (透過 Ollama 與 LiteLLM 代理)
*   **Embedding**: `BAAI/bge-base-en-v1.5` (本地端 HuggingFace)
*   **Reranker**: `BAAI/bge-reranker-base` (Cross-Encoder)
*   **Vector DB**: `FAISS` (Facebook AI Similarity Search)
*   **Framework**: `LangChain`

---

## 🚀 核心優化策略 (Optimization Strategies)

### 1. 斷句策略 (Chunking Strategy)
*   **設定**: `chunk_size=300`, `chunk_overlap=80`
*   **用意**: 
    *   **高精準度**: 論文內容通常細碎，較小的 Chunk 能減少無關文字進入 Context。
    *   **ROUGE-L 友好**: 較短且精準的片段在計算 Evidence Score 時，更容易獲得較高的 F-measure。

### 2. 多路檢索與融合 (Hybrid Retrieval & RRF)
我們不只依賴向量搜尋，而是結合了兩種檢索方式：
1.  **向量檢索 (Dense Retrieval)**: 捕捉語意相似度。
2.  **BM25 (Sparse Retrieval)**: 捕捉關鍵字精確匹配（對學術名詞與數值非常有效）。
*   **融合算法**: 使用 **RRF (Reciprocal Rank Fusion)**。這是一種無需模型訓練的融合方式，能穩定地合併兩者的優點，產出前 20 個候選片段。

### 3. Cross-Encoder 重排序 (Reranking)
*   **技術**: 使用 `bge-reranker-base`。
*   **優勢**: 不同於 Embedding 的雙編碼器 (Bi-encoder)，Cross-Encoder 會同時輸入問句與片段，捕捉深度交互語意。
*   **動態 k 值**: 我們不固定回傳數量，而是根據 Rerank 分數動態選擇前 3 到 5 個最相關的片段（score > 0）。這能確保提供的 Evidence 質量極高，不會混入雜訊。

### 4. 異常處理與重試機制 (Error Handling)
*   **Retry Logic**: 針對 LLM API 呼叫設計了 **5 次自動重試** 與 **15秒指數退避**。
*   **即時儲存**: 程式每做完一題就會立即寫入 JSON 檔。這確保了即使在處理 100 篇論文的長途過程中發生斷電或崩潰，已完成的進度也不會遺失。

---

## 📊 效能評估 (Evaluation Results)

根據目前的 **Public Dataset (100 題)** 測試結果：

| 指標 | 目前分數 | Weak Baseline | Strong Baseline | 狀態 |
| :--- | :--- | :--- | :--- | :--- |
| **Evidence Score** | **~0.29xx** | 0.2124 | 0.2619 | 🟢 超越 Strong |
| **Correctness** | **~0.xx** | 0.33 | 0.48 | 🟡 測試中 |

> [!TIP]
> **討論重點**：我們發現在處理含有大量數值或數學公式的論文時，BM25 的權重提升有助於 Correctness 的表現。

---

## 🛠️ 本地執行環境說明
為了符合 TA 的規範，我們搭建了以下環境：
1.  **LiteLLM (Port 8091)**: 用於 RAG 推論，過濾不支援參數。
2.  **Custom Proxy (Port 8095)**: 專門解決官方 `score_public.py` 與 Llama 3.2 之前的格式衝突，確保 Correctness 能夠被正確計分。

---
*本方案由 112511071 設計，旨在追求檢索精度與生成準確度的最大平衡度。*
