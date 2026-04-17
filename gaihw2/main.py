import json
import os
import torch
from sentence_transformers import SentenceTransformer, util, SentenceTransformerTrainingArguments, SentenceTransformerTrainer
import sentence_transformers.sentence_transformer.losses as losses
from torch.utils.data import DataLoader
from torch import optim
from datasets import Dataset
from langchain_ollama import OllamaLLM
from huggingface_hub import login
login("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")

# 1. 載入模型
# BGE-M3 支援多功能檢索，這裡使用其 Dense Embedding 功能
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

llm = OllamaLLM(model="llama3.2:3b")

def load_private_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_evidence(embed_model, question, full_text, top_k=10):
    """
    將 full_text 進行分割並透過 BGE-M3 檢索最相關的片段
    """
    chunks = [c.strip() for c in full_text.split('\n') if len(c.strip()) > 10]
    
    if not chunks:
        return []

    with torch.no_grad():
        q_emb = embed_model.encode(question, convert_to_tensor=True)
        c_embs = embed_model.encode(
            chunks, 
            convert_to_tensor=True,
            batch_size=4,
            show_progress_bar=False,
        )

        cos_scores = util.cos_sim(q_emb, c_embs)[0]
        top_results = torch.topk(cos_scores, k=min(top_k, len(chunks)))
    
    indices = top_results.indices.cpu().tolist()
    evidence_list = [chunks[i] for i in indices]
    
    del q_emb, c_embs, cos_scores
    return evidence_list

def main():
    # 1. 參數設定
    dataset_path = "public_dataset.json"
    os.makedirs("models", exist_ok=True)
    
    # 2. 載入資料
    subset_data = load_private_data(dataset_path)

    args = SentenceTransformerTrainingArguments(
        # --- 輸出與儲存 ---
        output_dir="models/bge-m3-finetuned", # 模型儲存路徑
        # overwrite_output_dir=True,
        
        # --- 訓練超參數 ---
        num_train_epochs=5,                     # 訓練輪數，RAG 通常 3-5 輪
        per_device_train_batch_size=2,          # 12GB VRAM 建議設 4-8 (BGE-M3 較大)
        gradient_accumulation_steps=4,          # 梯度累積：這會讓「實質 Batch Size」= 4*4 = 16
        learning_rate=2e-5,                     # 微調通常使用較小的學習率
        warmup_steps=0.1,                       # 前 10% 的步數會緩慢增加學習率
        weight_decay=0.01,                      # 防止過擬合
        
        # --- 效能與硬體優化 ---
        fp16=True,                              # 開啟半精度訓練，能省一半顯存並加速 (必開)
        bf16=False,                             # 如果是 30/40 系列顯卡可改開 bf16=True, fp16=False
        gradient_checkpointing=True,            # 以時間換空間，能大幅減少顯存佔用 (推論慢一點但能跑更大 Batch)
        
        # --- 評估與紀錄 ---
        eval_strategy="no",                  # 每隔固定步數評估一次
        # eval_steps=100,                         # 每 100 步評估一次
        save_strategy="no",                  # 與 eval 保持一致
        # save_total_limit=2,                     # 最多只保留 2 個模型存檔，否則硬碟會爆
        # logging_steps=10,                       # 每 10 步在終端機顯示一次進度
        
        # --- 報告 ---
        report_to="none",                       # 若沒有用 wandb，設為 none 避免噴警告
    )

    data_list = []
    for item in subset_data:
        question = item.get("question", "")
        gt_evidences = item.get("evidence", [])
        for gt_ev in gt_evidences:
            # MultipleNegativesRankingLoss 需要一對 (query, positive)
            data_list.append({
                "query": question,
                "positive": gt_ev
            })

    embed_model = SentenceTransformer("BAAI/bge-m3", device='cuda')
    train_dataset = Dataset.from_list(data_list)
    train_loss = losses.MultipleNegativesRankingLoss(model=embed_model)

    trainer = SentenceTransformerTrainer(
        model=embed_model,
        args=args,
        train_dataset=train_dataset, # 註：這裡需將 InputExample 轉為 Dataset 格式
        loss=train_loss,
    )
    trainer.train()

    trainer.save_model("models/bge-m3-finetuned")

def rag():
    raw_data = load_private_data("public_dataset.json")
    final_results = []
    embed_model = SentenceTransformer("models/bge-m3-finetuned", device='cuda')

    for item in raw_data:
        title = item.get("title", "")
        question = item.get("question", "")
        full_text = item.get("full_text", "") # 根據你提供的 JSON 格式

        # 步驟 1: 檢索 Evidence (設定 top_k=10 作為範例)
        retrieved_evidences = get_evidence(embed_model, question, full_text, top_k=10)

        # 步驟 2: 組合 Prompt
        # 參考 score_public.py 的格式，讓 LLM 容易理解
        context = "\n".join([f"- {ev}" for ev in retrieved_evidences])
        prompt = (
            f"Background: {title}\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            f"Answer correctly based on the context. If the answer is not in the context, say you don't know."
        )

        # 步驟 3: 送入 Llama 產生回答
        prediction = llm.invoke(prompt)

        # 封裝結果
        final_results.append({
            "title": title,
            "answer": prediction,
            "evidence": retrieved_evidences
        })
        
        print(f"Processed: {title[:30]}...")

    # 儲存結果供評分腳本使用
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    rag()