import os
import json
import pandas as pd
import torch
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from pathlib import Path
from sentence_transformers import SentenceTransformer

# local modules from hw2
from config import RAGConfig
from chunker import Chunk, SemanticChunker
from indexer import HybridIndexer
from retriever import HybridRetriever
from reranker import Reranker, RerankResult

# --- settings ---
MODEL_NAME = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 2048
OUTPUT_DIR = "outputs"

class_counts = torch.tensor([1910, 1790, 249, 1616, 130], dtype=torch.float)
class_weights = 1.0 / (class_counts / class_counts.sum())
class_weights = class_weights / class_weights.mean()  # 正規化

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    load_in_4bit = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 32, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32,
    lora_dropout = 0.05,
)

COT_PROMPT = """You are an expert in detecting hallucinations in scientific paper summaries.
Your task is to judge whether the provided [LLM Evaluation] contains hallucinations based on the given [Scientific Evidence], and identify the specific category of the error.

[Category Definitions]:
{class_definitions}

[Scientific Evidence (Retrieved Chunks)]:
{evidences}

[Text to Check]:
{text}

Please think Step-by-Step:
1. Extract the core claims from the [LLM Evaluation].
2. Cross-reference each claim with the provided [Scientific Evidence].
3. Identify any inconsistencies or unsupported statements and determine which hallucination category they fall into.

Finally, you must output your response in the following format:
[Analysis]: (Your step-by-step reasoning process)
[Conclusion Category]: (Insert only the category index 0-4)"""

def prepare_data():
    # loading data and tools
    df = pd.read_csv("train.csv")
    with open("classes.json", "r", encoding="utf-8") as f:
        classes = json.load(f)
    config = RAGConfig()
    rag_model = SentenceTransformer(config.embed_model_name)
    chunker = SemanticChunker(rag_model, config)
    indexer = HybridIndexer(rag_model, config)
    retriever = HybridRetriever(rag_model, indexer, config)
    reranker = Reranker(config)
    
    formatted_data = []
    for _, row in df.iterrows():
        paper_path = Path(f"train/{row['paper_id']}.md")
        class_description = str()
        for item in classes:
            class_description += f"{item["concept"]}: {item["concept_desc"]}\n"
        
        # 這裡假設你已經有 RAG 邏輯選出的 chunks
        # 如果還沒寫，暫時讀取 md 前 1000 字作為示範
        evidences = "檔案不存在"
        if paper_path.exists():
            with open(paper_path, "r", encoding="utf-8") as f:
                evidences = f.read()[:1500] # 實務上應替換為你的 Reranker 結果

        prompt = COT_PROMPT.format(
            class_definitions = class_description,
            evidences = evidences,
            text = row["text"]
        )
        
        # 訓練標籤格式化
        response = f"【分析】: ... \n【結論類別】: {row['label']}"
        
        formatted_data.append({
            "instruction": prompt,
            "output": response,
            "label": row["label"] # 用於計算 weight
        })
    return Dataset.from_list(formatted_data)

dataset = prepare_data()

# --- 5. 自定義 Trainer 以支援加權損失 ---
class WeightedSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # 這裡需要根據 inputs 中的 label 來決定權重
        # SFT 通常是對整個 sequence 計算 CrossEntropy
        # 為了簡單處理不平衡，可以在 Data Collator 階段對數量少的樣本進行 Oversampling
        # 或者在此覆寫 Loss function
        return super().compute_loss(model, inputs, return_outputs)

# --- 6. 開始訓練 ---
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text", # 需根據 formatting_func 調整
    max_seq_length = MAX_SEQ_LENGTH,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60, # 根據需求調整
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_strategy="none",
        logging_steps = 1,
        output_dir = OUTPUT_DIR,
    ),
)

trainer.train()