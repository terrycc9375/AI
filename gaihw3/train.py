import os
import json
import re
import gc
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Any, List, Dict
import torch
import numpy as np
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
from huggingface_hub import login

@dataclass
class Config:
    # Chunking
    semantic_threshold: float = 0.75
    parent_chunk_size: int = 1024
    child_chunk_size: int = 256
    chunk_overlap: int = 128

    embed_model_name: str = "BAAI/bge-large-en-v1.5" # "mixedbread-ai/mxbai-embed-large-v1",
    embed_dim: int = 768
    max_seq_length: int = 512

    # BM25
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    vector_weight: float = 0.6
    bm25_weight: float = 0.4
    top_k_retrieve: int = 40    
    top_k_rerank: int = 2              

    # Reranker
    rerank_model_name: str = "BAAI/bge-reranker-base" # "cross-encoder/ms-marco-MiniLM-L-6-v2", "mixedbread-ai/mxbai-rerank-base-v1"

    output_dir: str = "./outputs"
    seed: int = 42
    
@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    level: int
    parent_id: Optional[str] = None
    start_char: int = 0
    end_char: int = 0
    metadata: dict = field(default_factory=dict)
    
@dataclass
class RetrievalResult:
    chunk: Chunk
    parent_text: str
    score: float
    vector_rank: int = -1
    bm25_rank: int = -1

@dataclass
class RerankResult:
    chunk_id: str
    doc_id: str
    text: str
    parent_text: str
    rerank_score: float
    original_score: float
    
    
# --- Global settings ---
MODEL_NAME = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 2048
OUTPUT_DIR = "outputs"

def prepare_data(config: Config, data_file: str = "train") -> Dataset:
    df = pd.read_csv("train.csv")
    with open("classes.json", "r", encoding="utf-8") as f:
        classes = json.load(f)
    class_description = str()
    idx_to_class = [item["concept"] for item in classes]
    for item in classes:
        class_description += f"{item["concept"]}: {item["concept_desc"]}\n"

    
    # COT_PROMPT = """You are an expert in detecting hallucinations in scientific paper summaries.
    # Your task is to judge whether the provided [LLM Evaluation] contains hallucinations based on the given [Scientific Evidence], and identify the specific category of the error.

    # [Category Definitions]:
    # {class_definitions}

    # [Scientific Evidence (Retrieved Chunks)]:
    # {evidences}

    # [Text to Check]:
    # {text}

    # Please think Step-by-Step:
    # 1. Extract the core claims from the [LLM Evaluation].
    # 2. Cross-reference each claim with the provided [Scientific Evidence].
    # 3. Identify any inconsistencies or unsupported statements and determine which hallucination category they fall into.

    # Finally, you must output your response in the following format:
    # [Analysis]: (Your step-by-step reasoning process)
    # [Conclusion Category]: (Insert only the category index 0-4)"""

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

Finally, output ONLY the category name:"""
    
    embed_model = SentenceTransformer(config.embed_model_name, device="cuda")
    rerank_model = CrossEncoder(config.rerank_model_name, device="cuda")
    embed_model.max_seq_length = config.max_seq_length
    
    formatted_data = []
    for _, row in df.iterrows():
        paper_path = Path(f"{data_file}/{row['paper_id']}.md")
        with open(paper_path, "r", encoding="utf-8") as f:
            full_text = f.read()
            
        """Chunking"""
        doc_id = row["paper_id"]
        text = re.sub(r'\b(Fig|fig|et al|vs|Dr|Prof|Sr|Jr|No|Vol|pp)\.\s', r'\1<DOT> ', full_text)
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        sentences = [s.replace('<DOT>', '.') for s in sentences]
        sentences = [s.strip() for s in sentences if s.strip()]
        embeddings = embed_model.encode(
            sentences,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        boundaries = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity(
                embeddings[i].reshape(1, -1),
                embeddings[i + 1].reshape(1, -1)
            )[0][0]
            if sim < config.semantic_threshold:
                boundaries.append(i + 1)
        del embeddings
        gc.collect()
        
        parent_groups = []
        current_group = []
        current_len = 0
        boundary_set = set(boundaries)
        for i, sent in enumerate(sentences):
            token_est = len(sent.split())
            if (i in boundary_set or current_len + token_est > config.parent_chunk_size) and current_group:
                parent_groups.append(current_group)
                current_group = []
                current_len = 0
            current_group.append(sent)
            current_len += token_est
        if current_group:
            parent_groups.append(current_group)
        del current_group, current_len, boundary_set
        gc.collect()
        
        child_chunks: List[Chunk] = []
        parent_chunks: Dict[str, Chunk] = {}
        chunk_id_to_idx: Dict[str, int] = {}
        faiss_index: Optional[faiss.Index] = None
        bm25: Optional[BM25Okapi] = None
        
        for p_idx, parent_sents in enumerate(parent_groups):
            parent_text = " ".join(parent_sents)
            parent_id = f"{doc_id}_p{p_idx}"

            parent_chunk = Chunk(
                chunk_id=parent_id,
                doc_id=doc_id,
                text=parent_text,
                level=1,
                parent_id=doc_id,
            )
            parent_chunks[parent_id] = parent_chunk

            children = []
            current = []
            current_len = 0
            for sent in parent_sents:
                token_est = len(sent.split())
                if current_len + token_est > config.child_chunk_size and current:
                    children.append(" ".join(current))
                    overlap_sents = []
                    overlap_len = 0
                    for s in reversed(current):
                        if overlap_len + len(s.split()) <= config.chunk_overlap:
                            overlap_sents.insert(0, s)
                            overlap_len += len(s.split())
                        else:
                            break
                    current = overlap_sents
                    current_len = overlap_len

                current.append(sent)
                current_len += token_est
            if current:
                children.append(" ".join(current))
            
            for c_idx, child_text in enumerate(children):
                child_id = f"{parent_id}_c{c_idx}"
                child_chunk = Chunk(
                    chunk_id=child_id,
                    doc_id=doc_id,
                    text=child_text,
                    level=2,
                    parent_id=parent_id,
                )
                idx = len(child_chunks)
                child_chunks.append(child_chunk)
                chunk_id_to_idx[child_id] = idx
        
        """Indexing"""
        texts = [c.text for c in child_chunks]
        embeddings = embed_model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype(np.float32)
        dim = embeddings.shape[1]
        if len(child_chunks) > 10000:
            nlist = min(256, len(child_chunks) // 10)
            quantizer = faiss.IndexFlatIP(dim)
            faiss_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            faiss_index.train(embeddings)
        else:
            faiss_index = faiss.IndexFlatIP(dim)
        faiss_index.add(embeddings)
        
        tokenized = [c.text.lower().split() for c in child_chunks]
        bm25 = BM25Okapi(
            corpus=tokenized,
            k1=config.bm25_k1,
            b=config.bm25_b
        )
        del texts, embeddings, tokenized
        gc.collect()
        
        """Retrieval"""
        query: str = row["text"]
        top_k = config.top_k_retrieve * 3
        query_embedding = embed_model.encode(
            query,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype(np.float32).reshape(1, -1)
        faiss_scores, indices = faiss_index.search(query_embedding, top_k)
        vector_results = [(int(idx), float(score)) for idx, score in zip(indices[0], faiss_scores[0]) if idx >= 0]
        query_list = query.lower().split()
        bm25_scores = bm25.get_scores(query_list)
        top_indices = np.argsort(bm25_scores)[::-1][:top_k]
        bm25_results = [(int(idx), float(bm25_scores[idx])) for idx in top_indices]
        k = 60
        rrf_scores: Dict[int, float] = {}
        rank_info: Dict[int, dict] = {}
        for rank, (idx, _) in enumerate(vector_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + config.vector_weight / (k + rank + 1)
            rank_info.setdefault(idx, {})["vector_rank"] = rank
        for rank, (idx, _) in enumerate(bm25_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + config.bm25_weight / (k + rank + 1)
            rank_info.setdefault(idx, {})["bm25_rank"] = rank
        del faiss_scores, indices, query_embedding, bm25_scores, top_indices
        gc.collect()
        
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        retrieved_results: List[RetrievalResult] = []
        for idx, score in sorted_results[:top_k]:
            chunk = child_chunks[idx]
            parent = parent_chunks.get(chunk.parent_id)
            info = rank_info.get(idx, {})
            retrieved_results.append(RetrievalResult(
                chunk=chunk,
                parent_text=parent.text if parent else "",
                score=score,
                vector_rank=info.get("vector_rank", -1),
                bm25_rank=info.get("bm25_rank", -1),
            ))
        del rrf_scores, rank_info
        gc.collect()
        torch.cuda.empty_cache()
        
        """Reranking"""
        pairs = [(query, c.parent_text) for c in retrieved_results]
        rerank_score = rerank_model.predict(pairs, show_progress_bar=False)
        reranked = sorted(
            zip(retrieved_results, rerank_score),
            key=lambda x: x[1],
            reverse=True
        )
        evidences = []
        for i, (candidate, score) in enumerate(reranked):
            if i >= config.top_k_rerank:
                break
            evidences.append(RerankResult(
                chunk_id=candidate.chunk.chunk_id,
                doc_id=candidate.chunk.doc_id,
                text=candidate.chunk.text,
                parent_text=candidate.parent_text,
                rerank_score=float(score),
                original_score=candidate.score,
            ))
        del reranked, rerank_score, pairs
        gc.collect()
        torch.cuda.empty_cache()
        
        """Building prompt"""
        prompt = COT_PROMPT.format(
            class_definitions = class_description,
            evidences = evidences,
            text = row["text"]
        )
        
        """Correct response to Qwen"""
        response = idx_to_class[int(row["label"])]
        
        formatted_data.append({
            "instruction": prompt,
            "output": response,
            "label": row["label"] # 用於計算 weight
        })
    return Dataset.from_list(formatted_data)



class WeightedSFTTrainer(SFTTrainer):
    def __init__(self, class_weights: torch.Tensor, tokenizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.tokenizer = tokenizer
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        計算加權損失，只在 output 部分應用權重
        """
        labels = inputs.get("labels").clone()
        
        # 獲取模型輸出
        outputs = model(**inputs)
        logits = outputs.logits
        
        # 計算原始損失（所有 token）
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        
        # 展平
        loss = loss_fct(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )
        
        # 只在非 -100 位置應用權重
        mask = labels.view(-1) != -100
        
        # 根據標籤應用類別權重
        label_weights = torch.ones_like(labels.view(-1), dtype=torch.float)
        for class_idx in range(len(self.class_weights)):
            class_mask = (labels.view(-1) == class_idx) & mask
            label_weights[class_mask] = self.class_weights[class_idx].item()
        
        # 計算加權損失
        weighted_loss = (loss * label_weights * mask.float()).sum() / mask.sum()
        
        return (weighted_loss, outputs) if return_outputs else weighted_loss


def main():
    HF_TOKEN = os.getenv("HF_TOKEN")
    login(token=HF_TOKEN)
    config = Config()
    class_counts = torch.tensor([1910, 1790, 249, 1616, 130], dtype=torch.float)
    class_weights = 1.0 / (class_counts / class_counts.sum())
    class_weights = class_weights / class_weights.mean()  # 正規化
    
    # Train
    dataset = prepare_data(config)

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

    def formatting_func(examples):
        """格式化數據"""
        output_texts = []
        for instruction, output in zip(examples["instruction"], examples["output"]):
            text = f"""<|im_start|>system
            You are an expert in detecting hallucinations in scientific paper summaries.<|im_end|>
            <|im_start|>user
            {instruction}<|im_end|>
            <|im_start|>assistant
            {output}<|im_end|>"""
            output_texts.append(text)
        return {"text": output_texts}

    trainer = WeightedSFTTrainer(
        class_weights=class_weights,
        tokenizer=tokenizer,
        model=model,
        train_dataset=dataset,
        formatting_func=formatting_func,
        max_seq_length=MAX_SEQ_LENGTH,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=60,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_strategy="steps",
            logging_steps=5,
            output_dir=OUTPUT_DIR,
            save_strategy="no",
            eval_strategy="no",
        )
    )
    trainer.train()

    # Evaluate
    dataset = prepare_data(config, data_file="eval")

if __name__ == "__main__":
    main()
