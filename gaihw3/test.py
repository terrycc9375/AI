import os
import json
import re
import gc
import pandas as pd
import pickle
import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict
from pathlib import Path
from unsloth import FastLanguageModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import faiss
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

with open("classes.json", "r", encoding="utf-8") as f:
    classes = json.load(f)

idx_to_class = [item["concept"] for item in classes]
class_to_idx = {item["concept"]: i for i, item in enumerate(classes)}

print(f"Loaded {len(idx_to_class)} classes: {idx_to_class}")

@dataclass
class Config:
    semantic_threshold: float = 0.75
    parent_chunk_size: int = 1024
    child_chunk_size: int = 256
    chunk_overlap: int = 128
    embed_model_name: str = "BAAI/bge-large-en-v1.5"
    embed_dim: int = 768
    max_seq_length: int = 512
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    vector_weight: float = 0.6
    bm25_weight: float = 0.4
    top_k_retrieve: int = 40    
    top_k_rerank: int = 2
    rerank_model_name: str = "BAAI/bge-reranker-base"

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    level: int
    parent_id: Optional[str] = None
    start_char: int = 0
    end_char: int = 0
    metadata: dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

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

MODEL_NAME = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 2048
TRAIN_SAMPLES_DIR = "train"

def load_indices(filepath: str = "indices_cache.pkl") -> Optional[Dict]:
    """Load cached indices from disk"""
    if not Path(filepath).exists():
        return None
    print(f"Loading cached indices from {filepath}...")
    with open(filepath, "rb") as f:
        indices = pickle.load(f)
    print(f"Cached indices loaded successfully!")
    return indices

def retrieve_for_query(
    query: str,
    config: Config,
    doc_id: str,
    indices: Dict,
    embed_model,
) -> List[RerankResult]:
    """Retrieve and rerank results for a single query"""
    child_chunks = indices["child_chunks"][doc_id]
    parent_chunks = indices["parent_chunks"][doc_id]
    faiss_index = indices["faiss_indices"][doc_id]
    bm25 = indices["bm25_models"][doc_id]
    rerank_model = indices.get("rerank_model")
    
    top_k = config.top_k_retrieve * 3
    
    """Retrieval with FAISS + BM25"""
    query_embedding = embed_model.encode(
        query,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype(np.float32).reshape(1, -1)
    
    faiss_scores, faiss_indices_result = faiss_index.search(query_embedding, top_k)
    vector_results = [(int(idx), float(score)) for idx, score in zip(faiss_indices_result[0], faiss_scores[0]) if idx >= 0]
    
    query_list = query.lower().split()
    bm25_scores = bm25.get_scores(query_list)
    top_indices = np.argsort(bm25_scores)[::-1][:top_k]
    bm25_results = [(int(idx), float(bm25_scores[idx])) for idx in top_indices]
    
    """RRF combination"""
    k = 60
    rrf_scores: Dict[int, float] = {}
    rank_info: Dict[int, dict] = {}
    for rank, (idx, _) in enumerate(vector_results):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + config.vector_weight / (k + rank + 1)
        rank_info.setdefault(idx, {})["vector_rank"] = rank
    for rank, (idx, _) in enumerate(bm25_results):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + config.bm25_weight / (k + rank + 1)
        rank_info.setdefault(idx, {})["bm25_rank"] = rank
    
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
    rerank_scores = rerank_model.predict(pairs, show_progress_bar=False)
    reranked = sorted(
        zip(retrieved_results, rerank_scores),
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
    
    del reranked, rerank_scores, pairs
    gc.collect()
    torch.cuda.empty_cache()
    
    return evidences

def predict_sample(
    query: str,
    doc_id: str,
    config: Config,
    indices: Dict,
    embed_model,
    model,
    tokenizer,
    class_descriptions: str,
    max_retries: int = 3
) -> tuple:
    """
    Predict hallucination class for a single sample.
    Returns (predicted_class, predicted_label, confidence_score)
    """
    
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
    
    try:
        # Retrieve evidence
        evidences = retrieve_for_query(query, config, doc_id, indices, embed_model)
        
        # Format evidences
        evidences_text = "\n".join([
            f"[Chunk {i+1}]: {ev.parent_text[:500]}"
            for i, ev in enumerate(evidences)
        ])
        
        # Build prompt
        prompt = COT_PROMPT.format(
            class_definitions=class_descriptions,
            evidences=evidences_text,
            text=query
        )
        
        # Format for model
        full_prompt = f"""<|im_start|>system
You are an expert in detecting hallucinations in scientific paper summaries.<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""
        
        # Tokenize and generate
        inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        prediction_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract predicted class
        all_concepts_pattern = "|".join([re.escape(c) for c in idx_to_class])
        pattern = rf"\b({all_concepts_pattern})\b"
        matches = re.findall(pattern, prediction_text, re.IGNORECASE)
        
        if matches:
            final_pred = matches[-1]
            predicted_label = class_to_idx.get(final_pred, -1)
            confidence = 0.8  # Heuristic confidence
            return final_pred, predicted_label, confidence, prediction_text
        else:
            return "Unknown", -1, 0.0, prediction_text
            
    except Exception as e:
        print(f"Error predicting sample: {e}")
        return "Error", -1, 0.0, ""

def evaluate_on_dev_set(model_path: str = "model_standard.pt", approach: str = "standard"):
    """Evaluate model on dev.csv"""
    
    print(f"Starting evaluation on dev.csv using model: {model_path}")
    
    # Load dev set
    if not Path("dev.csv").exists():
        print("Error: dev.csv not found!")
        return
    
    dev_df = pd.read_csv("dev.csv")
    print(f"Loaded {len(dev_df)} test samples from dev.csv")
    
    # Load config and indices
    config = Config()
    indices = load_indices("indices_cache.pkl")
    
    if indices is None:
        print("Error: No cached indices found. Please run train.py first.")
        return
    
    # Load embedding and reranking models
    print("Loading embedding and reranking models...")
    embed_model = SentenceTransformer(config.embed_model_name, device="cuda")
    rerank_model = CrossEncoder(config.rerank_model_name, device="cuda")
    embed_model.max_seq_length = config.max_seq_length
    indices["rerank_model"] = rerank_model
    
    # Load trained model
    print(f"Loading trained model from {model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    
    # Load LoRA weights
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, model_path)
    model = model.eval()
    
    # Load class descriptions
    with open("classes.json", "r", encoding="utf-8") as f:
        classes = json.load(f)
    class_descriptions = "\n".join([f"{item['concept']}: {item['concept_desc']}" for item in classes])
    
    # Run predictions
    predictions = []
    true_labels = []
    pred_labels = []
    confidences = []
    
    for idx, row in tqdm(dev_df.iterrows(), total=len(dev_df), desc="Evaluating"):
        doc_id = row["paper_id"]
        text = row["text"]
        true_label = int(row["label"]) if "label" in row and pd.notna(row["label"]) else -1
        
        # Skip if document not in indices
        if doc_id not in indices["child_chunks"]:
            print(f"Skipping sample {idx}: {doc_id} not found in indexed documents")
            continue
        
        pred_class, pred_label, confidence, pred_text = predict_sample(
            text, doc_id, config, indices, embed_model, model, tokenizer, class_descriptions
        )
        
        predictions.append({
            "sample_id": idx,
            "paper_id": doc_id,
            "text": text,
            "true_label": true_label,
            "true_class": idx_to_class[true_label] if true_label >= 0 else "Unknown",
            "predicted_label": pred_label,
            "predicted_class": pred_class,
            "confidence": confidence,
            "generation": pred_text[:200]
        })
        
        true_labels.append(true_label)
        pred_labels.append(pred_label)
        confidences.append(confidence)
    
    # Filter out invalid predictions
    valid_indices = [i for i in range(len(predictions)) if pred_labels[i] >= 0 and true_labels[i] >= 0]
    true_labels_valid = [true_labels[i] for i in valid_indices]
    pred_labels_valid = [pred_labels[i] for i in valid_indices]
    
    # Compute metrics
    if len(true_labels_valid) > 0:
        accuracy = accuracy_score(true_labels_valid, pred_labels_valid)
        f1_weighted = f1_score(true_labels_valid, pred_labels_valid, average='weighted', zero_division=0)
        f1_macro = f1_score(true_labels_valid, pred_labels_valid, average='macro', zero_division=0)
        
        precision_weighted = precision_score(true_labels_valid, pred_labels_valid, average='weighted', zero_division=0)
        recall_weighted = recall_score(true_labels_valid, pred_labels_valid, average='weighted', zero_division=0)
        
        # Per-class metrics
        per_class_metrics = {}
        for class_idx, class_name in enumerate(idx_to_class):
            mask = np.array(true_labels_valid) == class_idx
            if mask.sum() > 0:
                pred_mask = np.array(pred_labels_valid)[mask]
                class_accuracy = (pred_mask == class_idx).sum() / len(pred_mask)
                per_class_metrics[class_name] = {
                    "count": int(mask.sum()),
                    "accuracy": float(class_accuracy)
                }
        
        # Confusion matrix
        cm = confusion_matrix(true_labels_valid, pred_labels_valid, labels=range(len(idx_to_class)))
        
        results = {
            "approach": approach,
            "model_path": model_path,
            "num_samples": len(predictions),
            "num_valid_predictions": len(valid_indices),
            "accuracy": float(accuracy),
            "f1_weighted": float(f1_weighted),
            "f1_macro": float(f1_macro),
            "precision_weighted": float(precision_weighted),
            "recall_weighted": float(recall_weighted),
            "per_class_metrics": per_class_metrics,
            "confusion_matrix": cm.tolist(),
            "class_names": idx_to_class
        }
        
        print("\n" + "="*60)
        print(f"Evaluation Results for Approach: {approach}")
        print("="*60)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score (Weighted): {f1_weighted:.4f}")
        print(f"F1 Score (Macro): {f1_macro:.4f}")
        print(f"Precision (Weighted): {precision_weighted:.4f}")
        print(f"Recall (Weighted): {recall_weighted:.4f}")
        print("\nPer-Class Metrics:")
        for class_name, metrics in per_class_metrics.items():
            print(f"  {class_name}: {metrics['count']} samples, {metrics['accuracy']:.4f} accuracy")
        print("="*60 + "\n")
        
        # Save results
        results_file = f"evaluation_results_{approach}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_file}")
        
        # Save predictions
        pred_file = f"predictions_{approach}.json"
        with open(pred_file, "w") as f:
            json.dump(predictions, f, indent=2)
        print(f"Predictions saved to {pred_file}")
        
        return results
    else:
        print("Error: No valid predictions generated!")
        return None

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        approach = sys.argv[1]
        model_path = f"model_{approach}.pt"
    else:
        approach = "standard"
        model_path = "model_standard.pt"
    
    evaluate_on_dev_set(model_path=model_path, approach=approach)
