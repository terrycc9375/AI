import os

import json
import re
import gc
import pandas as pd
import pickle
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
from tqdm import tqdm

@dataclass
class Config:
    # Chunking
    semantic_threshold: float = 0.75
    parent_chunk_size: int = 1024
    child_chunk_size: int = 256
    chunk_overlap: int = 128

    embed_model_name: str = "BAAI/bge-large-en-v1.5"
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
    rerank_model_name: str = "BAAI/bge-reranker-base"

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
TRAIN_SAMPLES_DIR = "train"

load_dotenv()

with open("classes.json", "r", encoding="utf-8") as f:
    classes = json.load(f)
    
class_description = str()
idx_to_class = [item["concept"] for item in classes]
for item in classes:
    class_description += f"{item['concept']}: {item['concept_desc']}\n"

print(f"Loaded {len(idx_to_class)} classes: {idx_to_class}")


def prediction(prediction_text, class_label, idx_to_class: List[str]) -> bool:
    """Check if predicted text matches the true class label"""
    all_concepts_pattern = "|".join([re.escape(c) for c in idx_to_class])
    pattern = rf"\b({all_concepts_pattern})\b"
    matches = re.findall(pattern, prediction_text, re.IGNORECASE)
    if matches:
        final_pred_lower = matches[-1].lower()
        return final_pred_lower == class_label.lower()
    return False


def index_all_papers(config: Config, embed_model, bm25_models: Dict = None) -> Dict:
    """
    Index all papers in TRAIN_SAMPLES_DIR.
    Returns a dictionary with doc_id as key, containing FAISS index and BM25 model.
    """
    if bm25_models is None:
        bm25_models = {}
    
    child_chunks_by_doc = {}
    parent_chunks_by_doc = {}
    chunk_id_to_idx_by_doc = {}
    faiss_indices = {}
    
    # Get all paper files
    paper_files = sorted(Path(TRAIN_SAMPLES_DIR).glob("paper_*.md"))
    print(f"Found {len(paper_files)} papers to index")
    
    for paper_path in tqdm(paper_files, desc="Indexing papers"):
        with open(paper_path, "r", encoding="utf-8") as f:
            full_text = f.read()
        
        doc_id = paper_path.stem  # e.g., "paper_5087"
        
        """Chunking"""
        text = re.sub(r'\b(Fig|fig|et al|vs|Dr|Prof|Sr|Jr|No|Vol|pp)\.\s', r'\1<DOT> ', full_text)
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        sentences = [s.replace('<DOT>', '.') for s in sentences]
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            continue
            
        embeddings = embed_model.encode(
            sentences,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
            device="cuda",
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
        
        """Building FAISS index and BM25 for this document"""
        texts = [c.text for c in child_chunks]
        embeddings = embed_model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
            device="cuda",
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
        
        child_chunks_by_doc[doc_id] = child_chunks
        parent_chunks_by_doc[doc_id] = parent_chunks
        chunk_id_to_idx_by_doc[doc_id] = chunk_id_to_idx
        faiss_indices[doc_id] = faiss_index
        bm25_models[doc_id] = bm25
        
        del texts, embeddings, tokenized
        gc.collect()
        torch.cuda.empty_cache()
    
    print(f"Indexed {len(child_chunks_by_doc)} documents")
    return {
        "child_chunks": child_chunks_by_doc,
        "parent_chunks": parent_chunks_by_doc,
        "chunk_id_to_idx": chunk_id_to_idx_by_doc,
        "faiss_indices": faiss_indices,
        "bm25_models": bm25_models,
    }


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
    
    del faiss_scores, faiss_indices_result, query_embedding, bm25_scores, top_indices
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


def prepare_training_data(config: Config, indices: Dict, embed_model, rerank_model, approach: str = "standard") -> Dataset:
    """
    Prepare training data by reading train.csv and retrieving evidence for each sample.
    
    Args:
        approach: "standard" | "oversample" | "undersample" | "combined"
    """
    df = pd.read_csv("train.csv")
    
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

    # Add rerank_model to indices for retrieval
    indices["rerank_model"] = rerank_model
    
    """Calculate class distribution for resampling"""
    class_counts = df["label"].value_counts().sort_index().to_dict()
    total_samples = len(df)
    
    print(f"Original class distribution: {class_counts}")
    
    formatted_data = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing training data"):
        doc_id = row["paper_id"]
        
        # Skip if document was not indexed
        if doc_id not in indices["child_chunks"]:
            print(f"Warning: {doc_id} not found in indexed documents")
            continue
        
        query = row["text"]
        label = int(row["label"])
        
        """Retrieve evidence"""
        evidences = retrieve_for_query(query, config, doc_id, indices, embed_model)
        
        """Format evidences for prompt"""
        evidences_text = "\n".join([
            f"[Chunk {i+1}]: {ev.parent_text[:500]}"
            for i, ev in enumerate(evidences)
        ])
        
        """Build prompt"""
        prompt = COT_PROMPT.format(
            class_definitions=class_description,
            evidences=evidences_text,
            text=query
        )
        
        """Get true class name"""
        true_class_name = idx_to_class[label]
        
        formatted_data.append({
            "instruction": prompt,
            "output": true_class_name,
            "label": label
        })
    
    print(f"Prepared {len(formatted_data)} training samples (before resampling)")
    
    """Apply resampling strategy"""
    if approach == "oversample":
        # Oversample minority classes to match the majority class count
        max_count = max(class_counts.values())
        resampled_data = formatted_data.copy()
        
        for label in class_counts.keys():
            samples_for_label = [d for d in formatted_data if d["label"] == label]
            current_count = len(samples_for_label)
            
            if current_count < max_count:
                shortage = max_count - current_count
                # Randomly duplicate samples from this class
                duplicates = np.random.choice(samples_for_label, size=shortage, replace=True).tolist()
                resampled_data.extend(duplicates)
                print(f"Oversampled class {label}: {current_count} -> {max_count}")
        
        formatted_data = resampled_data
        
    elif approach == "undersample":
        # Undersample majority classes to match minority class count
        min_count = min(class_counts.values())
        target_count = min_count * 4  # Balance but keep reasonable size
        resampled_data = []
        
        for label in class_counts.keys():
            samples_for_label = [d for d in formatted_data if d["label"] == label]
            current_count = len(samples_for_label)
            
            if current_count > target_count:
                # Randomly sample without replacement
                selected = np.random.choice(samples_for_label, size=target_count, replace=False).tolist()
                resampled_data.extend(selected)
                print(f"Undersampled class {label}: {current_count} -> {target_count}")
            else:
                resampled_data.extend(samples_for_label)
        
        formatted_data = resampled_data
        
    elif approach == "combined":
        # Oversample minority + adjusted learning approach (we'll handle LR in main)
        # First oversample minorities to 70% of majority
        max_count = max(class_counts.values())
        target_oversample = int(max_count * 0.7)
        resampled_data = formatted_data.copy()
        
        for label in class_counts.keys():
            samples_for_label = [d for d in formatted_data if d["label"] == label]
            current_count = len(samples_for_label)
            
            if current_count < target_oversample:
                shortage = target_oversample - current_count
                duplicates = np.random.choice(samples_for_label, size=shortage, replace=True).tolist()
                resampled_data.extend(duplicates)
                print(f"Resampled class {label}: {current_count} -> {target_oversample}")
        
        formatted_data = resampled_data
    
    print(f"Final training set size: {len(formatted_data)} samples")
    return Dataset.from_list(formatted_data)


class WeightedSFTTrainer(SFTTrainer):
    """Standard SFT trainer - Unsloth handles logits internally"""
    pass


def save_indices(indices: Dict, filepath: str = "indices_cache.pkl"):
    """Save indexed papers to disk for faster loading later"""
    print(f"Saving indices to {filepath}...")
    with open(filepath, "wb") as f:
        pickle.dump(indices, f)
    print(f"Indices saved successfully!")


def load_indices(filepath: str = "indices_cache.pkl") -> Optional[Dict]:
    """Load cached indices from disk"""
    if not Path(filepath).exists():
        return None
    print(f"Loading cached indices from {filepath}...")
    with open(filepath, "rb") as f:
        indices = pickle.load(f)
    print(f"Cached indices loaded successfully!")
    return indices


def main(approach: str = "standard"):
    """
    Training pipeline with different class imbalance handling strategies.
    
    Args:
        approach: "standard" | "oversample" | "undersample" | "combined"
            - standard: No resampling, baseline approach
            - oversample: Oversample minority classes to match majority
            - undersample: Undersample majority classes to balance dataset
            - combined: Moderate oversampling + adjusted learning rate
    """
    print(f"Starting training pipeline with approach: {approach}")
    
    HF_TOKEN = os.getenv("HF_TOKEN")
    if HF_TOKEN:
        login(token=HF_TOKEN)
    
    config = Config()
    
    """Step 1: Initialize embedding and reranking models"""
    print("Loading embedding and reranking models...")
    embed_model = SentenceTransformer(config.embed_model_name, device="cuda")
    rerank_model = CrossEncoder(config.rerank_model_name, device="cuda")
    embed_model.max_seq_length = config.max_seq_length
    
    """Step 2: Index all papers in train/ (or load from cache)"""
    print("Checking for cached indices...")
    indices = load_indices("indices_cache.pkl")
    
    if indices is None:
        print("No cache found. Indexing all papers in train/...")
        indices = index_all_papers(config, embed_model)
        save_indices(indices, "indices_cache.pkl")
    else:
        print("Using cached indices")
    
    """Step 3: Prepare training data with selected approach"""
    print(f"Preparing training dataset with '{approach}' approach...")
    dataset = prepare_training_data(config, indices, embed_model, rerank_model, approach=approach)
    
    if len(dataset) == 0:
        print("Error: No training data prepared!")
        return
    
    """Step 4: Load and prepare model"""
    print("Loading Qwen model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=32, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0.05,
    )

    """Step 5: Adjust hyperparameters based on approach"""
    if approach == "combined":
        warmup_steps = 500
        learning_rate = 1e-4
        max_steps = 100
    else:
        warmup_steps = 5
        learning_rate = 2e-4
        max_steps = 60
    
    print(f"Training config - warmup_steps: {warmup_steps}, lr: {learning_rate}, max_steps: {max_steps}")

    def formatting_func(examples):
        """Format data for SFT training - return list of formatted strings"""
        output_texts = []
        for instruction, output in zip(examples["instruction"], examples["output"]):
            text = f"""<|im_start|>system
You are an expert in detecting hallucinations in scientific paper summaries.<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""
            output_texts.append(text)
        return output_texts

    """Step 6: Train the model"""
    print("Starting training...")
    trainer = WeightedSFTTrainer(
        tokenizer=tokenizer,
        model=model,
        train_dataset=dataset,
        formatting_func=formatting_func,
        max_seq_length=MAX_SEQ_LENGTH,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            learning_rate=learning_rate,
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
    print("Training completed!")
    
    # Save model with approach suffix
    model_path = f"model_{approach}.pt"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(f"tokenizer_{approach}")
    print(f"Model saved to {model_path}!")


if __name__ == "__main__":
    import sys
    approach = sys.argv[1] if len(sys.argv) > 1 else "standard"
    main(approach=approach)
