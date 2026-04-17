import os
import json
import random
import numpy as np
import torch
import gc
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import RAGConfig
from data_loader import RAGSample, RetrievalTrainDataset
from chunker import SemanticChunker, Chunk
from indexer import HybridIndexer


def find_positive_chunks(evidence: str, chunks: List[Chunk], threshold: float = 0.5) -> List[Chunk]:
    """
    找出包含 evidence 的 child chunks（字串重疊判斷）
    """
    ev_token_sets = [set(ev.lower().strip().split()) for ev in evidence if ev.strip()]

    if not ev_token_sets:
        return []

    positives = []

    for chunk in chunks:
        if chunk.level != 2:
            continue

        ch_tokens = set(chunk.text.lower().strip().split())

        best_overlap = 0.0
        for ev_tokens in ev_token_sets:
            if not ev_tokens:
                continue
            overlap = len(ev_tokens & ch_tokens) / len(ev_tokens)
            if overlap > best_overlap:
                best_overlap = overlap

        if best_overlap >= threshold:
            positives.append((chunk, best_overlap))

    positives.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in positives]


def mine_hard_negatives(
    query: str,
    positive_chunks: List[Chunk],
    indexer: HybridIndexer,
    k: int = 10
) -> List[Chunk]:
    """用 BM25 挖掘 hard negatives（BM25 排名高但非 positive 的 chunks）"""
    tokenized_query = query.lower().split()
    scores = indexer.bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:k * 3]

    positive_ids = {c.chunk_id for c in positive_chunks}
    hard_negatives = []

    for idx in top_indices:
        chunk = indexer.child_chunks[idx]
        if chunk.chunk_id not in positive_ids:
            hard_negatives.append(chunk)
        if len(hard_negatives) >= k:
            break

    return hard_negatives


def build_training_examples(
    samples: List[RAGSample],
    chunker: SemanticChunker,
    config: RAGConfig
) -> Tuple[List[InputExample], Dict]:
    """
    為每筆資料建立訓練樣本
    回傳 InputExample list（供 MNRL loss 使用）
    """
    all_examples = []
    eval_queries = {}
    eval_corpus = {}
    eval_relevant = {}

    print("[Trainer] Building chunks and mining negatives...")

    # 先對所有文件做 chunking + 建索引（用於 hard negative mining）
    all_chunks = []
    doc_chunk_map: Dict[str, List[Chunk]] = {}

    for i, sample in enumerate(tqdm(samples, desc="Chunking")):
        doc_id = f"doc_{i}"
        chunks = chunker.chunk_document(doc_id, sample.full_text)
        all_chunks.extend(chunks)
        doc_chunk_map[doc_id] = chunks

    # 建立臨時索引用於 hard negative mining
    temp_indexer = HybridIndexer(chunker.model, config)
    temp_indexer.build(all_chunks)

    for i, sample in enumerate(tqdm(samples, desc="Building examples")):
        doc_id = f"doc_{i}"
        doc_chunks = doc_chunk_map[doc_id]
        query = sample.question

        # 找 positive chunks
        positives = find_positive_chunks(sample.evidence, doc_chunks)
        if not positives:
            continue  # 跳過找不到 positive 的樣本

        # 取最佳 positive
        best_positive = positives[0]

        # 挖掘 hard negatives
        hard_negs = mine_hard_negatives(query, positives, temp_indexer, k=config.hard_negative_k)

        # 建立 InputExample（MNRL：query, positive）
        # MNRL 會自動把 batch 內其他 positive 當 in-batch negative
        all_examples.append(InputExample(
            texts=[query, best_positive.text]
        ))

        # 額外加入 hard negative triplets（用 TripletLoss 格式）
        for neg in hard_negs[:2]:
            all_examples.append(InputExample(
                texts=[query, best_positive.text, neg.text]
            ))

        # 建立評估資料
        qid = f"q_{i}"
        eval_queries[qid] = query
        for chunk in doc_chunks:
            if chunk.level == 2:
                eval_corpus[chunk.chunk_id] = chunk.text
        for pos in positives:
            eval_relevant.setdefault(qid, set()).add(pos.chunk_id)

    print(f"[Trainer] Total training examples: {len(all_examples)}")
    return all_examples, {"queries": eval_queries, "corpus": eval_corpus, "relevant": eval_relevant}


def train(config: RAGConfig, samples: List[RAGSample]):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # 載入 base model
    print(f"[Trainer] Loading base model: {config.embed_model_name}")
    model = SentenceTransformer(config.embed_model_name, device="cuda")
    model.max_seq_length = config.max_seq_length

    # 建立 chunker
    chunker = SemanticChunker(model, config)

    # 分割訓練/驗證集
    split = int(len(samples) * 1) # 100% train
    train_samples = samples[:split]
    eval_samples = samples[split:]

    # 建立訓練樣本
    train_examples, eval_data = build_training_examples(train_samples, chunker, config)

    # DataLoader
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=config.train_batch_size
    )

    # 損失函數：MNRL（適合 retrieval，不需要明確負樣本標籤）
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # 評估器
    evaluator = InformationRetrievalEvaluator(
        queries=eval_data["queries"],
        corpus=eval_data["corpus"],
        relevant_docs=eval_data["relevant"],
        name="rag-eval",
        score_functions={"cos_sim": lambda a, b: torch.mm(a, b.T)},
    )

    # 訓練
    warmup_steps = int(len(train_dataloader) * config.num_epochs * config.warmup_ratio)
    os.makedirs(config.output_dir, exist_ok=True)

    print(f"[Trainer] Start training for {config.num_epochs} epochs...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=config.num_epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": config.learning_rate},
        output_path=config.output_dir,
        save_best_model=True,
        show_progress_bar=True,
        evaluation_steps=len(train_dataloader) // 2,  # 每半個 epoch 評估一次
    )
    del chunker, train_dataloader, train_loss, evaluator
    gc.collect()
    torch.cuda.empty_cache()

    print(f"[Trainer] Training complete. Best model saved to {config.output_dir}")
    return model