from dataclasses import dataclass, field
from typing import Optional

@dataclass
class RAGConfig:
    # 資料
    data_path: str = "public_dataset.json"

    # Chunking
    semantic_threshold: float = 0.85      # 語意相似度閾值
    parent_chunk_size: int = 512          # 父層 chunk token 數
    child_chunk_size: int = 128           # 子層 chunk token 數
    chunk_overlap: int = 32

    # 向量模型（用來訓練的 bi-encoder）
    embed_model_name: str = "BAAI/bge-large-en-v1.5" # "mixedbread-ai/mxbai-embed-large-v1"
    embed_dim: int = 768
    max_seq_length: int = 512

    # BM25
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # 混合搜尋權重
    vector_weight: float = 0.6
    bm25_weight: float = 0.4
    top_k_retrieve: int = 20             # 混合搜尋取前 k 個
    top_k_rerank: int = 5                # rerank 後取前 k 個

    # Reranker
    # rerank_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2" # 1
    rerank_model_name: str = "mixedbread-ai/mxbai-rerank-base-v1" # 2,

    # 訓練
    train_batch_size: int = 2
    eval_batch_size: int = 4
    num_epochs: int = 5
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    margin: float = 0.3                  # Triplet loss margin
    hard_negative_k: int = 5            # 每個 query 取幾個 hard negative
    output_dir: str = "./outputs"
    seed: int = 42