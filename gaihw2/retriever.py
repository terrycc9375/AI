import numpy as np
from typing import List, Tuple, Dict, Optional
from sentence_transformers import SentenceTransformer
from indexer import HybridIndexer, Chunk
from dataclasses import dataclass


@dataclass
class RetrievalResult:
    chunk: Chunk
    parent_text: str       # 回傳 parent 層級的完整文本
    score: float
    vector_rank: int = -1
    bm25_rank: int = -1


class HybridRetriever:
    def __init__(self, embed_model: SentenceTransformer, indexer: HybridIndexer, config):
        self.model = embed_model
        self.indexer = indexer
        self.config = config

    def _vector_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """回傳 (chunk_idx, score) list"""
        q_emb = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype(np.float32)

        scores, indices = self.indexer.faiss_index.search(q_emb, top_k)
        return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0]) if idx >= 0]

    def _bm25_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """回傳 (chunk_idx, score) list"""
        tokenized_query = query.lower().split()
        scores = self.indexer.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Tuple[int, float]],
        bm25_results: List[Tuple[int, float]],
        k: int = 60
    ) -> List[Tuple[int, float]]:
        """
        RRF score = Σ 1 / (k + rank_i)
        比直接加權分數更穩健，不受分數量綱影響
        """
        rrf_scores: Dict[int, float] = {}
        rank_info: Dict[int, dict] = {}

        for rank, (idx, _) in enumerate(vector_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank + 1)
            rank_info.setdefault(idx, {})["vector_rank"] = rank

        for rank, (idx, _) in enumerate(bm25_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank + 1)
            rank_info.setdefault(idx, {})["bm25_rank"] = rank

        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results, rank_info

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        top_k = top_k or self.config.top_k_retrieve

        vector_results = self._vector_search(query, top_k * 2)
        bm25_results = self._bm25_search(query, top_k * 2)

        fused, rank_info = self._reciprocal_rank_fusion(vector_results, bm25_results)

        results = []
        for idx, score in fused[:top_k]:
            chunk = self.indexer.child_chunks[idx]
            parent_text = self.indexer.get_parent_text(chunk)
            info = rank_info.get(idx, {})
            results.append(RetrievalResult(
                chunk=chunk,
                parent_text=parent_text,
                score=score,
                vector_rank=info.get("vector_rank", -1),
                bm25_rank=info.get("bm25_rank", -1),
            ))

        return results