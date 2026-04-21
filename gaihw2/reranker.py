from typing import List
from sentence_transformers import CrossEncoder
from retriever import RetrievalResult
from dataclasses import dataclass
from config import RAGConfig

@dataclass
class RerankResult:
    chunk_id: str
    doc_id: str
    text: str           # child chunk text
    parent_text: str    # parent chunk text（實際回傳給 LLM 的）
    rerank_score: float
    original_score: float


class Reranker:
    def __init__(self, config: RAGConfig):
        print(f"[Reranker] Loading {config.rerank_model_name}...")
        self.model = CrossEncoder(
            config.rerank_model_name,
            max_length=512,
            device="cuda" if self._has_cuda() else "cpu"
        )
        self.config = config

    @staticmethod
    def _has_cuda():
        import torch
        return torch.cuda.is_available()

    def rerank(self, query: str, candidates: List[RetrievalResult], top_k: int = None) -> List[RerankResult]:
        if not candidates:
            return []

        # Cross-encoder 輸入：(query, parent_text) — 用 parent 給更多上下文
        pairs = [(query, c.parent_text) for c in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)

        reranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True
        )

        top_k = top_k or self.config.top_k_rerank

        results = []
        for i, (candidate, score) in enumerate(reranked):
            if i >= 10:
                break
            if i >= top_k and score <= 0.25:  # 前 top_k = 2 個無條件保留，之後的只保留分數 > 0.2 的
                continue
            
            results.append(RerankResult(
                chunk_id=candidate.chunk.chunk_id,
                doc_id=candidate.chunk.doc_id,
                text=candidate.chunk.text,
                parent_text=candidate.parent_text,
                rerank_score=float(score),
                original_score=candidate.score,
            ))

        return results