import os
from typing import List, Optional
from sentence_transformers import SentenceTransformer

from config import RAGConfig
from data_loader import load_dataset, RAGSample
from chunker import SemanticChunker, Chunk
from indexer import HybridIndexer
from retriever import HybridRetriever
from reranker import Reranker, RerankResult


class RAGPipeline:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.embed_model: Optional[SentenceTransformer] = None
        self.chunker: Optional[SemanticChunker] = None
        self.indexer: Optional[HybridIndexer] = None
        self.retriever: Optional[HybridRetriever] = None
        self.reranker: Optional[Reranker] = None

    def load_models(self, embed_model_path: Optional[str] = None):
        """載入 embedding model（可指定訓練後的 checkpoint）"""
        path = embed_model_path or self.config.embed_model_name
        print(f"[Pipeline] Loading embed model from: {path}")
        self.embed_model = SentenceTransformer(path)
        self.embed_model.max_seq_length = self.config.max_seq_length

        self.chunker = SemanticChunker(self.embed_model, self.config)
        self.reranker = Reranker(self.config)

    def build_index(self, samples: List[RAGSample], index_save_path: Optional[str] = None):
        """對資料集建立索引"""
        all_chunks: List[Chunk] = []

        print("[Pipeline] Chunking documents...")
        for i, sample in enumerate(samples):
            doc_id = f"doc_{i}_{sample.title[:20].replace(' ', '_')}"
            chunks = self.chunker.chunk_document(doc_id, sample.full_text)
            all_chunks.extend(chunks)

        self.indexer = HybridIndexer(self.embed_model, self.config)
        self.indexer.build(all_chunks)
        self.retriever = HybridRetriever(self.embed_model, self.indexer, self.config)

        if index_save_path:
            os.makedirs(index_save_path, exist_ok=True)
            self.indexer.save(index_save_path)
            print(f"[Pipeline] Index saved to {index_save_path}")

    def load_index(self, index_load_path: str):
        """載入已存的索引"""
        self.indexer = HybridIndexer(self.embed_model, self.config)
        self.indexer.load(index_load_path)
        self.retriever = HybridRetriever(self.embed_model, self.indexer, self.config)
        print(f"[Pipeline] Index loaded from {index_load_path}")

    def query(self, question: str) -> List[RerankResult]:
        """完整查詢流程：混合搜尋 → Rerank"""
        assert self.retriever and self.reranker, "請先呼叫 load_models() 和 build_index()"

        # Step 1: 混合搜尋
        candidates = self.retriever.retrieve(question, top_k=self.config.top_k_retrieve)

        # Step 2: Rerank
        final_results = self.reranker.rerank(question, candidates)

        return final_results


# ── 主程式入口 ──────────────────────────────────────────────

def main():
    config = RAGConfig()

    # 1. 載入資料
    print("[Main] Loading dataset...")
    samples = load_dataset(config.data_path)
    print(f"[Main] {len(samples)} samples loaded")

    # 2. 訓練 retriever（如果還沒訓練）
    trained_model_path = config.output_dir
    from trainer import train
    train(config, samples)

    # 3. 建立 pipeline
    pipeline = RAGPipeline(config)
    pipeline.load_models(embed_model_path=trained_model_path)
    pipeline.build_index(samples, index_save_path="./index_store")

    # 4. 測試查詢
    test_sample = samples[0]
    print(f"\n[Test] Question: {test_sample.question}")
    print(f"[Test] Ground truth evidence: {test_sample.evidence[:200]}...\n")

    results = pipeline.query(test_sample.question)

    print(f"[Test] Top {len(results)} retrieved results:")
    for i, r in enumerate(results):
        print(f"\n--- Result {i+1} (rerank_score={r.rerank_score:.4f}) ---")
        print(f"Chunk ID: {r.chunk_id}")
        print(f"Text: {r.text[:300]}...")


if __name__ == "__main__":
    main()