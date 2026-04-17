import os
import json
from typing import List, Optional
import tqdm
from sentence_transformers import SentenceTransformer
from huggingface_hub import login

from config import RAGConfig
from data_loader import load_dataset, RAGSample, PrivateSample
from chunker import SemanticChunker, Chunk
from indexer import HybridIndexer
from retriever import HybridRetriever
from reranker import Reranker, RerankResult
from generator import LLMGenerator


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
    
    def query_with_evidence(self, question: str) -> List[str]:
        """
        執行完整 retrieval pipeline，回傳 evidence text list
        每個元素對應一個 reranked result 的 parent_text
        """
        reranked = self.query(question)
        return [r.parent_text for r in reranked]

    def run_inference(
        self,
        private_samples: List[PrivateSample],
        output_path: str = "results.json",
        llm_model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        load_in_4bit: bool = True,
    ):
        """
        完整推論流程：
        1. per-document chunking + 建立臨時索引
        2. Hybrid retrieval → Rerank → evidence list
        3. 組合 augmented prompt → LLaMA-3.2-3B-Instruct 生成答案
        4. 輸出 {"title", "answer", "evidence"} 到 results.json
        """

        # 初始化 LLM（只載入一次，所有 sample 共用）
        generator = LLMGenerator(
            model_name=llm_model_name,
            load_in_4bit=load_in_4bit,
        )

        results = []

        for i, sample in enumerate(tqdm.tqdm(private_samples, desc="Inference")):
            doc_id = f"private_doc_{i}"

            # ── Step 1: Chunking ──────────────────────────────────────
            chunks = self.chunker.chunk_document(doc_id, sample.full_text)

            # ── Step 2: 建立 per-document 臨時索引 ───────────────────
            doc_indexer = HybridIndexer(self.embed_model, self.config)
            doc_indexer.build(chunks)
            doc_retriever = HybridRetriever(self.embed_model, doc_indexer, self.config)

            # ── Step 3: Hybrid Search + Rerank ───────────────────────
            candidates = doc_retriever.retrieve(sample.question)
            reranked = self.reranker.rerank(sample.question, candidates)

            # ── Step 4: 整理 evidence list（去重，保持 rerank 順序）──
            seen = set()
            evidence_list = []
            for r in reranked:
                text = r.parent_text.strip()
                if text and text not in seen:
                    seen.add(text)
                    evidence_list.append(text)

            # ── Step 5: LLM 生成答案 ──────────────────────────────────
            answer = generator.generate(
                question=sample.question,
                evidence_list=evidence_list,
            )

            results.append({
                "title":    sample.title,
                "answer":   answer,
                "evidence": evidence_list,
            })

            # 即時寫入（防止中途崩潰遺失進度）
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)

        print(f"[Pipeline] Inference complete. {len(results)} results saved to {output_path}")
        return results

# ── 主程式入口 ──────────────────────────────────────────────

def main():
    login("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")
    config = RAGConfig()

    # 1. 載入資料
    print("[Main] Loading dataset...")
    samples = load_dataset(config.data_path)
    print(f"[Main] {len(samples)} samples loaded")

    # 2. 訓練 retriever
    trained_model_path = config.output_dir
    from trainer import train
    train(config, samples)

    # 3. 建立 pipeline
    pipeline = RAGPipeline(config)
    pipeline.load_models(embed_model_path=trained_model_path)

    from data_loader import load_private_dataset
    print("[Main] Loading private dataset...")
    private_samples = load_private_dataset("private_dataset.json")
    print(f"[Main] {len(private_samples)} private samples loaded")

    pipeline.run_inference(
        private_samples=private_samples,
        output_path="results.json",
    )    


if __name__ == "__main__":
    main()