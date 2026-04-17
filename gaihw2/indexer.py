import pickle
import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from chunker import Chunk


class HybridIndexer:
    def __init__(self, embed_model: SentenceTransformer, config):
        self.model = embed_model
        self.config = config

        # 儲存結構
        self.child_chunks: List[Chunk] = []          # 所有 child chunks
        self.parent_chunks: Dict[str, Chunk] = {}    # parent_id → Chunk
        self.chunk_id_to_idx: Dict[str, int] = {}    # chunk_id → list index

        # 索引
        self.faiss_index: Optional[faiss.Index] = None
        self.bm25: Optional[BM25Okapi] = None

    def build(self, all_chunks: List[Chunk]):
        """從所有 chunks 建立索引"""
        # 分離 parent / child
        for chunk in all_chunks:
            if chunk.level == 1:
                self.parent_chunks[chunk.chunk_id] = chunk
            elif chunk.level == 2:
                idx = len(self.child_chunks)
                self.child_chunks.append(chunk)
                self.chunk_id_to_idx[chunk.chunk_id] = idx

        print(f"[Indexer] {len(self.child_chunks)} child chunks, "
              f"{len(self.parent_chunks)} parent chunks")

        self._build_vector_index()
        self._build_bm25_index()

    def _build_vector_index(self):
        texts = [c.text for c in self.child_chunks]
        print("[Indexer] Encoding child chunks...")
        embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype(np.float32)

        dim = embeddings.shape[1]
        # 使用 IVF + PQ 壓縮（大規模時使用），小資料集用 Flat
        if len(self.child_chunks) > 10000:
            nlist = min(256, len(self.child_chunks) // 10)
            quantizer = faiss.IndexFlatIP(dim)
            self.faiss_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            self.faiss_index.train(embeddings)
        else:
            self.faiss_index = faiss.IndexFlatIP(dim)

        self.faiss_index.add(embeddings)
        print(f"[Indexer] FAISS index built: {self.faiss_index.ntotal} vectors")

    def _build_bm25_index(self):
        tokenized = [c.text.lower().split() for c in self.child_chunks]
        self.bm25 = BM25Okapi(
            tokenized,
            k1=self.config.bm25_k1,
            b=self.config.bm25_b
        )
        print("[Indexer] BM25 index built")

    def get_parent_text(self, child_chunk: Chunk) -> str:
        """Hierarchical：child → parent，回傳更大的上下文"""
        parent = self.parent_chunks.get(child_chunk.parent_id)
        return parent.text if parent else child_chunk.text

    def save(self, path: str):
        faiss.write_index(self.faiss_index, f"{path}/faiss.index")
        with open(f"{path}/indexer_data.pkl", "wb") as f:
            pickle.dump({
                "child_chunks": self.child_chunks,
                "parent_chunks": self.parent_chunks,
                "chunk_id_to_idx": self.chunk_id_to_idx,
                "bm25": self.bm25,
            }, f)

    def load(self, path: str):
        self.faiss_index = faiss.read_index(f"{path}/faiss.index")
        with open(f"{path}/indexer_data.pkl", "rb") as f:
            data = pickle.load(f)
        self.child_chunks = data["child_chunks"]
        self.parent_chunks = data["parent_chunks"]
        self.chunk_id_to_idx = data["chunk_id_to_idx"]
        self.bm25 = data["bm25"]