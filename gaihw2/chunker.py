from typing import List, Tuple, Optional
import re
import numpy as np
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer 
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str               # 來源文件識別
    text: str
    level: int                # 0=doc, 1=parent, 2=child
    parent_id: Optional[str] = None
    start_char: int = 0
    end_char: int = 0
    metadata: dict = field(default_factory=dict)


from typing import Optional


class SemanticChunker:
    """
    步驟：
    1. 先用規則切句子
    2. 用 embedding 計算相鄰句子相似度
    3. 相似度低於 threshold → 語意邊界 → 切段 (parent chunk)
    4. 每個 parent chunk 再切成 child chunks
    """

    def __init__(self, embed_model: SentenceTransformer, config):
        self.model = embed_model
        self.config = config

    def _split_sentences(self, text: str) -> List[str]:
        """簡易句子分割，保留學術文本結構"""
        # 先處理常見縮寫避免誤切
        text = re.sub(r'\b(Fig|fig|et al|vs|Dr|Prof|Sr|Jr|No|Vol|pp)\.\s', r'\1<DOT> ', text)
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        # 還原縮寫
        sentences = [s.replace('<DOT>', '.') for s in sentences]
        return [s.strip() for s in sentences if s.strip()]

    def _find_semantic_boundaries(self, sentences: List[str]) -> List[int]:
        """回傳語意邊界的 sentence index list"""
        if len(sentences) <= 1:
            return []

        # Batch encode
        embeddings = self.model.encode(
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
            if sim < self.config.semantic_threshold:
                boundaries.append(i + 1)

        return boundaries

    def _group_sentences_to_parent(
        self,
        sentences: List[str],
        boundaries: List[int],
        max_tokens: int
    ) -> List[List[str]]:
        """根據語意邊界 + token 上限，將句子分組為 parent chunks"""
        groups = []
        current_group = []
        current_len = 0
        boundary_set = set(boundaries)

        for i, sent in enumerate(sentences):
            token_est = len(sent.split())  # 粗估 token 數

            # 遇到語意邊界或超過長度上限 → 切斷
            if (i in boundary_set or current_len + token_est > max_tokens) and current_group:
                groups.append(current_group)
                current_group = []
                current_len = 0

            current_group.append(sent)
            current_len += token_est

        if current_group:
            groups.append(current_group)

        return groups

    def _split_parent_to_children(
        self,
        parent_sentences: List[str],
        max_tokens: int,
        overlap_tokens: int
    ) -> List[str]:
        """將 parent chunk 的句子切成帶 overlap 的 child chunks"""
        children = []
        current = []
        current_len = 0

        for sent in parent_sentences:
            token_est = len(sent.split())
            if current_len + token_est > max_tokens and current:
                children.append(" ".join(current))
                # overlap：保留最後幾個 token 的句子
                overlap_sents = []
                overlap_len = 0
                for s in reversed(current):
                    if overlap_len + len(s.split()) <= overlap_tokens:
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

        return children

    def chunk_document(self, doc_id: str, text: str) -> List[Chunk]:
        """主入口：對一篇文章做完整的 hierarchical chunking"""
        chunks = []
        sentences = self._split_sentences(text)

        if not sentences:
            return chunks

        # Step 1: 找語意邊界
        boundaries = self._find_semantic_boundaries(sentences)

        # Step 2: 組成 parent chunks
        parent_groups = self._group_sentences_to_parent(
            sentences, boundaries, self.config.parent_chunk_size
        )

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
            chunks.append(parent_chunk)

            # Step 3: 切 child chunks
            children_texts = self._split_parent_to_children(
                parent_sents,
                self.config.child_chunk_size,
                self.config.chunk_overlap
            )

            for c_idx, child_text in enumerate(children_texts):
                child_id = f"{parent_id}_c{c_idx}"
                chunks.append(Chunk(
                    chunk_id=child_id,
                    doc_id=doc_id,
                    text=child_text,
                    level=2,
                    parent_id=parent_id,
                ))

        return chunks