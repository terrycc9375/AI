import json
from dataclasses import dataclass
from typing import List, Optional
from torch.utils.data import Dataset


@dataclass
class RAGSample:
    title: str
    full_text: str
    question: str
    answer: str
    evidence: str  # ground truth evidence

def load_dataset(path: str) -> List[RAGSample]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    samples = []
    for item in raw:
        samples.append(RAGSample(
            title=item.get("title", ""),
            full_text=item.get("full_text", ""),
            question=item.get("question", ""),
            answer=item.get("answer", ""),
            evidence=item.get("evidence", ""),
        ))
    return samples

@dataclass
class PrivateSample:
    title: str
    full_text: str
    question: str

def load_private_dataset(path: str) -> List[PrivateSample]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    samples = []
    for item in raw:
        samples.append(PrivateSample(
            title=item.get("title", ""),
            full_text=item.get("full_text", ""),
            question=item.get("question", ""),
        ))
    return samples


class RetrievalTrainDataset(Dataset):
    """
    每筆資料：(query, positive_chunk, negative_chunk)
    positive_chunk  = 包含 evidence 的 chunk
    negative_chunk  = hard negative（由 BM25/向量搜尋挖掘）
    """
    def __init__(self, triplets: List[dict]):
        self.triplets = triplets  # [{"query":..., "positive":..., "negative":...}]

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx]