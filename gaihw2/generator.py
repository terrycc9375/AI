import torch
from typing import List, Optional
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
import ollama


# SYSTEM_PROMPT = """You are a precise academic research assistant.
# Answer the question using ONLY the provided evidence passages.
# Be concise and factual. If the evidence does not contain enough information, say so."""

# ANSWER_PROMPT_TEMPLATE = """Evidence passages:
# {evidence_block}

# ---
# Question: {question}

# Answer:"""




class LLMGenerator:
    def __init__(
        self,
        model_name: str = "llama3.2:3b",
        device: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[Generator] Initializing Ollama with {model_name}...")
        self.client = ollama.Client()

    def _build_prompt(self, question: str, evidence_list: List[str]) -> str:
        """
        組合 augmented prompt
        - 每段 evidence 加上編號，方便模型定位
        - 使用 LLaMA-3 的 chat template 格式
        """
        # SYSTEM_PROMPT = """You are an expert Research Integrity Officer. 
        # Your task is to synthesize a precise answer using ONLY the provided Research Evidence. Do not use outside knowledge or make assumptions.

        # STRICT RULES:
        # 1. Use ONLY the provided evidence. Do not invent or infer information not in the passages.
        # 2. Direct Quote or Paraphrase: If the information is in the evidence, extract it accurately. Always cite the passage number (e.g., [Passage 1]).
        # 3. Synthesis: If multiple passages discuss the same point, combine them concisely.
        # 4. Uncertainty: If the evidence is insufficient or contradictory, state exactly: "The provided context does not contain enough information."
        # 5. Conciseness: Use academic language but keep it direct. Do not add unrelated details.
        # 6. Verification: After answering, list the passages used (e.g., "Based on [Passage 1] and [Passage 3]")."""

        # ANSWER_PROMPT_TEMPLATE = """You are analyzing a research paper.

        # [Research Evidence]
        # {evidence_block}

        # [Question]: {question}

        # [Task]
        # Based on the [Research Evidence] above, answer the [Question]. Extract specific datasets, methodologies, or experimental results directly from the passages. Do not expand beyond the evidence.

        # [Examples]
        # - Correct: Question: "What datasets were used?" Evidence includes [Passage 1] "GENIA and CoNLL2003". Answer: "The datasets used are GENIA and CoNLL2003. Based on [Passage 1]."
        # - Incorrect: Do not say "LUSDB" if not in evidence. Instead: "The provided context does not contain enough information."

        # Answer:"""
        
        # SYSTEM_PROMPT = """You are an expert Research Integrity Officer. Use ONLY the provided evidence. No outside knowledge.

        # STRICT RULES:
        # 1. Extract directly from passages. Cite numbers (e.g., [Passage 1]).
        # 2. If insufficient, say: "The provided context does not contain enough information."
        # 3. Be concise and academic."""

        # ANSWER_PROMPT_TEMPLATE = """[Research Evidence]
        # {evidence_block}

        # [Question]: {question}

        # [Task]
        # Answer step-by-step:
        # 1. Identify relevant passages.
        # 2. Extract key info (datasets, etc.).
        # 3. Synthesize answer with citations.

        # Examples:
        # - For "datasets used": "GENIA and CoNLL2003. Based on [Passage 2] and [Passage 5]."

        # Answer:"""

        SYSTEM_PROMPT = """You are a precise Information Extraction specialist.
        Your goal is to extract ALL relevant details requested from the provided context.

        STRICT RULES:
        1. EXHAUSTIVE EXTRACTION: Capture every mentioned dataset, metric, or entity found in the evidence.
        2. SOURCE CITATION: You MUST cite the source numbers (e.g., [Passage 1]).
        3. ZERO OUTSIDE KNOWLEDGE: Use ONLY the provided evidence.
        4. CONCISION: Provide only the facts. No conversational fillers or meta-talk."""

        ANSWER_PROMPT_TEMPLATE = """[Research Evidence]
        {evidence_block}

        [Target Question]: {question}

        [Task]
        Synthesize a comprehensive but concise answer based on the evidence. 
        If the question asks for items (like datasets or methods), list ALL of them that appear in the passages.

        Answer Format Example:
        "The study uses [A] (Passage 1), [B] (Passage 3), and [C] (Passage 5)."

        Answer:"""

        evidence_block = "\n\n".join(
            f"[Passage {i+1}] {ev.strip()}"
            for i, ev in enumerate(evidence_list)
            if ev.strip()
        )

        user_content = ANSWER_PROMPT_TEMPLATE.format(
            evidence_block=evidence_block,
            question=question,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]

        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            prompt_parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>")

        prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        prompt = "".join(prompt_parts)
        return prompt

    def generate(self, question: str, evidence_list: List[str]) -> str:
        """
        輸入 question + evidence list，回傳 LLM 生成的答案字串
        """
        if not evidence_list:
            return "No evidence retrieved."

        prompt = self._build_prompt(question, evidence_list)

        # 使用 Ollama 生成回應
        response = self.client.generate(
            model=self.model_name,
            prompt=prompt,
            options={
                "num_predict": self.max_new_tokens,  # 對應 max_new_tokens
                "temperature": self.temperature,     # 對應 temperature
                "repetition_penalty": 1.1,           # 保留 repetition_penalty（Ollama 支援）
            }
        )

        # Ollama 的回應是字典，直接取出 'response' 鍵
        answer = response['response'].strip()
        return answer
    
    def generate_hypothetical(self, question: str, title: str) -> str:
        """
        生成假設答案 (HyDE 方法) - 優化版本
        - 使用問題，讓 LLM 生成一個精確、學術性的假設回應
        - 聚焦於技術細節和關鍵術語，提升檢索相關性
        """
        hypothetical_prompt = f"""
    <|start_header_id|>system<|end_header_id|>

    You are an expert in scientific research. Your task is to generate a "Hypothetical Answer" based on the question, as if you are explaining it from a research paper. Follow these guidelines:
    1. Be factual, detailed, and concise: Use academic language, key terms, and technical phrases (e.g., "lexical overlaps", "SERA formula").
    2. Structure the answer logically: Include definitions, comparisons, or workflows relevant to the topic.
    3. Avoid generic responses: Focus on plausible, high-recall details that would appear in evidence chunks.

    Paper title: {title}
    Question: {question}

    <|eot_id|><|start_header_id|>user<|end_header_id|>

    Generate the Hypothetical Answer:
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>

    Hypothetical Answer:"""

        messages = [
            {"role": "user", "content": hypothetical_prompt},
        ]
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            prompt_parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>")

        prompt = "".join(prompt_parts)

        response = self.client.generate(
            model=self.model_name,
            prompt=prompt,
            options={
                "num_predict": 128,  # 保持長度，適合詳細答案
                "temperature": 0.5,  # 降低隨機性，提升精確性
                "repetition_penalty": 1.5,  # 減少重複廢話
            }
        )

        hypothetical_answer = response['response'].strip()
        # 後處理：移除多餘空行或縮短過長部分（e.g., 限制到 100 個詞）
        hypothetical_answer = ' '.join(hypothetical_answer.split()[:100])  # 粗估詞數限制
        return hypothetical_answer

    def generate_augmented_question(self, question: str, title: str) -> str:
        """
        生成強化版學術性問題（Augmented Academic Question 方法）
        - 使用問題和論文標題，讓 LLM 生成一個更精確、學術性的強化版本
        - 這將用來增強查詢，例如聚焦於特定領域或文件內容
        """
        augmented_prompt = f"""
<|start_header_id|>system<|end_header_id|>

You are an expert research analyst. Your goal is to transform a user query into a "High-Recall Search Query" specifically for scientific paper retrieval.

Follow these steps to generate the Augmented Question:
1. Identify potential keywords related to the "Paper Title" (e.g., specific architectures like 'Transformer', 'CVT', or datasets).
2. Expand the "Question" by anticipating academic phrases used to describe the answer (e.g., "how they use" -> "technical implementation and algorithmic workflow").
3. Ensure the result is a dense, factual statement that reflects how scientists write.

Paper Title: {title}
Original Question: {question}

<|eot_id|><|start_header_id|>user<|end_header_id|>

Based on the title and question above, generate an augmented search query:
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Augmented Search Query:"""

        messages = [
            {"role": "user", "content": augmented_prompt},
        ]
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            prompt_parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>")

        prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        prompt = "".join(prompt_parts)

        response = self.client.generate(
            model=self.model_name,
            prompt=prompt,
            options={
                "num_predict": 80,  # 問題可能稍長於假設答案
                "temperature": self.temperature,
                "repetition_penalty": 1.5,
            }
        )

        augmented_question = response['response'].strip()
        return augmented_question

    