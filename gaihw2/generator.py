import torch
from typing import List, Optional
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
import ollama


SYSTEM_PROMPT = """You are a precise academic research assistant.
Answer the question using ONLY the provided evidence passages.
Be concise and factual. If the evidence does not contain enough information, say so."""

ANSWER_PROMPT_TEMPLATE = """Evidence passages:
{evidence_block}

---
Question: {question}

Answer:"""


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
        evidence_block = "\n\n".join(
            f"[{i+1}] {ev.strip()}"
            for i, ev in enumerate(evidence_list)
            if ev.strip()
        )

        user_content = ANSWER_PROMPT_TEMPLATE.format(
            evidence_block=evidence_block,
            question=question,
        )

        # 使用 apply_chat_template 確保符合 LLaMA-3 instruct 格式
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
    
    def generate_hypothetical(self, question: str) -> str:
        """
        生成假設答案(HyDE 方法)
        - 僅使用問題，讓 LLM 生成一個假設的回應
        - 這將用來增強查詢
        """
        hypothetical_prompt = f"""You are a helpful assistant. Based on the following question, generate a hypothetical, detailed answer as if you know the information. Be factual and concise.
        Question: {question}
        Hypothetical Answer:"""

        messages = [
            {"role": "user", "content": hypothetical_prompt},
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
                "num_predict": 32,  # may be shorter
                "temperature": self.temperature,
                "repetition_penalty": 1.1,
            }
        )

        hypothetical_answer = response['response'].strip()
        return hypothetical_answer
    