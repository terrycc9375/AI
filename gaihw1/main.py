from huggingface_hub import login
from unsloth import FastLanguageModel
import torch

class Prompt:
    def __init__(self):
        self.sys_header = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        self.user_header = "<|start_header_id|>user<|end_header_id|>\n\n"
        self.assistant_header = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        self.eot = "<|eot_id|>"

        self.system_message = "You are a medical expert in Pathology. Please select the correct answer for the following multiple-choice question."

    def build_question_block(self, question, opa, opb, opc, opd):
        return (
            f"Question: {question}\n"
            f"Options:\n"
            f"(A) {opa}\n"
            f"(B) {opb}\n"
            f"(C) {opc}\n"
            f"(D) {opd}\n\n"
            f"Answer with the option letter (A, B, C, or D) only."
        )

    def get_zero_shot_prompt(self, item, include_answer=False):
        """Zero-Shot 模板 (常用於推論)"""
        mapping = {0: "A", 1: "B", 2: "C", 3: "D"}
        
        # 組合 User 部分
        user_content = self.build_question_block(
            item['question'], item['opa'], item['opb'], item['opc'], item['opd']
        )
        
        prompt = (
            f"{self.sys_header}{self.system_message}{self.eot}"
            f"{self.user_header}{user_content}{self.eot}"
            f"{self.assistant_header}"
        )
        
        if include_answer:
            ans_letter = mapping.get(item['ans'])
            prompt += f"The correct answer is ({ans_letter}).{self.eot}"
            
        return prompt
    
    def get_cot_prompt(self, item, include_answer=False):
        """Chain-of-Thought 模板 (引導模型進行邏輯推理)"""
        user_content = self.build_question_block(
            item['question'], item['opa'], item['opb'], item['opc'], item['opd']
        )
        # 改變結尾指令
        user_content = user_content.replace("Answer with the option letter (A, B, C, or D) only.", 
                                            "Let's think step by step to find the correct answer.")
        
        prompt = (
            f"{self.sys_header}{self.system_message}{self.eot}"
            f"{self.user_header}{user_content}{self.eot}"
            f"{self.assistant_header}"
        )
        
        if include_answer:
            mapping = {0: "A", 1: "B", 2: "C", 3: "D"}
            ans_letter = mapping.get(item['ans'])
            # 這裡在訓練時可以放入解釋，但若 dataset.csv 沒提供解釋，通常只放答案
            prompt += f"Reasoning: ... Therefore, the correct answer is ({ans_letter}).{self.eot}"
            
        return prompt

def main():
    token = input("Input token: ")
    login(token)

    model_name = "unsloth/Llama-3.2-1B-Instruct"
    max_seq_length = 2048
    dtype = None
    load_in_4bit = True

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # LoRA Rank 
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    print("模型已成功載入並完成 LoRA 配置！")

if __name__ == "__main__":
    main()
