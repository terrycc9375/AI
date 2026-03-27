from huggingface_hub import login
from unsloth import FastLanguageModel
import torch
import os
import pandas as pd
from datasets import Dataset
from transformers import TrainingArguments
from trl.trainer.sft_trainer import SFTTrainer

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
    
def evaluate_accuracy(model, tokenizer, dataset):
    correct = 0
    model.eval()
    for item in dataset:
        prompt = Prompt.get_zero_shot_prompt(item, include_answer=False)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        outputs = model.generate(**inputs, max_new_tokens=10)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        mapping = {0: "A", 1: "B", 2: "C", 3: "D"}
        correct_ans = mapping[item['ans']]
        
        if f"({correct_ans})" in result or f"Answer: {correct_ans}" in result:
            correct += 1
            
    print(f"Validation Accuracy: {correct / len(dataset) * 100:.2f}%")

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

    data_frame = pd.read_csv("dataset.csv")
    prompt_builder = Prompt()
    
    def formatting_prompts_func(examples):
        output_texts = []
        for i in range(len(examples["question"])):
            item = {
                "question": examples["question"][i],
                "opa": examples["opa"][i],
                "opb": examples["opb"][i],
                "opc": examples["opc"][i],
                "opd": examples["opd"][i],
                "ans": examples["ans"][i]
            }
            text = prompt_builder.get_zero_shot_prompt(item, include_answer=True)
            output_texts.append(text)
        return { "text" : output_texts }
    
    dataset = Dataset.from_pandas(data_frame)
    dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = dataset["train"].map(formatting_prompts_func, batched = True)
    test_dataset = dataset["test"].map(formatting_prompts_func, batched = True)

    training_args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # max_steps = 60,
        num_train_epochs=3,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 5,
        output_dir = "outputs",
        eval_strategy= "epoch",
        save_strategy = "epoch",
        load_best_model_at_end = True,
        metric_for_best_model = "loss",
    )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset= test_dataset,
        dataset_text_field = "text",
        max_seq_length = 2048,
        args = training_args,
    )

    trainer.train()
    
    evaluate_accuracy(model, tokenizer, dataset["test"])

    if not os.path.exists("saved_models"):
        os.makedirs("saved_models")
    model.save_pretrained("saved_models/01")
    tokenizer.save_pretrained("saved_models/01")
    
def gputest():
    import torch
    print(torch.cuda.is_available())

if __name__ == "__main__":
    main()
    # gputest()
