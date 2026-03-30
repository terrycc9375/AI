import os
import sys
import warnings

os.environ["UNSLOTH_SKIP_TORCHAO"] = "1" 
os.environ["ACCELERATE_USE_CPU"] = "0"

warnings.filterwarnings("ignore", category=FutureWarning)

from huggingface_hub import login
from unsloth import FastLanguageModel
import torch
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

        # self.system_message = "You are a medical expert in Pathology. Please select the correct answer for the following multiple-choice question."
        # self.system_message = "You are an expert Medical Examiner specializing in Pathology. Evaluate the clinical presentation and histological findings provided. Select the single best option (A, B, C, or D) that represents the most likely diagnosis or underlying pathological mechanism."
        # self.system_message = "Expert Pathologist. Objective: Solve medical multiple-choice questions with high precision. Format: Provide only the letter of the correct option."
        self.system_message = """You are an elite Pathologist board-certified in Histopathology and Clinical Diagnostics. 
Your task is to provide the most precise diagnosis for the following medical case. 
Use your knowledge of cellular morphology, pathophysiology, and clinical presentation.
Select the correct answer from the provided options (A, B, C, D)."""
        
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
        mapping = {0: "A", 1: "B", 2: "C", 3: "D"}
        user_content = self.build_question_block(item['question'], item['opa'], item['opb'], item['opc'], item['opd'])
        user_content += "Provide the letter (A, B, C, or D) of the correct answer."
        
        prompt = f"{self.sys_header}{self.system_message}{self.eot}{self.user_header}{user_content}{self.eot}{self.assistant_header}"
        
        if include_answer:
            ans_letter = mapping.get(item['ans'])
            prompt += f"\nAnswer: ({ans_letter}){self.eot}"
        else:
            prompt += "\nAnswer: ("
        return prompt

    def get_few_shot_prompt(self, item, include_answer=False):
        mapping = {0: "A", 1: "B", 2: "C", 3: "D"}
        
        few_shot_example = (
            "Example Question: What is the primary characteristic of acute inflammation?\n"
            "Options:\n(A) Fibrosis\n(B) Granuloma formation\n(C) Neutrophil infiltration\n(D) Angiogenesis\n"
            "Answer: (C)\n\n"
        )

        user_content = self.build_question_block(
            item['question'], item['opa'], item['opb'], item['opc'], item['opd']
        )
        
        prompt = (
            f"{self.sys_header}{self.system_message}{self.eot}"
            f"{self.user_header}{few_shot_example}{user_content}{self.eot}"
            f"{self.assistant_header}"
        )
        
        if include_answer:
            ans_letter = mapping.get(item['ans'])
            prompt += f"Rationale: Based on pathological principles, the correct option is selected.\nAnswer: ({ans_letter}){self.eot}"
        else:
            prompt += "Rationale: Let's think step by step.\nAnswer: ("
            
        return prompt
    
    def get_cot_prompt(self, item, include_answer=False):
        mapping = {0: "A", 1: "B", 2: "C", 3: "D"}
        user_content = self.build_question_block(item['question'], item['opa'], item['opb'], item['opc'], item['opd'])
        user_content += "Let's think step by step to find the correct answer."
        
        prompt = f"{self.sys_header}{self.system_message}{self.eot}{self.user_header}{user_content}{self.eot}{self.assistant_header}"
        
        if include_answer:
            ans_letter = mapping.get(item['ans'])
            prompt += f"\nReasoning: Step 1: Analyze tissue; Step 2: Confirm symptoms. Therefore, the answer is ({ans_letter}).{self.eot}"
        else:
            prompt += "\nReasoning: Let's analyze the histological features."
            prompt += "\nAnswer: ("
        return prompt

def main():
    token = input("Input token: ")
    login(token)

    model_name = "unsloth/Llama-3.2-1B-Instruct"
    max_seq_length = 2048
    dtype = None
    load_in_4bit = True

    data_frame = pd.read_csv("dataset/dataset.csv")
    prompt_builder = Prompt()
    
    def format_zero_shot_prompts(examples):
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
    
    def format_few_shot_prompts(examples):
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
            text = prompt_builder.get_few_shot_prompt(item, include_answer=True)
            output_texts.append(text)
        return { "text" : output_texts }
    
    def format_cot_prompts(examples):
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
            text = prompt_builder.get_cot_prompt(item, include_answer=True)
            output_texts.append(text)
        return { "text" : output_texts }
    
    dataset = Dataset.from_pandas(data_frame)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    # zero-shot model
    train_dataset = dataset["train"].map(format_zero_shot_prompts, batched = True)
    test_dataset = dataset["test"].map(format_zero_shot_prompts, batched = True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 32, # LoRA Rank 
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 32,
        lora_dropout = 0.05,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    training_args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 2,
        warmup_steps = 5,
        num_train_epochs=3,
        learning_rate = 1e-4,
        lr_scheduler_type="cosine",
        optim = "adamw_8bit",
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_strategy="epoch",
        output_dir = "outputsA",
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

    trail = "A"
    model.save_pretrained(f"saved_models/{trail}")
    tokenizer.save_pretrained(f"saved_models/{trail}")
    
    # few-shot model
    train_dataset = dataset["train"].map(format_few_shot_prompts, batched = True)
    test_dataset = dataset["test"].map(format_few_shot_prompts, batched = True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 32,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 32,
        lora_dropout = 0.05,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )
    
    training_args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 2,
        warmup_steps = 5,
        num_train_epochs=3,
        learning_rate = 1e-4,
        lr_scheduler_type="cosine",
        optim = "adamw_8bit",
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_strategy="epoch",
        output_dir = "outputsB",
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

    trail = "B"
    model.save_pretrained(f"saved_models/{trail}")
    tokenizer.save_pretrained(f"saved_models/{trail}")
    
    # cot model
    train_dataset = dataset["train"].map(format_cot_prompts, batched = True)
    test_dataset = dataset["test"].map(format_cot_prompts, batched = True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 32,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 32,
        lora_dropout = 0.05,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )
    
    training_args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 2,
        warmup_steps = 5,
        num_train_epochs=3,
        learning_rate = 1e-4,
        lr_scheduler_type="cosine",
        optim = "adamw_8bit",
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_strategy="epoch",
        output_dir = "outputsC",
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

    trail = "C"
    model.save_pretrained(f"saved_models/{trail}")
    tokenizer.save_pretrained(f"saved_models/{trail}")
    
def gputest():
    import torch
    print(torch.cuda.is_available())

if __name__ == "__main__":
    main()
    # gputest()
