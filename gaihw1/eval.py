from main import Prompt
import os
import torch
# 解決之前遇到的 AttributeError: int1 問題
if not hasattr(torch, "int1"):
    torch.int1 = torch.int8

from unsloth import FastLanguageModel
from datasets import Dataset
import pandas as pd
from main import Prompt # 假設你的 Prompt 類別在 main.py


def evaluate_accuracy(model, tokenizer, prompt_builder, dataset):
    correct = 0
    model.eval()
    print(f"開始評估 {len(dataset)} 筆資料...")
    
    for i in range(len(dataset)):
        item = dataset[i]
        # 使用你定義的 Prompt 邏輯
        prompt = prompt_builder.get_zero_shot_prompt(item, include_answer=False)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            # 加上 tokenizer.pad_token_id 確保生成穩定
            outputs = model.generate(
                **inputs, 
                max_new_tokens=10,
                max_length=None,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=False,
                do_sample=False,
            )
        
        result = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        
        mapping = {0: "A", 1: "B", 2: "C", 3: "D"}
        correct_ans = mapping[int(item['ans'])]
        
        if f"({correct_ans})" in result or result.strip().startswith(correct_ans):
            correct += 1
            
        if (i + 1) % 50 == 0:
            print(f"已完成: {i + 1}/{len(dataset)}")
            
    print(f"--- 評估結果 ---")
    print(f"Validation Accuracy: {correct / len(dataset) * 100:.2f}%")

def main():
    model_path = "saved_models/01" 
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = 2048,
        load_in_4bit = True,
    )
    model.forward = model.base_model.forward 
    if hasattr(model, "generate"):
        model.generate = model.base_model.generate

    prompt_builder = Prompt()
    df = pd.read_csv("dataset.csv")
    dataset = Dataset.from_pandas(df)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    test_dataset = dataset["test"]

    evaluate_accuracy(model, tokenizer, prompt_builder, test_dataset)

if __name__ == "__main__":
    main()
