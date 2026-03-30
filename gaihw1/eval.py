import os
import sys
import warnings

os.environ["UNSLOTH_SKIP_TORCHAO"] = "1" 
os.environ["ACCELERATE_USE_CPU"] = "0"

warnings.filterwarnings("ignore", category=FutureWarning)

from main import Prompt
import torch
import gc

if not hasattr(torch, "int1"):
    torch.int1 = torch.int8 # type: ignore

from collections import Counter
from unsloth import FastLanguageModel
from datasets import Dataset
import pandas as pd
import tqdm


def evaluate_accuracy(model, tokenizer, prompt_builder: Prompt, dataset):
    correct = 0
    model.eval()
    
    prompt_method = [
        prompt_builder.get_zero_shot_prompt,
        prompt_builder.get_few_shot_prompt,
        prompt_builder.get_cot_prompt,
    ]
    
    for i in tqdm.tqdm(range(len(dataset))):
        item = dataset[i]
        mappings = {0: "A", 1: "B", 2: "C", 3: "D"}
        correct_ans = mappings[int(item['ans'])]
        
        votes = []
        for method in prompt_method:
            prompt = method(item, include_answer=False)
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=10,
                    max_length=None,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=False,
                    do_sample=False,
                )
            
            result = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip().upper()
            pred = "N"
            for char in ["A", "B", "C", "D"]:
                if result.startswith(char) or f"({char})" in result:
                    pred = char
                    break
            votes.append(pred)
        
        vote_counts = Counter(votes)
        final_pred = vote_counts.most_common(1)[0][0]
        
        if final_pred == correct_ans:
            correct += 1
            
    print(f"--- Results ---")
    print(f"Validation Accuracy: {correct / len(dataset) * 100:.2f}%")
    

def evaluate_ensemble():
    model_configs = ["saved_models/A", "saved_models/B", "saved_models/C"]
    data_path = "dataset.csv"
    prompt_builder = Prompt()
    prompt_methods = [
        prompt_builder.get_zero_shot_prompt,
        prompt_builder.get_few_shot_prompt,
        prompt_builder.get_cot_prompt,
    ]

    df = pd.read_csv(data_path)
    dataset = Dataset.from_pandas(df)
    dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
    test_ds = dataset_split["test"]

    all_votes = {i: [] for i in range(len(test_ds))}
    ground_truth = {i: int(test_ds[i]['ans']) for i in range(len(test_ds))}
    mapping = {"A": 0, "B": 1, "C": 2, "D": 3}

    for idx in range(3):
        model_path = model_configs[idx]
        method = prompt_methods[idx]
        
        print(f"\n>>> 載入第 {idx+1}/3 個模型: {model_path}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_path,
            max_seq_length = 2048,
            load_in_4bit = True,
        )

        for i in tqdm.tqdm(range(len(test_ds))):
            item = test_ds[i]
            prompt = method(item, include_answer=False)
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=15,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=False,
                    do_sample=False,
                )
            
            result = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip().upper()
            
            pred_letter = "A" 
            found = False
            for char in ["A", "B", "C", "D"]:
                if result.startswith(char) or f"({char})" in result:
                    pred_letter = char
                    found = True
                    break
            
            if not found:
                print("Cannot resolve output")
                exit(0)
            
            all_votes[i].append(pred_letter)
        
        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()

    correct_count = 0
    print("\n---Result---")
    for i in range(len(test_ds)):
        votes = all_votes[i]
        vote_counts = Counter(votes)
        final_pred_letter = vote_counts.most_common(1)[0][0]
        final_pred_idx = mapping.get(final_pred_letter, 0)
        actual_idx = ground_truth[i]
        if final_pred_idx == actual_idx:
            correct_count += 1
            
    accuracy = (correct_count / len(test_ds)) * 100
    print(f"\nAccuracy: {accuracy:.2f}%")

def main():
    model_path = "saved_models/02" 
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = 2048,
        load_in_4bit = True,
    )
    model.forward = model.base_model.forward 
    if hasattr(model, "generate"):
        model.generate = model.base_model.generate

    prompt_builder = Prompt()
    df = pd.read_csv("dataset/dataset.csv")
    dataset = Dataset.from_pandas(df)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    test_dataset = dataset["test"]

    evaluate_accuracy(model, tokenizer, prompt_builder, test_dataset)
    
def comparison():
    df1 = pd.read_csv("output.csv")
    df2 = pd.read_csv("3_ensemble_3.csv")
    same = (df1['pred'] == df2['pred']).sum()
    total = 900
    print(f"Similarity: {same}/{total} ({same/total*100:.2f}%)")

if __name__ == "__main__":
    # main()
    evaluate_ensemble()
    # comparison()
