import os
import sys
import warnings

os.environ["UNSLOTH_SKIP_TORCHAO"] = "1" 
os.environ["ACCELERATE_USE_CPU"] = "0"

warnings.filterwarnings("ignore", category=FutureWarning)

from main import Prompt
import torch

if not hasattr(torch, "int1"):
    torch.int1 = torch.int8 # type: ignore

from collections import Counter
from unsloth import FastLanguageModel
from datasets import Dataset
import pandas as pd
import tqdm


def main():
    model_configs = ["saved_models/A", "saved_models/B", "saved_models/C"]
    input_file = "dataset/benchmark.csv"
    output_file = "3_ensemble_3.csv"
    prompt_builder = Prompt()
    prompt_method = [
        prompt_builder.get_zero_shot_prompt,
        prompt_builder.get_few_shot_prompt,
        prompt_builder.get_cot_prompt,
    ]
    df = pd.read_csv(input_file)
    dataset = Dataset.from_pandas(df)
    # answer to write to csv
    results = []
    all_votes = {item['question_id']: [] for _, item in df.iterrows()}
    
    for index in range(3):
        model_path = model_configs[index]
        method = prompt_method[index]
        
        model, tokenizer = FastLanguageModel.from_pretrained(
			model_name = model_path,
			max_seq_length = 2048,
			load_in_4bit = True,
		)
        model.forward = model.base_model.forward 
        if hasattr(model, "generate"):
            model.generate = model.base_model.generate
            
        for _, item in tqdm.tqdm(df.iterrows(), total=len(df)):
            prompt = method(item, include_answer=False)
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=15,
                    max_length=None,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=False,
                    do_sample=False,
                )
            
            result = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip().upper()
            
            pred = "A"
            found = False
            for char in ["A", "B", "C", "D"]:
                if result.startswith(char) or f"({char})" in result:
                    pred = char
                    found = True
                    break
            if not found:
                print("Cannot resolve output")
                return
            
            all_votes[item['question_id']].append(pred)
        
        del model
        del tokenizer
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        
    mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
    for question_id, votes in all_votes.items():
        vote_counts = Counter(votes)
        final_pred = vote_counts.most_common(1)[0][0]
        pred_idx = mapping.get(final_pred, 0)
        results.append({
            "question_id": question_id,
            "pred": pred_idx
        })
        
    # for i in tqdm.tqdm(range(len(dataset))):
    #     item = dataset[i]
        
    #     votes = []
    #     for method in prompt_method:
    #         prompt = method(item, include_answer=False)
    #         inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            
    #         with torch.no_grad():
    #             outputs = model.generate(
    #                 **inputs, 
    #                 max_new_tokens=10,
    #                 max_length=None,
    #                 pad_token_id=tokenizer.pad_token_id,
    #                 use_cache=False,
    #                 do_sample=False,
    #             )
            
    #         result = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip().upper()
    #         pred = "N"
    #         found = False
    #         for char in ["A", "B", "C", "D"]:
    #             if result.startswith(char) or f"({char})" in result:
    #                 pred = char
    #                 found = True
    #                 break
    #         if not found:
    #             print("Output error, not found")
    #             exit(1)
    #         votes.append(pred)
        
    #     vote_counts = Counter(votes)
    #     final_pred = vote_counts.most_common(1)[0][0]
    #     pred_idx = {"A": 0, "B": 1, "C": 2, "D": 3}.get(final_pred, 0)
    #     results.append({
    #         "question_id": item['question_id'],
    #         "pred": pred_idx
    #     })

    output_df = pd.DataFrame(results)
    output_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    main()