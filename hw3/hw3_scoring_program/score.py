import argparse
import importlib.util
import json
import os
import sys

import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score)
from tqdm import trange
from transformers import AutoTokenizer, PreTrainedModel


@torch.no_grad()
def predict(model, tokenizer, texts, max_length=256, batch_size=64):
    preds = []
    for i in trange(0, len(texts), batch_size, ncols=0, desc="Predicting"):
        batch_texts = texts[i:i+batch_size]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to('cuda')
        outputs = model(**enc)
        logits = outputs["logits"]
        batch_pred = logits.argmax(dim=-1).tolist()
        preds.extend(batch_pred)
    return preds


def main():
    parser = argparse.ArgumentParser(
        description='Inference and evaluate on the test set.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input_dir',
        required=True,
        default="input",
        help=(
            'Input directory containing the `res` directory with the images '
            'and `ref` directory with the testing resources.'
        )
    )
    parser.add_argument(
        '--output_dir',
        required=True,
        default="output",
        help='Output directory where the scores will be saved.'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='checkpoint',
        help=(
            "Path to the pretrained model directory. This is relative to "
            "the ${input_dir}/res directory."
        )
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='private_test_ans.csv',
        help=(
            "Path to the dataset CSV file for inference. This is relative to "
            "the ${input_dir}/ref directory."
        )
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=256,
        help="Maximum sequence length for tokenizer"
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--verbose",
        action='store_true',
        help="Enable verbose output"
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help=(
            'Path to the configuration file in JSON format. This is relative '
            'to the ${input_dir}/ref directory. If provided, it will override '
            'the parameters set in the command line.'
        )
    )
    args = parser.parse_args()

    # ================ Set up directories and paths ===================
    # Reference directory. Contains the statistics
    ref_dir = os.path.join(args.input_dir, 'ref')
    # Results directory. Submitted predictions
    res_dir = os.path.join(args.input_dir, 'res')
    if args.verbose:
        print(f"[INFO] Reference directory: {ref_dir}.")
        print(f"[INFO] Results directory: {res_dir}.")

    # ================ Load configuration if provided =================
    if args.config is not None:
        config_path = os.path.join(ref_dir, args.config)
        if args.verbose:
            print(f"[INFO] Loading configuration from {config_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file {config_path} does "
                                    f"not exist.")
        with open(config_path, 'r') as f:
            config = json.load(f)
        for key in args.__dict__:
            if key in config:
                setattr(args, key, config[key])
                del config[key]
        if args.verbose and len(config) > 0:
            print(f"[WARNING] The following parameters were not found in the "
                  f"command line: {config.keys()}")
    else:
        if args.verbose:
            print("[INFO] No configuration file provided, using command line "
                  "arguments.")
    
    # ======================== Import model ===========================
    model_file = os.path.join(res_dir, "model.py")
    if not os.path.exists(model_file):
        raise FileNotFoundError("Model file model.py does not exist.")
    try:
        module_name = "model"
        spec = importlib.util.spec_from_file_location(module_name, model_file)
        module = importlib.util.module_from_spec(spec) # type: ignore
        sys.modules[module_name] = module
        spec.loader.exec_module(module) # type: ignore
    except Exception as e:
        if args.verbose:
            print("[ERROR] Failed to import module `model.py`. Please check "
                  "that you only use the allowed packages.")
        raise e
    try:
        SentimentClassifier = module.SentimentClassifier
    except Exception as e:
        if args.verbose:
            print("[ERROR] Failed to load `SentimentClassifier` class from "
                  "the imported module. Please check your class name is "
                  "correct.")
        raise e
    if not issubclass(SentimentClassifier, PreTrainedModel):
        raise TypeError(
            "The `SentimentClassifier` class must be a subclass of "
            "`PreTrainedModel`.")

    # ======================== Load model =============================
    checkpoint_dir = os.path.join(res_dir, args.model)
    model = SentimentClassifier.from_pretrained(checkpoint_dir).cuda()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    
    # ======================= Load dataset ============================
    dataset_path = os.path.join(ref_dir, args.dataset)
    df = pd.read_csv(dataset_path)

    # ======================== Inference ==============================
    model.eval()
    true_labels = df['label'].values
    pred_labels = []
    for i in trange(
        0, len(df), args.batch_size,
        ncols=0,
        desc="[INFO] Inferencing",
        disable=not args.verbose
    ):
        texts = df["text"].iloc[i: i + args.batch_size].tolist()
        tokens = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        ).to('cuda')
        with torch.no_grad():
            outputs = model(**tokens)
        batch_preds = outputs["logits"].argmax(dim=-1).cpu()
        pred_labels.append(batch_preds)
    pred_labels = torch.cat(pred_labels).numpy()

    # ======================== Evaluate ===============================
    output_json = {
        "accuracy": accuracy_score(true_labels, pred_labels), # type: ignore
        "f1": f1_score(true_labels, pred_labels, average="macro"), # type: ignore
        "precision": precision_score(true_labels, pred_labels, average="macro"), # type: ignore
        "recall": recall_score(true_labels, pred_labels, average="macro"), # type: ignore
    }

    # ======================== Save scores ============================
    output_file = os.path.join(args.output_dir, 'scores.json')
    if args.verbose:
        print(f"[INFO] Saving scores to: {output_file}")
        print(f"[INFO] The content of the output file: {json.dumps(output_json)}")
    with open(output_file, 'w') as f:
        json.dump(output_json, f, indent=4)


if __name__ == "__main__":
    main()
