'''
HW3: Sentiment Analysis with Deep Learning

In this homework, we will explore the fascinating field of sentiment analysis using deep learning techniques. 
Specifically, we will focus on multi-class classification, where the goal is to predict each sentence from social media 
as belonging to the label.

Label definition:
0 -> Negative
1 -> Neutral
2 -> Positive
'''

import os
import re
import gc
import json
import random
import argparse
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
    PretrainedConfig,
    PreTrainedModel
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Reproducibility
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# FLOPs estimation
def estimate_flops(hidden_size: int, num_layers: int, seq_len: int, batch_size: int) -> float:
    '''
    Roughly estimate the number of floating point operations (FLOPs) 
    per training step for models

    Args:
        hidden_size: model embedding dimension
        num_layers: number of encoder layers
        seq_len: number of tokens per input
        batch_size: number of samples processed per step

    Returns:
        Estimated FLOPs per training step (in GFLOPs)
    '''
    return 0.0


# Dataset
class SentimentDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer: AutoTokenizer, max_length: int):
        df = pd.read_csv(csv_path)
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }
        return item


# Model Architecture Components
class CustomBlock(nn.Module): 
    def __init__(self, hidden_size: int, dropout_rate: float = 0.1, expansion_factor: int = 4): 
        """
        Initialize the layers and parameters of this block.

        HINTS:
        - Always call super().__init__() first to inherit from nn.Module.
        - Define any sub-layers you need (e.g., Linear, Conv1d, Dropout).
        - Store any configuration parameters (e.g., hidden size, kernel size).
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.expansion_factor = expansion_factor
        self.linear1 = torch.nn.Linear(hidden_size, hidden_size * expansion_factor)
        self.linear2 = torch.nn.Linear(hidden_size * expansion_factor, hidden_size)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.norm = torch.nn.LayerNorm(hidden_size)
        self.activation = torch.nn.GELU()

    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.dropout(out)
        out = out + residual  # Residual connection
        out = self.norm(out)
        return out


# Example of Custom Block
class CustomMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# Model Config
class SentimentConfig(PretrainedConfig):
    model_type = "sentiment-bert"

    def __init__(
        self,
        model_name="bert-large-uncased", # name of pre-trained model backbone
        num_labels=3,     # number of output classes (Negative, Neutral, Positive)
        head="mlp",       # classifier head
        hidden_dropout=0.1,
        layer_norm_eps=1e-12,
        loss_type="cross_entropy",
        **kwargs,
    ):
        # Always call the parent class initializer first
        pretrained_config = AutoModel.from_pretrained(model_name).config
        super().__init__(**pretrained_config.to_dict(), **kwargs)

        self.model_name = model_name
        self.num_labels = num_labels
        self.head = head
        self.hidden_dropout = hidden_dropout
        self.layer_norm_eps = layer_norm_eps
        self.loss_type = loss_type


# Model (DO NOT change the name "SentimentClassifier")
class SentimentClassifier(nn.Module):
    config_class = SentimentConfig # Which config class to use

    def __init__(self, config: SentimentConfig):
        super().__init__()

        # print(encoder.config)
        # encoder.config.loss_type = "cross_entropy"
        # print(encoder.config)
        self.encoder = AutoModel.from_pretrained(config.model_name)
        self.encoder.config.loss_type = config.loss_type
        self.hidden_size = self.encoder.config.hidden_size
        self.norm = torch.nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        self.dropout = torch.nn.Dropout(config.hidden_dropout)

        if config.head == "mlp":
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(self.hidden_size, self.hidden_size),
                torch.nn.GELU(),
                torch.nn.Dropout(config.hidden_dropout),
                torch.nn.Linear(self.hidden_size, config.num_labels)
            )
        elif config.head == "linear":
            self.classifier = torch.nn.Linear(self.hidden_size, config.num_labels)
        else:
            raise ValueError(f"Unknown head type: {config.head}")
        
        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, labels=None):
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequential_output = encoder_outputs.last_hidden_state
        cls_token = sequential_output[:, 0, :]
        cls_token = self.dropout(self.norm(cls_token))
        logits = self.classifier(cls_token)
        result = {"logits": logits}
        if labels is not None:
            result["loss"] = self.loss_function(logits, labels)
        return result


# Evaluation
@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluate model accuracy on a given dataset.

    Args:
        model: the trained PyTorch model
        dataloader: DataLoader for validation or test set

    Returns:
        acc: overall accuracy
        all_y: true labels
        all_pred: predicted labels
    """
    model.eval()  
    all_y, all_pred = [], []
    with torch.inference_mode():  
        for batch in dataloader:
            '''
            HINTS:
            - Move the batch to the correct device (GPU/CPU)
            - Run a forward pass through the model
            - Get predicted class from logits
            - Save ground-truth and predicted labels
            '''
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs["logits"]
            preds = logits.argmax(dim=-1)

            all_y.extend(labels.cpu().numpy())
            all_pred.extend(preds.cpu().numpy())
    acc = float(accuracy_score(all_y, all_pred))
    return acc, np.array(all_y), np.array(all_pred)


# Training Loop
def train(
    model_name: str,
    train_csv: str,
    val_csv: str,
    test_csv: str,
    out_dir: str,
    epochs: int,
    batch_size: int,
    max_length: int,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    dropout: float = 0.1,
    warmup_steps: int = 100,
    grad_clip: float = 1.0,
    seed: int = 42,
):
    '''
    HINTS:
    - Setup & Reproductibility
    - Prepare datasets and dataloaders
    - Initialize the model
    - Set up optimizer and learning rate scheduler
    - Run the training loop (and save the best checkpoint)
    - Evaluation and save results and metrics
    '''

    # 1. Setup & Reproducibility
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    # DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Prepare datasets and dataloaders (train, val, test)
    '''
    Example:
    tokenizer = AutoTokenizer.from_pretrained(...)
    ds = SentimentDataset(...)
    dl = DataLoader(...)
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    training_set = SentimentDataset(train_csv, tokenizer, max_length)
    validation_set = SentimentDataset(val_csv, tokenizer, max_length)
    testing_set = SentimentDataset(test_csv, tokenizer, max_length)
    dl_train = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
    dl_test = DataLoader(testing_set, batch_size=batch_size, shuffle=False)

    # 3. Initialize the model
    '''
    Example:
    config = SentimentConfig(...)
    model = SentimentClassifier(...).to(DEVICE)
    '''
    config = SentimentConfig(
        model_name=model_name,
        num_labels=3,
        head="mlp",
        hidden_dropout=dropout,
        layer_norm_eps=1e-12,
        loss_type="cross_entropy"
    )
    model = SentimentClassifier(config).to(DEVICE)
    print(model.loss_function)

    # 4. Set up optimizer and learning rate scheduler
    '''
    Example:
    optimizer = optim.AdamW(...)
    scheduler = get_linear_schedule_with_warmup(...)
    '''
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = len(dl_train) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # 5. Run the training loop
    best_val = -1.0
    ckpt_dir = os.path.join(out_dir, "checkpoint") # DO NOT change the file name
    os.makedirs(ckpt_dir, exist_ok=True)
    tokenizer.save_pretrained(ckpt_dir)

    for epoch in range(1, epochs + 1):
        model.train()  
        running_loss = 0.0
        pbar = tqdm(dl_train, desc=f"Epoch {epoch}/{epochs}")

        for batch in pbar:
            '''
            HINTS:
            - Move data to GPU/CPU (the same when doing evaluation)
              -> batch = ...

            - Reset gradients 
              -> optimizer.zero_grad(...)

            - Forward pass
              -> outputs = model(...)
              -> loss = outputs[...]
            
            - Backpropagation
              -> loss.backward()
              -> torch.nn.utils.clip_grad_norm_(...)
            
            - Optimizer step and scheduler update
              -> optimizer.step()
              -> scheduler.step()

            - Update running loss
              -> running_loss += loss.item()
            '''
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            optimizer.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            # print(model.loss_function)
            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

            # Display
            pbar.set_postfix(loss=f"{running_loss/(pbar.n or 1):.4f}")

        # Validation Phase
        val_acc, _, _ = evaluate(model, dl_val)
        print(f"Epoch {epoch}: Val Acc = {val_acc:.4f}")

        # Save best model checkpoint
        if val_acc > best_val:
            best_val = val_acc
            model.save_pretrained(ckpt_dir)

    # 6. Evaluation and save results and metrics
    best = SentimentClassifier.from_pretrained(ckpt_dir).to(DEVICE)

    def eval(split, dl):
        acc, y, yhat = evaluate(best, dl)
        '''
        Save confusion matrix and classification report (you should plot the result prettier)

        Example:
        cm = confusion_matrix(y, yhat, labels=[0,1,2])
        pd.DataFrame(cm).to_csv(os.path.join(ckpt_dir, f"{split}_cm.csv"))
        rpt = classification_report(y, yhat, digits=4, labels=[0,1,2])
        with open(os.path.join(ckpt_dir, f"{split}_report.txt"), "w") as f:
            f.write(rpt)
        '''
        cm = confusion_matrix(y, yhat, labels=[0, 1, 2])
        pd.DataFrame(cm, index=['TN', "TNEU", "TP"], columns=['PN', "PNEU", "PP"]).to_csv(os.path.join(ckpt_dir, f"{split}_cm.csv"))
        report = str(classification_report(y, yhat, digits=4, target_names=['Negative', 'Neutral', 'Positive']))
        with open(os.path.join(ckpt_dir, f"{split}_report.txt"), "w") as f:
            f.write(report)
        return float(acc)

    train_acc = eval("train", dl_train)
    val_acc   = eval("val", dl_val)
    test_acc  = eval("test", dl_test)

    # Save Summary in json format
    summary = {
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "params_trainable": int(sum(p.numel() for p in best.parameters() if p.requires_grad))
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    # Cleanup
    try:
        best.to("cpu"); model.to("cpu")
    except Exception:
        pass
    del best, model, tokenizer, optimizer, scheduler, dl_train, dl_val, dl_test
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Main
def main():
    parser = argparse.ArgumentParser()
    # file paths
    parser.add_argument("--train_csv", type=str, default="./dataset/train.csv")
    parser.add_argument("--test_csv", type=str, default="./dataset/test.csv")
    parser.add_argument("--out_dir", type=str, default="./saved_models/") # DO NOT change the file name

    # model / data
    parser.add_argument("--model_name", type=str, default="...")
    parser.add_argument("--max_length", type=int, default=int)
    parser.add_argument("--batch_size", type=int, default=int)
    parser.add_argument("--epochs", type=int, default=int)

    # architecture
    parser.add_argument("--head", type=str, choices=["mlp"], default="mlp")
    parser.add_argument("--dropout", type=float, default=float)

    # optimization
    parser.add_argument("--lr_encoder", type=float, default=float)
    parser.add_argument("--lr_head", type=float, default=float)
    parser.add_argument("--warmup_ratio", type=float, default=float)

    # Setup
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    '''
    HINTS:
    - Load the dataset and split into training / validation 
    - Save split data
    - Start Training

    Example:
    # train/val split
    full = pd.read_csv(...)

    train_df, val_df = train_test_split(...)
    os.makedirs(arg.out_dir, exist_ok=True)
    train_split = os.path.join(...)
    val_split = os.path.join(...)
    train_df.to_csv(...)
    val_df.to_csv(...)

    # Start training
    train(
        model_name=args.model_name,
        train_csv=train_split,
        val_csv=val_split,
        test_csv=args.test_csv,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
                                     # any other hyperparameters you want to add (e.g., learning rate, dropout, etc.)
        seed=args.seed,
    )
    '''
    print("Loading dataset...")
    full = pd.read_csv(args.train_csv)

    train_df, val_df = train_test_split(full, test_size=0.2, random_state=args.seed, stratify=full["label"])

    os.makedirs(args.out_dir, exist_ok=True)
    train_split = os.path.join(args.out_dir, "train_split.csv")
    val_split = os.path.join(args.out_dir, "val_split.csv")
    train_df.to_csv(train_split, index=False)
    val_df.to_csv(val_split, index=False)

    print("="*50 + "\nStarting training...\n" + "="*50)
    train(
        model_name=args.model_name,
        train_csv=train_split,
        val_csv=val_split,
        test_csv=args.test_csv,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        learning_rate=args.lr_encoder,
        weight_decay=0.01,
        dropout=args.dropout,
        warmup_steps=int(args.warmup_ratio * (len(train_df) // args.batch_size) * args.epochs),
        seed=args.seed,
    )

if __name__ == "__main__":
    main()

