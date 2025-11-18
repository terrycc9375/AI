# Neural Network Library for NLP
import torch
import sklearn.model_selection
import numpy
import random

# Pretrained models and tokenizers
import transformers

# Data processers
import pandas
import json
import argparse
import os
import datasets

# Terminal progress bar
import tqdm
import time
import rich

# Reproducibility
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True

def set_seed(seed: int = 42):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path: str, tokenizer: transformers.AutoTokenizer, max_length: int):
        file = pandas.read_csv(csv_path)
        self.texts = file['text'].tolist()
        self.labels = file['label'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
       text = self.texts[idx]
       label = self.labels[idx]
       encoded = self.tokenizer(
           text,
           padding='max_length',
           truncation=True,
           max_length=self.max_length,
           return_tensors='pt'
       )
       item = {
           'input_ids': encoded['input_ids'].squeeze(0),
           'attention_mask': encoded['attention_mask'].squeeze(0),
           'label': torch.tensor(label, dtype=torch.long)
       }
       return item

class CustomBlock(torch.nn.Module):
    pass

class CustomMLP(torch.nn.Module):
    pass

class SentimentConfig(transformers.PretrainedConfig):
    model_type = "sentiment_model"
    def __init__(
            self,
            model_name: str = "bert-base-uncased",
            num_labels: int = 3,
            head: str = "mlp",
            **kwargs
        ):
        config_dict = transformers.AutoConfig.from_pretrained(model_name).to_dict()
        config_dict["num_labels"] = num_labels
        config_dict["id2label"] = {'0': "Negative", '1': "Neutral", '2': "Positive"}
        config_dict["label2id"] = {"Negative": 0, "Neutral": 1, "Positive": 2}
        config_dict.update(kwargs)
        super().__init__(**config_dict)
        self.head = head

class SentimentClassifier(transformers.PreTrainedModel):
    config_class = SentimentConfig

    def __init__(self, config: SentimentConfig):
        super().__init__(config)
        self.bert = transformers.AutoModel.from_config(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        if config.head == "mlp":
            self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
        elif config.head == "custom":
            self.classifier = CustomMLP()
        
        #initialize weights
        self.post_init()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
            **kwargs
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_function = torch.nn.CrossEntropyLoss()
            loss = loss_function(logits.view(-1, self.config.num_labels), labels.view(-1))

        return {"loss": loss, "logits": logits}

def train(
        model_name: str,
        train_csv: str,
        test_csv: str,
        out_dir: str,
        max_length: int = 128,
        batch_size: int = 4,
        epochs: int = 1,
        seed: int = 42,
):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    test_dataset = datasets.Dataset.from_pandas(pandas.read_csv(test_csv))
    tokenized_test = test_dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding=True, max_length=128), batched=True)

    config = SentimentConfig(model_name=model_name, head="mlp")
    model = SentimentClassifier(config).to(DEVICE) # type: ignore

    best_value = -1.0
    checkpoint_dir = os.path.join(out_dir, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    tokenizer.save_pretrained(checkpoint_dir)

    for epoch in range(epochs):
        # split training set and validation set
        data_frame = pandas.read_csv(train_csv)
        train_set, valid_set = sklearn.model_selection.train_test_split(
            data_frame,
            test_size=0.2,
            train_size=0.8,
            random_state=seed,
            shuffle=True,
            stratify=data_frame["label"]
        )
        train_dataset = datasets.Dataset.from_pandas(train_set)
        valid_dataset = datasets.Dataset.from_pandas(valid_set)
        tokenized_train = train_dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding=True, max_length=128), batched=True)
        tokenized_valid = valid_dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding=True, max_length=128), batched=True)

        training_arguments = transformers.TrainingArguments(
            output_dir=out_dir,
            evaluation_strategy="epoch",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=3e-5,
            num_train_epochs=epochs,
            logging_strategy="epoch",
            save_strategy="epoch",
            seed=seed,
            fp16=torch.cuda.is_available(),
            greater_is_better=True,
            report_to="none",
        )

        trainer = transformers.Trainer(
            model=model,
            args=training_arguments,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_valid,
            tokenizer=tokenizer,
            compute_loss_func=torch.nn.CrossEntropyLoss(),
            compute_metrics=lambda p: {
                "accuracy": (p.predictions.argmax(-1) == p.label_ids).mean() # type: ignore
            }
        )

        trainer.train()
        # 1. show dynamic progress bar
        # 2. evaluate accuracy by validation set
        # 3. save best model

def main():
    parser = argparse.ArgumentParser()
    # file paths, no need to change
    parser.add_argument("--train_csv", type=str, default="./dataset/train.csv")
    parser.add_argument("--test_csv", type=str, default="./dataset/test.csv")
    parser.add_argument("--out_dir", type=str, default="./saved_models/")
    # model parameters
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, required=True, default=4)
    parser.add_argument("--epochs", type=int, required=True, default=1)
    parser.add_argument("--head", type=str, choices=["custom_block", "mlp"], default="mlp")
    parser.add_argument("-lr", "--learning_rate", type=float, default=5e-5)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.01)
    parser.add_argument("-ws", "--warmup_steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    raw_dataset = pandas.read_csv(args.train_csv)
    # 90% for training, 10% for validation as default
    used2train, used2test = sklearn.model_selection.train_test_split(
        raw_dataset,
        test_size=0.1,
        random_state=args.seed,
        shuffle=True,
        stratify=raw_dataset['label']
    )
    # save splitted datasets
    used2train.to_csv(os.path.join(args.out_dir, args.train_csv), index=False)
    used2test.to_csv(os.path.join(args.out_dir, args.test_csv), index=False)

    # start training
    train(
        model_name=args.model_name,
        train_csv=os.path.join(args.out_dir, args.train_csv),
        test_csv=os.path.join(args.out_dir, args.test_csv),
        out_dir=args.out_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()
