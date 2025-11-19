# Neural Network Library for NLP
import torch
import sklearn, sklearn.model_selection, sklearn.metrics
import numpy
import random

# Pretrained models and tokenizers
import transformers, transformers.modeling_outputs

# Data processers
import pandas
import json
import argparse
import os
import datasets
import warnings, logging
import gc

# Terminal progress bar
import tqdm
import time
import rich, rich.progress
import typing

# Reproducibility
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True

# disable warnings
datasets.disable_progress_bar()
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_EXPERIMENTAL_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
transformers.logging.set_verbosity_error()
transformers.logging.get_logger("transformers").setLevel(logging.CRITICAL)
transformers.logging.get_logger("transformers.trainer").setLevel(logging.CRITICAL)

# ========= 核彈級追蹤：抓住所有 print =========
import builtins
_original_print = builtins.print

def detective_print(*args, **kwargs):
    import traceback
    stack = traceback.extract_stack()
    # 看是哪個檔案、哪一行在 print
    for frame in stack[-5:]:  # 印最近 5 層呼叫堆疊
        if "trainer.py" in frame.filename or "training_loop" in frame.filename:
            print("\n罪魁禍首發現！")
            print(f"File: {frame.filename}")
            print(f"Line: {frame.lineno}")
            print(f"Code: {frame.line}")
            print("-" * 50)
            break
    _original_print(*args, **kwargs)

# 啟用偵探模式
builtins.print = detective_print
# ============================================

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
        self.model_name = model_name

class SentimentClassifier(transformers.PreTrainedModel):
    config_class = SentimentConfig

    def __init__(self, config: SentimentConfig):
        super().__init__(config)
        self.bert = transformers.AutoModel.from_pretrained(config.model_name)
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
        # allowed_kwargs = ["output_attentions", "output_hidden_states", "return_dict", "head_mask", "inputs_embeds"]
        # bert_kwargs = {k: v for k, v in kwargs.items() if k in allowed_kwargs}
        kwargs.pop("num_items_in_batch", None)  # sol.1 remove unexpected kwarg
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **{k: v for k, v in kwargs.items() if k in ["output_attentions", "output_hidden_states"]}
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_function = torch.nn.CrossEntropyLoss()
            loss = loss_function(logits.view(-1, self.config.num_labels), labels.view(-1))

        return transformers.modeling_outputs.SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

console = rich.console.Console()
class RichProgressCallback(transformers.TrainerCallback):
    def __init__(self, current_epoch: int, total_epochs: int):
        self.progress = None
        self.train_task = None
        self.epoch_start_time = float()
        self.current_epoch = current_epoch
        self.steps_per_epoch = None
        self.total_epochs = total_epochs
        self.running_loss = 0.0
        self.loss_steps = 0

    def on_train_begin(self, args, state, control, **kwargs):
        # self.total_epochs = args.num_train_epochs
        self.steps_per_epoch = state.max_steps# // args.num_train_epochs
        if state.max_steps % args.num_train_epochs != 0:
            self.steps_per_epoch += 1
        
        self.progress = rich.progress.Progress(
            rich.progress.TextColumn(f"[bold blue] Epoch {self.current_epoch}/{self.total_epochs}"),
            rich.progress.BarColumn(),
            rich.progress.MofNCompleteColumn(),
            rich.progress.TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            rich.progress.TextColumn("•"),
            rich.progress.TextColumn("{task.fields[time_info]}"),
            rich.progress.TextColumn("• {task.fields[speed_info]} it/s"),
            rich.progress.TextColumn("• loss = {task.fields[loss]:.4f}"),
            console=console,
            transient=False,
        )
        self.progress.start()
        self._new_epoch(state)

    def _new_epoch(self, state=None):
        if self.train_task is not None:
            self.progress.update(self.train_task, completed=self.steps_per_epoch)

        self.train_task = self.progress.add_task(
            f"[green]Training",
            total=self.steps_per_epoch,
            # epoch=self.current_epoch,
            # total_epochs=self.total_epochs,
            time_info="0:00:00 / -:--:--",
            speed_info="0.0",
            loss=0.0,
        )
        self.epoch_start_time = time.time()
        self.running_loss = 0.0
        self.loss_steps = 0

    def on_step_end(self, args, state, control, **kwargs):
        if self.train_task is None: return

        step_in_epoch = state.global_step % self.steps_per_epoch
        if step_in_epoch == 0 and state.global_step > 0:
            step_in_epoch = self.steps_per_epoch

        # accumulate time
        elapsed = time.time() - self.epoch_start_time
        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed))

        #calculate loss
        if state.log_history and len(state.log_history) > 0:
            latest = state.log_history[-1]
            if "loss" in latest:
                self.running_loss += latest["loss"]
                self.loss_steps += 1
        avg_loss = self.running_loss / self.loss_steps if self.loss_steps > 0 else 0.0

        if step_in_epoch > 0 and elapsed > 0:
            speed = step_in_epoch / elapsed
            estimated_time = time.strftime("%H:%M:%S", time.gmtime(elapsed * self.steps_per_epoch / step_in_epoch))
            time_info = f"[bold white]{elapsed_time}[/] / [bold cyan]{estimated_time}[/]"
            speed_info = f"{speed:.1f}"
        else:
            time_info = f"[bold white]{elapsed_time}[/] / [bold cyan]-:--:--[/]"
            speed_info = "0.0"

        self.progress.update(self.train_task,
            advance=1, 
            completed=step_in_epoch, 
            time_info=time_info, 
            speed_info=speed_info,
            loss=avg_loss,
        )

    def on_epoch_end(self, args, state, control, **kwargs):
        # self.current_epoch += 1
        # if self.current_epoch <= self.total_epochs:
        #     self._new_epoch(state)
        # self.progress.update(self.train_task, epoch=int(state.epoch))
        pass # since epoch = 1

    def on_train_end(self, args, state, control, **kwargs):
        if self.progress:
            if self.train_task:
                self.progress.update(self.train_task, completed=self.steps_per_epoch)
            self.progress.stop()

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
    tokenized_test = test_dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding=True, max_length=128), batched=True, remove_columns=["text"])

    console.print(f"[bold blue]Using device: {DEVICE}[/bold blue]")
    config = SentimentConfig(model_name=model_name, head="mlp")
    model = SentimentClassifier(config).to(DEVICE) # type: ignore

    best_value = -1.0
    checkpoint_dir = os.path.join(out_dir, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    tokenizer.save_pretrained(checkpoint_dir)

    # print("Initialize successful")

    class SentimentTrainer(transformers.Trainer):
        def __init__(self, *args, **kwargs):
            if "callbacks" in kwargs:
                kwargs["callbacks"] = [
                    cb for cb in kwargs["callbacks"] if not cb.__class__.__name__.startswith("LoggerCallback")
                ]
            else: 
                kwargs["callbacks"] = []
            super().__init__(*args, **kwargs)
            
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = torch.nn.CrossEntropyLoss()(outputs.logits, labels) if labels is not None else None
            return (loss, outputs) if return_outputs else loss
        
        def log(self, logs, start_time: typing.Optional[float] = None):
            if self.state.epoch is not None:
                logs["epoch"] = self.state.epoch
            
            output = {**logs, **{"step": self.state.global_step}}
            self.state.log_history.append(output)
            return self.control

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
        tokenized_train = train_dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding=True, max_length=128), batched=True, remove_columns=["text"])
        tokenized_valid = valid_dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding=True, max_length=128), batched=True, remove_columns=["text"])

        os.makedirs("./logs", exist_ok=True)
        training_arguments = transformers.TrainingArguments(
            output_dir=out_dir,
            eval_strategy="epoch",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            learning_rate=3e-5,
            num_train_epochs=1,
            logging_dir="./logs",
            logging_strategy="steps",
            logging_first_step=True,
            logging_steps=10,
            save_strategy="no",
            seed=seed,
            fp16=torch.cuda.is_available(),
            greater_is_better=True,
            report_to=[],
            # report_to="tensorboard",
            disable_tqdm=True
        )

        trainer = SentimentTrainer(
            model=model,
            args=training_arguments,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_valid,
            processing_class=tokenizer,
            compute_loss_func=torch.nn.CrossEntropyLoss(),
            compute_metrics=lambda p: {
                "accuracy": (p.predictions.argmax(-1) == p.label_ids).mean() # type: ignore
            },
            callbacks=[RichProgressCallback(epoch + 1, epochs)]
        )

        trainer.train()

        metrics = trainer.evaluate()
        val_accuracy = metrics.get("eval_accuracy", -1.0)
        console.print(f"[bold yellow]Validation Accuracy: {val_accuracy:.4f}[/bold yellow]")
        if val_accuracy > best_value:
            best_value = val_accuracy
            trainer.save_model(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)

    console.print(f"\n[bold magenta]Training complete.")
    best_checkpoint = trainer.state.best_model_checkpoint
    best_model = SentimentClassifier.from_pretrained(best_checkpoint).to(DEVICE) if best_checkpoint is not None else model.to(DEVICE) # type: ignore
    test_result = trainer.predict(tokenized_test)
    predictions = test_result.predictions.argmax(-1)
    true_labels = test_result.label_ids
    accuracy = sklearn.metrics.accuracy_score(true_labels, predictions)
    summary = {
        "Accuracy": accuracy,
        "params": sum(p.numel() for p in best_model.parameters()),
        "params_trainable": sum(p.numel() for p in best_model.parameters() if p.requires_grad)
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
    console.print(f"[bold green]Test Accuracy: {accuracy:.4f}[/bold green]")

    try:
        best_model.cpu()
    except:
        pass
    del best_model, model, tokenizer, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    # file paths, no need to change
    parser.add_argument("--train_csv", type=str, default="./train.csv")
    parser.add_argument("--test_csv", type=str, default="./test.csv")
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

    raw_dataset = pandas.read_csv(os.path.join("./dataset/", "dataset.csv"))
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
