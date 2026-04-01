import os
import sys
import warnings
import time
import random
import argparse

import numpy
import pandas
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
import rich.console
import datasets
import matplotlib.pyplot as plt

console = rich.console.Console()

class Linear():
    def __init__(self, input_size: int = 1, output_size: int = 1):
        self.input_size = input_size
        self.output_size = output_size
        self.W = numpy.random.rand(input_size, output_size)
        self.b = numpy.random.rand(output_size)
        
    def forward(self, x: numpy.ndarray) -> numpy.ndarray:
        return x @ self.W + self.b
    
    def backward(self, x: numpy.ndarray, grad_output: numpy.ndarray) -> tuple:
        grad_W = x.T @ grad_output
        grad_b = numpy.sum(grad_output, axis=0)
        return grad_W, grad_b
    
    def export(self) -> tuple:
        return (self.W, self.b)
    
class Logistic(Linear):
    def forward(self, x: numpy.ndarray) -> numpy.ndarray:
        z = x @ self.W + self.b
        return 1 / (1 + numpy.exp(-z))

    def backward(self, x: numpy.ndarray, grad_output: numpy.ndarray) -> tuple:
        grad_W = x.T @ grad_output
        grad_b = numpy.sum(grad_output, axis=0)
        return grad_W, grad_b

class TrainingArguments():
    def __init__(
        self, 
        learning_rate: float = 0.01, 
        num_epochs: int = 1000,
        loss_type: str = "mse"
    ):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        if loss_type == "mse":
            self.loss_fn = lambda y_pred, y_true: numpy.mean((y_pred - y_true) ** 2)
            self.grad_fn = lambda y_pred, y_true: 2 * (y_pred - y_true) / y_true.shape[0]
            
        elif loss_type == "bce":
            self.loss_fn = lambda y_pred, y_true: -numpy.mean(
                y_true * numpy.log(y_pred + 1e-9) + (1 - y_true) * numpy.log(1 - y_pred + 1e-9)
            )
            self.grad_fn = lambda y_pred, y_true: (y_pred - y_true) / y_true.shape[0]

class Trainer():
    def __init__(
        self,
        model: Linear,
        training_args: TrainingArguments,
        training_set: datasets.Dataset, 
        validation_set: datasets.Dataset
    ):
        self.model = model
        self.training_args = training_args
        self.training_set = training_set
        self.validation_set = validation_set

    def train(self):
        feature_cols = ['avg', 'final']
        used_cols = feature_cols[:self.model.input_size]
        x_train = numpy.column_stack([self.training_set[col] for col in used_cols])
        target_col = 'class' if 'class' in self.training_set.column_names else 'final'
        y_train = numpy.array(self.training_set[target_col]).reshape(-1, 1)
        n = len(self.training_set)

        with Progress(
            TextColumn("[#238ce8][progress.description]{task.description}"),
            BarColumn(),                                          
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),                                   
            TextColumn("/"),
            TimeRemainingColumn(),                                  
            TextColumn("[#faac2f]loss: {task.fields[loss]:.3f}"), 
        ) as progress:
            
            task = progress.add_task(
                description=f"Epoch 0/{self.training_args.num_epochs}", 
                total=self.training_args.num_epochs,
                loss=0.0
            )

            for epoch in range(self.training_args.num_epochs):
                output = self.model.forward(x_train)
                
                loss = self.training_args.loss_fn(output, y_train)
                loss_val = loss.item() if isinstance(loss, numpy.ndarray) else loss
                grad_output = self.training_args.grad_fn(output, y_train)
                grad_W, grad_b = self.model.backward(x_train, grad_output)
                
                self.model.W -= self.training_args.learning_rate * grad_W
                self.model.b -= self.training_args.learning_rate * grad_b

                progress.update(
                    task, 
                    advance=1, 
                    description=f"Epoch {epoch+1}/{self.training_args.num_epochs}",
                    loss=loss_val
                )
        

def p1(seed: int = 42):
    avg = pandas.read_csv("Problem 1/Averaged homework scores.csv")
    final = pandas.read_csv("Problem 1/Final exam scores.csv")
    df = pandas.concat([avg, final], axis=1)
    df.columns = ['avg', 'final']
    df['avg'] = df['avg'] / 100.0
    df['final'] = df['final'] / 100.0
    dataset = datasets.Dataset.from_pandas(df)
    training_set = dataset.train_test_split(test_size=0.2, seed=seed)['train']
    validation_set = dataset.train_test_split(test_size=0.2, seed=seed)['test']
    
    model = Linear(input_size=1, output_size=1)
    training_args = TrainingArguments(learning_rate=0.01, num_epochs=1000, loss_type="mse")
    trainer = Trainer(model, training_args, training_set, validation_set)
    trainer.train()
    
    W, b = trainer.model.export()
    b *= 100.0
    x_test = numpy.array(validation_set['avg']) * 100.0
    y_test = numpy.array(validation_set['final']) * 100.0
    x_range = numpy.linspace(x_test.min(), x_test.max(), 100)
    y_pred = x_range * W.flatten()[0] + b.flatten()[0]
    mse_loss = numpy.mean((y_pred - y_test) ** 2)
    console.print(f"[bold #fa8eec]MSE Loss: {mse_loss:.4f}[/bold #fa8eec]")

    plt.scatter(x_test, y_test, color='blue', label='Test Set', alpha=0.6)
    plt.plot(x_range, y_pred, color='red', linewidth=2, label=f'Regression Line: y={W.flatten()[0]:.2f}x+{b.flatten()[0]:.2f}')

    plt.xlabel('Averaged homework scores')
    plt.ylabel('Final exam scores')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('part1.png')

def p2(seed: int = 42):
    avg = pandas.read_csv("Problem 2/Averaged homework scores.csv")
    final = pandas.read_csv("Problem 2/Final exam scores.csv")
    label = pandas.read_csv("Problem 2/Results.csv")
    df = pandas.concat([avg, final, label], axis=1)
    df.columns = ['avg', 'final', 'class']
    df['avg'] = df['avg'] / 100.0
    df['final'] = df['final'] / 100.0
    dataset = datasets.Dataset.from_pandas(df)
    train_split = dataset.train_test_split(test_size=0.2, seed=seed)
    training_set = train_split['train']
    validation_set = train_split['test']

    model = Logistic(input_size=2, output_size=1)
    training_args = TrainingArguments(learning_rate=0.75, num_epochs=1000, loss_type="bce")
    
    trainer = Trainer(model, training_args, training_set, validation_set)
    trainer.train()
    
    W, b = model.export()
    w1, w2 = W.flatten()
    b_val = b[0]

    x1_test_orig = numpy.array(validation_set['avg']) * 100.0
    x2_test_orig = numpy.array(validation_set['final']) * 100.0
    y_test = numpy.array(validation_set['class'])

    X_test_mat = numpy.column_stack((validation_set['avg'], validation_set['final']))
    y_pred = model.forward(X_test_mat).flatten()
    log_loss = training_args.loss_fn(y_pred.reshape(-1, 1), y_test.reshape(-1, 1))
    console.print(f"[bold #fa8eec]Logistic Loss: {log_loss:.4f}[/bold #fa8eec]")

    plt.figure(figsize=(10, 6))
    plt.scatter(x1_test_orig[y_test==0], x2_test_orig[y_test==0], color='blue', label='Class 0', alpha=0.6)
    plt.scatter(x1_test_orig[y_test==1], x2_test_orig[y_test==1], color='orange', label='Class 1', alpha=0.6)

    x1_range = numpy.linspace(x1_test_orig.min(), x1_test_orig.max(), 100)
    
    decision_boundary = -(w1 / w2) * x1_range - (100.0 * b_val / w2)
    
    plt.plot(x1_range, decision_boundary, color='red', linewidth=2, label='Decision Boundary')

    plt.xlim(40, 105)
    plt.ylim(60, 105)
    plt.xlabel('Averaged homework scores')
    plt.ylabel('Final exam scores')
    plt.title('Logistic Regression Decision Boundary')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('part2.png')

def p3(seed: int = 42):
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", "-s", type=int, default=42, required=False)
    parser.add_argument("--problem", "-p", type=int, choices=[1, 2, 3], required=True)
    args = parser.parse_args()
    
    random.seed(args.seed)
    numpy.random.seed(args.seed)
    
    if args.problem == 1:
        # p1
        p1(args.seed)
    elif args.problem == 2:
        # p2
        p2()

    else:
        # p3
        p3()


if __name__ == "__main__":
    main()
