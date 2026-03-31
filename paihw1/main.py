import os
import sys
import warnings
import time
import random
import argparse

import numpy
import pandas
import rich
import datasets
import matplotlib.pyplot as plt

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

class TrainingArguments():
    def __init__(
        self, 
        learning_rate: float = 0.01, 
        num_epochs: int = 1000
    ):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        
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
        for epoch in range(self.training_args.num_epochs):
            # Forward pass
            output = self.model.forward(numpy.array(self.training_set['avg']).reshape(-1, 1)) # type: ignore
            loss = numpy.mean((output - self.training_set['final']) ** 2) # type: ignore
            grad_W, grad_b = self.model.backward(numpy.array(self.training_set['avg']).reshape(-1, 1), 2 * (output - numpy.array(self.training_set['final']).reshape(-1, 1)) / len(self.training_set)) # type: ignore
            self.model.W -= self.training_args.learning_rate * grad_W
            self.model.b -= self.training_args.learning_rate * grad_b
        

def p1(seed: int = 42):
    avg = pandas.read_csv("Problem 1/Averaged homework scores.csv")
    final = pandas.read_csv("Problem 1/Final exam scores.csv")
    df = pandas.concat([avg, final], axis=1)
    df.columns = ['avg', 'final']
    dataset = datasets.Dataset.from_pandas(df)
    training_set = dataset.train_test_split(test_size=0.2, seed=seed)['train']
    validation_set = dataset.train_test_split(test_size=0.2, seed=seed)['test']
    model = Linear(input_size=1, output_size=1)
    training_args = TrainingArguments(learning_rate=0.01, num_epochs=1000)
    trainer = Trainer(model, training_args, training_set, validation_set)
    trainer.train()
    
    W, b = trainer.model.export()
    x_test = numpy.array(validation_set['Averaged homework scores'])
    y_test = numpy.array(validation_set['Final exam scores'])
    x_range = numpy.linspace(x_test.min(), x_test.max(), 100)
    y_pred = x_range * W.flatten()[0] + b.flatten()[0]

    plt.scatter(x_test, y_test, color='blue', label='Actual Data (Test Set)', alpha=0.6)
    plt.plot(x_range, y_pred, color='red', linewidth=2, label=f'Regression Line: y={W.flatten()[0]:.2f}x+{b.flatten()[0]:.2f}')

    plt.xlabel('Averaged homework scores')
    plt.ylabel('Final exam scores')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('part1.png')

def p2(seed: int = 42):
    pass

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
