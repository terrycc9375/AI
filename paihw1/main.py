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
        batch_size: int = 32,
        loss_type: str = "mse",
        early_stopping: bool = False,
        patience: int = 20,
        min_delta: float = 1e-6
    ):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
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
        
        x_val = numpy.column_stack([self.validation_set[col] for col in used_cols])
        y_val = numpy.array(self.validation_set[target_col]).reshape(-1, 1)
        best_val_loss = float('inf')
        wait = 0
        stopped_epoch = self.training_args.num_epochs

        n = len(self.training_set)
        bs = self.training_args.batch_size
        loss_val = float('inf')
        loss_history = []

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
                
                indices = numpy.random.permutation(n)
                x_train_shuffled = x_train[indices]
                y_train_shuffled = y_train[indices]
                for i in range(0, n, bs):
                    xb, yb = x_train_shuffled[i : i + bs], y_train_shuffled[i : i + bs]
                    out = self.model.forward(xb)
                    gw, gb = self.model.backward(xb, self.training_args.grad_fn(out, yb))
                    
                    self.model.W -= self.training_args.learning_rate * gw
                    self.model.b -= self.training_args.learning_rate * gb
                    if numpy.isnan(self.model.W).any() or numpy.isinf(self.model.W).any():
                        return {
                            'best_val_loss': 1e10, # 回傳一個很大的數值代表失敗
                            'stopped_epoch': epoch + 1,
                            'loss_history': loss_history
                        }

                current_full_output = self.model.forward(x_train)
                current_loss = self.training_args.loss_fn(current_full_output, y_train)
                loss_val = current_loss.item() if hasattr(current_loss, 'item') else current_loss
                loss_history.append(loss_val)
                
                if self.training_args.early_stopping:
                    val_output = self.model.forward(x_val)
                    current_val_loss = self.training_args.loss_fn(val_output, y_val)
                    
                    if current_val_loss < (best_val_loss - self.training_args.min_delta):
                        best_val_loss = current_val_loss
                        wait = 0
                    else:
                        wait += 1
                    
                    if wait >= self.training_args.patience:
                        stopped_epoch = epoch + 1
                        progress.update(task, completed=self.training_args.num_epochs) # 強制進度條滿格
                        break

                progress.update(
                    task, 
                    advance=1, 
                    description=f"Epoch {epoch+1}/{self.training_args.num_epochs}",
                    loss=loss_val
                )
        return {'best_val_loss': best_val_loss if self.training_args.early_stopping else loss_val, 
                'stopped_epoch': stopped_epoch,
                'loss_history': loss_history
                }

def p1(seed: int = 42, isPlot = True, lr=0.01, epochs=1000, early_stop=False, patience=20, bs=1):
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
    training_args = TrainingArguments(
        learning_rate=lr, 
        num_epochs=epochs, 
        batch_size=bs, 
        loss_type="mse", 
        early_stopping=early_stop,
        patience=patience
    )
    trainer = Trainer(model, training_args, training_set, validation_set)
    metrics = trainer.train()
    
    W, b = trainer.model.export()
    b *= 100.0
    x_test = numpy.array(validation_set['avg']) * 100.0
    y_test = numpy.array(validation_set['final']) * 100.0
    x_range = numpy.linspace(x_test.min(), x_test.max(), 100)
    y_pred = x_range * W.flatten()[0] + b.flatten()[0]
    mse_loss = numpy.mean((y_pred - y_test) ** 2)
    console.print(f"[bold #fa8eec]MSE Loss: {mse_loss:.4f}[/bold #fa8eec]")

    if isPlot:
        plt.scatter(x_test, y_test, color='blue', label='Test Set', alpha=0.6)
        plt.plot(x_range, y_pred, color='red', linewidth=2, label=f'Regression Line: y={W.flatten()[0]:.2f}x+{b.flatten()[0]:.2f}')
        plt.xlabel('Averaged homework scores')
        plt.ylabel('Final exam scores')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig('part1.png')

    return {
        'metrics': metrics, 
        'loss_history': metrics.get('loss_history', []), 
        'W': W, 'b': b, 'lr': lr, 'bs': bs
    }
def p2(seed: int = 42, isPlot=True, lr=0.75, epochs=1000, early_stop=False, patience=30, bs = 1):
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
    training_args = TrainingArguments(
        learning_rate=lr, 
        num_epochs=epochs, 
        batch_size=bs, 
        loss_type="bce", 
        early_stopping=early_stop,
        patience=patience
    )
    
    trainer = Trainer(model, training_args, training_set, validation_set)
    metrics = trainer.train()

    W, b = model.export()


    x1_test_orig = numpy.array(validation_set['avg']) * 100.0
    x2_test_orig = numpy.array(validation_set['final']) * 100.0
    y_test = numpy.array(validation_set['class'])
    X_test_mat = numpy.column_stack((validation_set['avg'], validation_set['final']))
    
    y_pred = model.forward(X_test_mat).flatten()
    log_loss = training_args.loss_fn(y_pred.reshape(-1, 1), y_test.reshape(-1, 1))
    acc = numpy.mean((y_pred > 0.5).astype(int) == y_test)
    console.print(f"[bold #fa8eec]Logistic Loss: {log_loss:.4f}[/bold #fa8eec]")

    if isPlot:
        w1, w2 = W.flatten()
        b_val = b[0]
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

    return {
        'metrics': metrics, 
        'log_loss': log_loss, 
        'W': W, 
        'b': b,
        'x_test_orig': (x1_test_orig, x2_test_orig),
        'y_test': y_test,
        'acc': acc,
        'eta': lr,
        'loss_history': metrics.get('loss_history', []),
        'bs': bs
    }

def p3(seed: int = 42):
    grid_size = 50
    etas = numpy.logspace(-4, -1, grid_size)
    max_T = 1000
    patience = 20
    results = []

    best_overall_mse = float('inf')
    console.print(f"[bold yellow]開始 1D 網格搜索 (含 Early Stopping)...[/bold yellow]")
    
    
    for eta in etas:
        p1_results = p1(seed=seed, isPlot=False, lr=eta, epochs=max_T, early_stop=True, patience=patience)
        metrics = p1_results['metrics']
        final_mse_scaled = metrics['best_val_loss'] * 10000
        stopped_epoch = metrics['stopped_epoch']
        results.append({
            'eta': eta,
            'mse': final_mse_scaled,
            'T_at_stop': stopped_epoch
        })
        if final_mse_scaled < best_overall_mse:
            best_overall_mse = final_mse_scaled
            
        console.print(f"η={eta:.6f} | MSE: {final_mse_scaled:.2f} | 於 Epoch {stopped_epoch} 停止")

    df_results = pandas.DataFrame(results)
    plot_df = df_results[df_results['mse'] < 1e10].copy()
    if plot_df.empty:
        console.print("[bold red]所有學習率都爆炸了！請調低 etas 的上限。[/bold red]")
        return
    csv_filename = "grid_search_1d_linear.csv"
    df_results.to_csv(csv_filename, index=False)
    console.print(f"[green]數據已儲存至 {csv_filename}[/green]")

    console.print("[bold cyan]正在生成分析圖表...[/bold cyan]")

    best_idx = df_results['mse'].idxmin()
    best_eta = df_results.loc[best_idx, 'eta']
    best_mse = df_results.loc[best_idx, 'mse']

    plt.figure(figsize=(10, 6))
    plt.plot(df_results['eta'], df_results['mse'], marker='o', markersize=4, color="#1574c7", label='Validation MSE')
    # plt.annotate(f'Best: η={best_eta:.4f}\nMSE={best_mse:.2f}', 
    #              xy=(best_eta, best_mse), xytext=(best_eta*5, best_mse*10),
    #              arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Learning Rate (eta) - Log Scale')
    plt.ylabel('MSE (Log Scale)')
    plt.title('Model Performance vs. Learning Rate (Log-Log Plot)')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig('analysis_mse_log.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(df_results['eta'], df_results['T_at_stop'], marker='s', markersize=4, color='#faac2f', label='Epochs to Converge')
    plt.xscale('log')
    plt.axhline(y=max_T, color='r', linestyle='--', alpha=0.5, label='Max Epochs Limit')
    plt.xlabel('Learning Rate (eta) - Log Scale')
    plt.ylabel('Stopping Epoch (T)')
    plt.title('Convergence Speed (Early Stopping Insight)')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig('analysis_efficiency_fixed.png')
    plt.close()

    console.print("[bold green]所有圖表已生成：analysis_mse_log.png, analysis_efficiency_fixed.png[/bold green]")


def p4(seed: int = 42):
    etas = numpy.logspace(-4, -1, 50)
    max_T = 5000
    patience = 30
    results = []
    best_loss = float('inf')
    best_data = None

    console.print(f"[bold magenta]開始 Logistic 1D 搜索 (透過 p2 迭代)...[/bold magenta]")

    for eta in etas:
        res = p2(seed=seed, isPlot=False, lr=eta, epochs=max_T, early_stop=True, patience=patience)
        
        current_loss = res['log_loss']
        stopped_epoch = res['metrics']['stopped_epoch']
        acc = res['acc']
        
        results.append({
            'eta': eta,
            'loss': current_loss,
            'T_at_stop': stopped_epoch
        })

        if current_loss < best_loss:
            best_loss = current_loss
            best_data = res
            
        console.print(f"η={eta:.6f} | Loss: {current_loss:.4f} | Acc: {acc:.2%} | T: {stopped_epoch}")

    df_results = pandas.DataFrame(results)
    df_results.to_csv("grid_search_1d_logistic.csv", index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(df_results['eta'], df_results['loss'], marker='o', markersize=4, color="#e84393")
    plt.xscale('log'); plt.xlabel('Learning Rate (eta)'); plt.ylabel('BCE Loss')
    plt.title('Logistic Regression Loss Trend'); plt.grid(True, which="both", alpha=0.2)
    plt.savefig('logistic_analysis_loss.png'); plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(df_results['eta'], df_results['T_at_stop'], marker='s', markersize=4, color="#1eb007", label='Epochs to Converge')
    plt.xscale('log')
    plt.axhline(y=max_T, color='r', linestyle='--', alpha=0.5, label='Max Epochs Limit')
    plt.xlabel('Learning Rate (eta) - Log Scale')
    plt.ylabel('Stopping Epoch (T)')
    plt.title('Convergence Speed (Early Stopping Insight)')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig('p4_epoch2convergence.png')
    plt.close()

    if best_data:
        w1, w2 = best_data['W'].flatten()
        b_val = best_data['b'][0]
        x1_orig, x2_orig = best_data['x_test_orig']
        y_test = best_data['y_test'].flatten()
        
        plt.figure(figsize=(10, 6))
        plt.scatter(x1_orig[y_test==0], x2_orig[y_test==0], color='blue', label='Class 0', alpha=0.6)
        plt.scatter(x1_orig[y_test==1], x2_orig[y_test==1], color='orange', label='Class 1', alpha=0.6)
        x1_range = numpy.linspace(x1_orig.min(), x1_orig.max(), 100)
        decision_boundary = -(w1 / w2) * x1_range - (100.0 * b_val / w2)
        plt.plot(x1_range, decision_boundary, color='red', linewidth=2, label='Min-Loss Boundary')
        plt.xlim(40, 105); plt.ylim(60, 105)
        plt.title(f'Best Logistic Boundary (eta={best_data["eta"]:.4f})')
        plt.savefig('logistic_best_boundary.png'); plt.close()

    console.print("[bold green]p4 完成！圖表與 CSV 已儲存。[/bold green]")

def p5(seed: int = 42):
    fixed_eta = 0.75
    max_epochs = 1000
    
    full_bs = 400 
    batch_sizes = [1, 8, 32, 128, full_bs]
    
    plt.figure(figsize=(10, 6))
    console.print(f"[bold yellow]開始 Batch Size 收斂行為分析 (η={fixed_eta})...[/bold yellow]")

    for bs in batch_sizes:
        res = p2(seed=seed, isPlot=False, lr=fixed_eta, epochs=max_epochs, early_stop=False, bs=bs)
        
        history = res['loss_history']
        label_name = f"Batch Size: {bs}" if bs < full_bs else "Full Batch"

        plt.plot(history, label=label_name, alpha=0.8, linewidth=1.5)
        
        console.print(f"已完成 Batch Size = {bs} 的訓練")

    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss (Log Scale)')
    plt.title(f'Convergence Behavior Comparison (eta={fixed_eta})')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    
    plt.savefig('batch_size_convergence.png')
    plt.close()
    
    console.print("[bold green]分析完成！收斂曲線圖已儲存為 batch_size_convergence.png[/bold green]")

def p6(seed: int = 42):
    fixed_eta = 0.01
    max_epochs = 1000
    
    full_bs = 400 
    batch_sizes = [1, 8, 32, 128, full_bs]
    
    plt.figure(figsize=(10, 6))
    console.print(f"[bold blue]開始 Linear Regression Batch Size 分析 (η={fixed_eta})...[/bold blue]")

    for bs in batch_sizes:
        
        res = p1(seed=seed, isPlot=False, lr=fixed_eta, epochs=max_epochs, early_stop=False, bs=bs)
        
        history = res['loss_history']
        label_name = f"Batch Size: {bs}" if bs < full_bs else "Full Batch"
        
        plt.plot(history, label=label_name, alpha=0.8)
        console.print(f"已完成 Linear Batch Size = {bs}")

    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss (Log Scale)')
    plt.title(f'Linear Regression Convergence (eta={fixed_eta})')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    
    plt.savefig('linear_batch_size_convergence.png')
    plt.close()
    
    console.print("[bold green]p6 分析完成！圖表已儲存為 linear_batch_size_convergence.png[/bold green]")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", "-s", type=int, default=42, required=False)
    parser.add_argument("--problem", "-p", type=int, choices=[1, 2, 3, 4, 5, 6], required=True)
    args = parser.parse_args()
    
    random.seed(args.seed)
    numpy.random.seed(args.seed)
    
    if args.problem == 1:
        p1(args.seed)
    elif args.problem == 2:
        p2()
    elif args.problem == 3:
        p3()
    elif args.problem == 4:
        p4()
    elif args.problem == 5:
        p5()
    else:
        p6()
        


if __name__ == "__main__":
    main()
