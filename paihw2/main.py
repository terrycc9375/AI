import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.console import Console
from matplotlib import pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="numpy")

console = Console()

class Classifier_CNN(nn.Module):
    def __init__(self, model_path: str | None = None):
        super(Classifier_CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 10)
        )
        
        if model_path:
            self.load_state_dict(torch.load(model_path))

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class Classifier_P3_Enhanced(nn.Module):
    def __init__(self):
        super(Classifier_P3_Enhanced, self).__init__()
        
        def conv_block(in_f, out_f):
            return nn.Sequential(
                nn.Conv2d(in_f, out_f, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_f),
                nn.ReLU(),
                nn.Conv2d(out_f, out_f, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_f),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.1) # 輕微的 Spatial Dropout
            )

        self.features = nn.Sequential(
            conv_block(3, 64),    # 32x32 -> 16x16
            conv_block(64, 128),  # 16x16 -> 8x8
            conv_block(128, 256), # 8x8 -> 4x4
            nn.Conv2d(256, 512, 3, padding=1), # 最後一層提取深層特徵
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) # Global Average Pooling
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class ASGRActivation(nn.Module):
    def __init__(self, alpha=0.1, learnable=True):
        super(ASGRActivation, self).__init__()
        if learnable:
            self.alpha = nn.Parameter(torch.tensor([alpha]))
        else:
            self.alpha = alpha

    def forward(self, x):
        # g(x) = relu(x) + alpha * swish(x)
        return F.relu(x) + self.alpha * (x * torch.sigmoid(x))  

class ASGRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            ASGRActivation(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.05),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            ASGRActivation(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.05),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            ASGRActivation(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        return self.classifier(x)

class ASGRModel2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            ASGRActivation(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.05),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            ASGRActivation(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.05),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            ASGRActivation(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.05),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            ASGRActivation(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        return self.classifier(x)

class ASGRModel3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            ASGRActivation(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.05),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            ASGRActivation(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.05),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            ASGRActivation(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.05),
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            ASGRActivation(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.05),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 128),
            ASGRActivation(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        return self.classifier(x)
    
class ASGRModel4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            ASGRActivation(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.05),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            ASGRActivation(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.05),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            ASGRActivation(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.05),
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            ASGRActivation(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.05),
        )
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            ASGRActivation(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.05),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            ASGRActivation(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        return self.classifier(x)

class ResNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            ASGRActivation(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.05),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            ASGRActivation(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.05),
        )
        self.res_block = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            ASGRActivation(),
            nn.Dropout2d(0.05),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.BatchNorm1d(128),
            ASGRActivation(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        res = self.res_block(x)
        x = x + res # 簡單的殘差連接
        return self.classifier(x)

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(3072, 512),  # 第一層: 32*32*3=3072
            nn.ReLU(),
            nn.Linear(512, 256),   # 第二層
            nn.ReLU(),
            nn.Linear(256, 128),   # 第三層
            nn.ReLU(),
            nn.Linear(128, 10)     # 輸出層 (Softmax 包含在 CrossEntropyLoss 中)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)

class PlainCNN_3n(nn.Module):
    def __init__(self, n):
        super(PlainCNN_3n, self).__init__()
        self.n = n
        self.stage1 = self._make_stage(3, 32, n, downsample=True)  # 32x32 -> 16x16
        self.stage2 = self._make_stage(32, 64, n, downsample=True) # 16x16 -> 8x8
        self.stage3 = self._make_stage(64, 128, n, downsample=False) # 維持 8x8
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.BatchNorm1d(256),
            ASGRActivation(),
            nn.Linear(256, 10)
        )

    def _make_stage(self, in_channels: int, out_channels: int, num_layers: int, downsample: bool):
        layers = []
        for i in range(num_layers):
            input_f = in_channels if i == 0 else out_channels
            layers.append(nn.Conv2d(input_f, out_channels, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(ASGRActivation())
            layers.append(nn.Dropout2d(0.05))
            
        if downsample:
            layers.append(nn.MaxPool2d(2, 2))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.classifier(x)

def main():
    # current_task 決定要做第幾題，1 是 MLP，2 是 CNN，3 是 Hyperparameter selection
    current_task = 3
    if torch.cuda.is_available():
        device = "cuda"
    else:
        raise Exception("GPU is not available.")
        exit(0)
    
    if current_task == 1:
        print("正在執行 Part 1: Simple MLP 訓練")
        model = SimpleMLP().to(device)
        save_path = "cifar10_mlp_model.pth"
    elif current_task == 2:
        print("正在執行 Part 2: CNN 訓練")
        model = Classifier_CNN().to(device)
        save_path = "cifar10_cnn_model.pth"
    elif current_task == 3:
        print("正在執行 Part 3: Hyperparameter selection 訓練")
        #model = Classifier_P3().to(device)
        model = Classifier_P3_Enhanced().to(device)
        save_path = "cifar10_P3_model.pth"

        
    # Training arguments
    batch_size = 1024
    learning_rate = 0.001
    epochs = 50
    

    if current_task == 3:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(), # 隨機水平翻轉
            transforms.RandomCrop(32, padding=4), # 隨機裁剪
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    #trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    with Progress(
        TextColumn("[#238ce8][progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[#faac2f]Test Acc: {task.fields[test_acc]:>5.2f}%"),
        TimeRemainingColumn(),
    ) as progress:
        main_task = progress.add_task("Training Progress", total=epochs, test_acc=0.0)

        for epoch in range(epochs):
            model.train()
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in testloader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
            test_accuracy = 100 * correct / total

            progress.update(
                main_task, 
                advance=1, 
                description=f"Epoch {epoch+1}/{epochs}",
                test_acc=test_accuracy,
            )

    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    console.print(f'[bold #fa8eec]Test Accuracy: {100 * correct / total:.2f}%')

def p4():
    # configs = [
    #     ASGRModel(),
    #     ASGRModel2(),
    #     ASGRModel3(),
    #     ASGRModel4(),
    #     ResNet2(),
    # ]
    DEVICE = torch.device("cuda")
    batch_size = 512
    learning_rate = 0.001
    epochs = 12
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(), # 隨機水平翻轉
        transforms.RandomCrop(32, padding=4), # 隨機裁剪
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    #trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    # results = [[] for _ in range(len(configs))]
    results = [[] for _ in range(7)]
    # for i, model_class in enumerate(configs):
    #     model = model_class.to(DEVICE)
    for i in range(7):
        model = PlainCNN_3n(i + 1).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # save_path = f"cifar10_P4_model_{i+1}.pth"
        save_path = f"P4_{3*(i+1)}_layers.pth"

        with Progress(
            TextColumn("[#238ce8][progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[#faac2f]Test Acc: {task.fields[test_acc]:>5.2f}%"),
            TimeRemainingColumn(),
        ) as progress:
            main_task = progress.add_task("Training Progress", total=epochs, test_acc=0.0)

            for epoch in range(epochs):
                model.train()
                for inputs, labels in trainloader:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for data, target in testloader:
                        data, target = data.to(DEVICE), target.to(DEVICE)
                        output = model(data)
                        
                        _, predicted = torch.max(output.data, 1)
                        total += target.size(0)
                        correct += (predicted == target).sum().item()
                test_accuracy = 100 * correct / total
                results[i].append(test_accuracy)

                progress.update(
                    main_task, 
                    advance=1, 
                    description=f"Epoch {epoch+1}/{epochs}",
                    test_acc=test_accuracy,
                )
        torch.save(model.state_dict(), save_path)
        print(f"\nModel saved to {save_path}")
    plot_results(results)

def plot_results(results):
    """
    results: 形式為 (6, 50) 的 list 或 numpy array
    config_names: 長度為 6 的 list，包含每條線的標籤 (例如 ['Shallow', 'Deep', ...])
    """
    results = np.array(results)
    epochs = np.arange(1, 13) # 1 到 12
    config_names = [f"{3 * (depth + 1)} layers" for depth in range(len(results))]
    # config_names[-1] = "ResNet (2 layers + 1 res block)"

    plt.figure(figsize=(10, 6), dpi=100)
    
    # 定義一些漂亮的顏色
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    for i in range(len(results)):
        plt.plot(epochs, results[i], label=config_names[i], 
                 color=colors[i], linewidth=2, marker='o', markersize=3, alpha=0.8)

    # 圖表裝飾
    plt.title('CIFAR-10 Test Accuracy across Different Architectures', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', frameon=True, shadow=True)
    
    # 設定 X 軸刻度間隔
    plt.xticks(np.arange(0, 14, 5))
    
    plt.tight_layout()
    
    # 儲存與顯示
    plt.savefig('part4_comparison.png')

if __name__ == "__main__":
    main()
    # p4()