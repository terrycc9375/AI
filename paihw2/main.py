import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.console import Console

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
    else:
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

if __name__ == "__main__":
    main()