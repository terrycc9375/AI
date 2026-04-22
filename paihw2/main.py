import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.console import Console

console = Console()

class Classifier(nn.Module):
    def __init__(self, model_path=None):
        super(Classifier, self).__init__()
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

def main():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        raise Exception("GPU is not available.")
        exit(0)
        
    # Training arguments
    batch_size = 1024
    learning_rate = 0.001
    epochs = 50
    save_path = "cifar10_model.pth"

    # 1. 載入 CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    model = Classifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    with Progress(
        TextColumn("[#238ce8][progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TextColumn("[#faac2f]Loss: {task.fields[loss]:>6.4f}"),
        TimeRemainingColumn(),
    ) as progress:
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            epoch_task = progress.add_task(
                f"Epoch {epoch+1}/{epochs}", total=len(trainloader), loss=0.0
            )

            for i, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss = loss.item()
                # 更新進度條資訊
                progress.update(epoch_task, advance=1, loss=running_loss)

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