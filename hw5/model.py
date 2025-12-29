import torch
import torch_geometric.nn

class GCN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size = 40):
        super().__init__()
        self.conv1 = torch_geometric.nn.GCNConv(input_size, hidden_size)
        self.conv2 = torch_geometric.nn.GCNConv(hidden_size, hidden_size)
        self.conv3 = torch_geometric.nn.GCNConv(hidden_size, hidden_size)

        self.linear1 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        x0 = x # 256
        x = self.conv2(x0, edge_index)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        x0 = x + x0 # 256
        x = self.conv3(x0, edge_index)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        x0 = x + x0 # 256
        x = self.linear1(x0)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.normalize(x)
        x = self.linear2(x)
        return x
