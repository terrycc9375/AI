import torch
from torch.amp import autocast, GradScaler
import torch_geometric
from torch_geometric import (
    data,
    utils,
    nn,
)
import networkx
import pandas
import numpy
import ast
import time
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def main():
    graph = networkx.DiGraph()

    # add nodes
    dataframe = pandas.read_csv("./dataset/train.csv", usecols=["node_id"])
    all_nodes = set(dataframe["node_id"].unique())
    graph.add_nodes_from(all_nodes)
    dataframe = pandas.read_csv("./dataset/test.csv", usecols=["node_id"])
    all_nodes = set(dataframe["node_id"].unique())
    graph.add_nodes_from(all_nodes)
    
    # add edges
    dataframe = pandas.read_csv("dataset/treads_graph.csv")
    graph.add_edges_from(dataframe[['src_node', 'dst_node']].values)

    # build 2D features
    node2id = {node: i for i, node in enumerate(graph.nodes())}
    degree = networkx.degree_centrality(graph)
    clustering = networkx.clustering(graph.to_undirected())
    structural_features = []
    for node in graph.nodes():
        structural_features.append([degree[node], clustering[node]])
    structural_features = torch.tensor(structural_features, dtype=torch.float)
    edge_index = utils.from_networkx(graph).edge_index

    # load 128D features and labels from training set
    dataframe = pandas.read_csv("./dataset/train.csv")
    dataframe["ft_vec"] = dataframe["feature"].apply(lambda ft: numpy.array(ast.literal_eval(ft), dtype=numpy.float32))
    features = dict()
    for _, row in dataframe.iterrows():
        features[row["node_id"]] = row["ft_vec"]
    labels = dict(zip(dataframe["node_id"], dataframe["label"]))

    # load 128D features from testing set
    dataframe = pandas.read_csv("./dataset/test.csv")
    dataframe["ft_vec"] = dataframe["feature"].apply(lambda ft: numpy.array(ast.literal_eval(ft), dtype=numpy.float32))
    for _, row in dataframe.iterrows():
        features[row["node_id"]] = row["ft_vec"]

    num_nodes = graph.number_of_nodes()
    train_features = torch.zeros((num_nodes, 128), dtype=torch.float)
    train_labels = torch.full((num_nodes,), -1, dtype=torch.long)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    for i, node in enumerate(graph.nodes()):
        if node in features:
            train_features[i] = torch.tensor(features[node], dtype=torch.float)
        if node in labels:
            train_labels[i] = labels[node]
            train_mask[i] = True
    x = torch.cat([train_features, structural_features], dim=1)
    train_data = data.Data(
        x=x,
        edge_index=edge_index,
        y=train_labels,
        train_mask=train_mask,
    ).to("cuda")
    model = GAT(
        num_nodes=num_nodes,
        in_channels=130,
        hidden_channels=32,
        out_channels=40,
    ).to("cuda")
    # model = torch.load("model_epoch100.pt").to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.7,
        patience=5,
        min_lr=1e-5,
    )
    criterion = torch.nn.CrossEntropyLoss()
    scaler = GradScaler()
    
    start = time.time()
    epochs = 500
    for epoch in range(epochs):
        # for batch in train_loader:
        model.train()
        optimizer.zero_grad()
        with autocast(device_type="cuda", dtype=torch.float16):
            out = model(train_data.x, train_data.edge_index)
            loss = criterion(out[train_data.train_mask], train_data.y[train_data.train_mask])
        # loss.backward()
        # optimizer.step()
        # scheduler.step(loss)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # if (epoch + 1) > 30 and (epoch + 1) < 40:
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                pred = out.argmax(dim=1)
                correct = (pred[train_data.train_mask] == train_data.y[train_data.train_mask]).sum()
                acc = int(correct) / int(train_data.train_mask.sum())
            end = time.time()
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Train Acc: {acc:.4f}, Time: {end - start:.2f}s")
            start = end
            if acc > 0.7 and acc < 0.8:
                torch.save(model, f"model_epoch{epoch+1}.pt")
                inference(model, train_data, graph.nodes(), epoch)

    print(f"Params: {sum(p.numel() for p in model.parameters())}")

def inference(model, train_data, graph_node, epoch):
    # load model
    # model = torch.load(model_name).to("cuda")
    model.eval()
    with torch.no_grad():
        out = model(train_data.x, train_data.edge_index)
        pred = out.argmax(dim=1).cpu().numpy()
    node_id_list = list(graph_node)
    node_to_pred = {node: pred[i] for i, node in enumerate(node_id_list)}

    # load testing data
    df_test = pandas.read_csv("./dataset/test.csv")
    df_test["label"] = df_test["node_id"].map(lambda node: node_to_pred[node])
    df_test[["node_id", "label"]].to_csv(f"ep{epoch + 1}.csv", index=False)

class GAT(torch.nn.Module):
    def __init__(self, num_nodes, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.node_embeddings = torch.nn.Embedding(num_nodes, 64)
        self.encoder = torch.nn.Linear(in_channels + 64, hidden_channels)
        self.conv1 = nn.GATv2Conv(hidden_channels, 64, heads=8, concat=True, dropout=0.5)
        self.bn1 = nn.BatchNorm(64 * 8)
        self.conv2 = nn.GATv2Conv(64 * 8, 64, heads=8, concat=True, dropout=0.5)
        self.bn2 = nn.BatchNorm(64 * 8)
        self.conv3 = nn.GATv2Conv(64 * 8, out_channels, heads=1, concat=False)

    def forward(self, x, edge_index):
        node_ids = torch.arange(x.size(0), device="cuda")
        node_emb = self.node_embeddings(node_ids)
        x = torch.cat([x, node_emb], dim=1)
        x = self.encoder(x).relu()
        x = self.conv1(x, edge_index)
        x = self.bn1(x).relu()
        x_skip = x
        x = self.conv2(x, edge_index)
        x = self.bn2(x).relu()
        x = x + x_skip
        x = self.conv3(x, edge_index)
        return x
    


if __name__ == "__main__":
    main()
