
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import networkx
from node2vec import Node2Vec
import numpy
import pandas
import torch
import torch_geometric, torch_geometric.utils, torch_geometric.data, torch_geometric.nn, torch_geometric.loader
import ast
import gc
import os
import time
import pickle


class GAT(torch.nn.Module):
    def __init__(self, 
            in_channels, 
            hidden_channels, 
            num_layers, 
            out_channels = 40, 
            dropout = 0.5, 
            heads = 4, 
            act = "relu",
            norm = None, 
            jk = None, 
            **kwargs
        ):
        super().__init__()
        self.gat = torch_geometric.nn.GAT(
            in_channels = in_channels,
            hidden_channels = hidden_channels,
            num_layers = num_layers,
            out_channels = out_channels,
            dropout = dropout,
            heads = heads,
            act = act,
            norm = norm,
            jk = jk,
        )

    def forward(self, data: torch_geometric.data.Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        x = self.gat(x, edge_index)
        return x
    
class SAGE(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_layers,
        out_channels,
        dropout=0.5,
        act="relu",
        norm="BatchNorm",
        jk=None,
    ):
        super().__init__()
        self.linear = torch.nn.Linear(in_channels, hidden_channels)
        # attention = torch_geometric.nn.aggr.AttentionalAggregation(
        #     gate_nn=torch.nn.Sequential(
        #         torch.nn.Linear(hidden_channels, hidden_channels),
        #         torch.nn.ReLU(),
        #         torch.nn.Linear(hidden_channels, 1),
        #     )
        # )
        self.sage = torch_geometric.nn.GraphSAGE(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=out_channels,
            dropout=dropout,
            act=act,
            norm=norm,
            jk=jk,
            aggr="max",
        )
    def forward(self, data: torch_geometric.data.Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        x = self.linear(x).relu()
        x = self.sage(x, edge_index)
        return x

def train(model, data):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    optimizer.zero_grad()
    out = model(data)
    loss = torch.nn.functional.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

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
    gc.collect()

    # debug
    # print(f"Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")

    # create feature embeddings from graph
    step1 = time.time()
    encoder_path = "./node2vec.pkl"
    if os.path.exists(encoder_path):
        with open(encoder_path, "rb") as f:
            embeddings = pickle.load(f)
    else:
        encoder = Node2Vec(
            graph=graph, 
            dimensions=128, 
            walk_length=30, 
            num_walks=20, 
            p=0.8,
            q=1.2,
            workers=4
        )
        n2v = encoder.fit(
            window=10,
            min_count=1,
            batch_words=4,
        )
        embeddings = {node: n2v.wv[str(node)] for node in graph.nodes()}
        with open(encoder_path, "wb") as f:
            pickle.dump(embeddings, f)
        del encoder, n2v
    gc.collect()

    # load another feature and label
    step2 = time.time()
    x_path = "x.pkl"
    dataframe = pandas.read_csv("./dataset/train.csv")
    dataframe["ft_vec"] = dataframe["feature"].apply(lambda ft: numpy.array(ast.literal_eval(ft), dtype=numpy.float32))
    features = dict(zip(dataframe["node_id"], dataframe["ft_vec"]))
    labels = dict(zip(dataframe["node_id"], dataframe["label"]))
    # x_prime = dataframe["ft_vec"].tolist()
    del dataframe

    # concatenate features
    nodes = sorted(graph.nodes())
    num_nodes = len(nodes)
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    if os.path.exists(x_path):
        with open(x_path, "rb") as f:
            x = pickle.load(f)
    else:
        x = numpy.zeros((num_nodes, 256), dtype=numpy.float32)
        for i, node in enumerate(nodes):
            ft1 = features.get(node, numpy.zeros(128, dtype=numpy.float32))
            ft2 = embeddings.get(node, numpy.zeros(128, dtype=numpy.float32))
            x[i, :128] = ft1
            x[i, 128:] = ft2
        ft1 = torch.nn.functional.normalize(
            torch.from_numpy(x[:, :128]),
            p=2,
            dim=1,
        )
        ft2 = torch.nn.functional.normalize(
            torch.from_numpy(x[:, 128:]),
            p=2,
            dim=1,
        )
        x = torch.cat([ft1, ft2], dim=1)
        with open(x_path, "wb") as f:
            pickle.dump(x, f)

    # debug
    # print(f"Feature shape: {x.shape}")
    
    step3 = time.time()
    y = [-1] * num_nodes
    train_mask = [False] * num_nodes
    for node, label in labels.items():
        idx = node_to_idx[node]
        y[idx] = label
        train_mask[idx] = True

    y = torch.tensor(y, dtype=torch.long)
    train_mask = torch.tensor(train_mask, dtype=torch.bool)
    test_mask = ~train_mask
    edge_index = torch_geometric.utils.from_networkx(graph).edge_index
    del features, embeddings, graph
    gc.collect()

    # prepare data
    step4 = time.time()
    data = torch_geometric.data.Data(
        x=x,
        edge_index=edge_index, 
        y=y, 
        train_mask=train_mask, 
        test_mask=test_mask
    )
    data.num_classes = 40
    # print(f"Train: {data.train_mask.sum().item()}, Test: {data.test_mask.sum().item()}")

    
    print(f"load x time: {step3 - step2:.2f}s, load y time: {step4 - step3:.2f}s")

    # train
    # model = GAT(
    #     in_channels=128,
    #     hidden_channels=64,
    #     num_layers=4,
    #     dropout=0.4,
    #     heads=8,
    #     norm="BatchNorm",
    #     jk="max",
    # ).to("cuda")
    model = SAGE(
        in_channels=256,
        hidden_channels=128,
        num_layers=3,
        out_channels=40,
        dropout=0.5,
        jk="mean",
    ).to("cuda")
    data = data.to("cuda")
    # train_loader = torch_geometric.loader.NeighborLoader(
    #     data,
    #     num_neighbors=[15, 10, 5],
    #     batch_size=1024,
    #     input_nodes=data.train_mask,
    #     shuffle=True,
    # )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.7,
        patience=5,
        min_lr=1e-5,
    )
    criterion = torch.nn.CrossEntropyLoss()

    start = time.time()
    epochs = 180
    for epoch in range(epochs):
        # for batch in train_loader:
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        # scheduler.step(loss)


        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                pred = out.argmax(dim=1)
                correct = (pred[data.train_mask] == data.y[data.train_mask]).sum()
                acc = int(correct) / int(data.train_mask.sum())
            end = time.time()
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Train Acc: {acc:.4f}, Time: {end - start:.2f}s")
            start = end

    print(f"Params: {sum(p.numel() for p in model.parameters())}")
    torch.save(model, "model.pt")

    # inference
    model.eval()
    with torch.no_grad():
        df_test = pandas.read_csv("./dataset/test.csv")
        with open("node2vec.pkl", "rb") as f:
            embeddings = pickle.load(f)
        df_test["ft_vec"] = df_test["feature"].apply(lambda ft: numpy.array(ast.literal_eval(ft), dtype=numpy.float32))
        features = dict(zip(df_test["node_id"], df_test["ft_vec"]))
        x_test = list()
        for node in nodes:
            ft1 = features.get(node, numpy.zeros(128, dtype=numpy.float32))
            ft2 = embeddings.get(node, numpy.zeros(128, dtype=numpy.float32))
            x_test.append(numpy.concatenate([ft1, ft2], axis=0))
        x_test = torch.tensor(numpy.stack(x_test, axis=0), dtype=torch.float32)
        test_data = data.clone()
        test_data.x = x_test.to("cuda")
        out = model(test_data)
        pred = out.argmax(dim=1).cpu().numpy()
    test_node_ids = [nodes[i] for i in range(len(nodes)) if test_mask[i]]
    df_submission = pandas.DataFrame({
        "node_id": test_node_ids,
        "label": pred[test_mask.cpu().numpy()]
    })
    df_submission.to_csv(f"ep{epochs}.csv", index=False)

if __name__ == "__main__":
    main()
