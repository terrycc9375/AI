import torch
import torch_geometric.data, torch_geometric.utils, torch_geometric.transforms
import pandas
import numpy
import ast
import time
import os
import rich.console
from model import GCN


def fearture_parser(ft):
    innate_feature = numpy.array(ast.literal_eval(ft))
    return innate_feature
    return numpy.append(innate_feature, [innate_feature.mean(), innate_feature.std()])

def train():
    edge_df = pandas.read_csv("./dataset/treads_graph.csv")
    train_df = pandas.read_csv('./dataset/train.csv')
    test_df = pandas.read_csv('./dataset/test.csv')

    all_nodes = pandas.concat([train_df, test_df], ignore_index=True)
    all_nodes = all_nodes.sort_values('node_id').drop_duplicates('node_id')

    node_id_map = {old_id: new_id for new_id, old_id in enumerate(all_nodes['node_id'].unique())}
    all_nodes['new_node_id'] = all_nodes['node_id'].map(node_id_map)
    train_df['new_node_id'] = train_df['node_id'].map(node_id_map)
    test_df['new_node_id'] = test_df['node_id'].map(node_id_map)
    num_nodes = len(all_nodes)

    x = torch.tensor(numpy.stack(all_nodes["feature"].apply(fearture_parser).values), dtype=torch.float)

    y = torch.full((num_nodes,), -1, dtype=torch.long)
    train_indices = torch.tensor(train_df['new_node_id'].values, dtype=torch.long)
    y[train_indices] = torch.tensor(train_df['label'].values, dtype=torch.long)

    edge_df = edge_df[edge_df["src_node"].isin(node_id_map) & edge_df["dst_node"].isin(node_id_map)]
    src = edge_df["src_node"].map(node_id_map).values
    dst = edge_df["dst_node"].map(node_id_map).values
    edge_index = torch.tensor(numpy.stack([src, dst]), dtype=torch.long).to("cuda")
    edge_index = torch_geometric.utils.to_undirected(edge_index)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_indices] = True

    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_indices = torch.tensor(test_df['new_node_id'].values, dtype=torch.long)
    test_mask[test_indices] = True

    data = torch_geometric.data.Data(x=x, y=y, edge_index=edge_index, train_mask=train_mask, test_mask=test_mask).to("cuda")
    # cross validation
    split: torch_geometric.data.Data = torch_geometric.transforms.RandomNodeSplit(
        split="train_rest",
        num_val=0.2,
        num_test=0,
        key=None,
    )
    data = split(data)
    data.train_mask = data.train_mask & train_mask
    data.val_mask = data.val_mask & train_mask

    model = GCN(
        input_size=128,
        hidden_size=256,
    ).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.7,
        patience=10,
        min_lr=1e-5,
    )
    criterion = torch.nn.CrossEntropyLoss()

    console = rich.console.Console()
    epochs = 300
    best_acc = 0.0
    for epoch in range(epochs):
        
        # prepare
        model.train()
        optimizer.zero_grad()
        train_start = time.time()

        out = model(data.x, data.edge_index)

        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        training_time = time.time() - train_start

        # validation
        model.eval()
        val_start = time.time()
        with torch.no_grad():
            pred = out.argmax(dim=1)
            val_acc = (pred[data.val_mask] == data.y[data.val_mask]).sum() / data.val_mask.sum()
        val_time = time.time() - val_start

        # summary
        console.print(f"Epoch {epoch+1:02d}/{epochs}\t[#e3eb73]Train: {training_time:.2f}[/#e3eb73]\t[#73eba3]Validate: {val_time:.2f}[/#73eba3]\t[#e37ff0]Accuracy: {val_acc}[/#e37ff0]")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model, "./112511089.pt")
        
    console.print(f"Params: {sum(p.numel() for p in model.parameters())}")

    # inference
    del model
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    model = torch.load("./112511089.pt", weights_only=False).to("cuda")
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1).cpu()
    df_test = pandas.read_csv("./dataset/test.csv")
    node_ids = df_test["node_id"].map(node_id_map).values
    pred_test = pred[node_ids].numpy()

    try:
        submission = pandas.DataFrame({
            "node_id": df_test["node_id"],
            "label":   pred_test
        })
        submission.to_csv("submission.csv", index=False)
    except:
        print(len(node_ids), len(pred_test))



def main():
    train()

if __name__ == "__main__":
    main()
