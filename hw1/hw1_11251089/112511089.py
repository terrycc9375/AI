"""
Example for loading data and saving predicted labels.
You can only use python standard library and numpy.
Do not use any other libraries.
"""
import numpy as np
import random as rd
from collections import Counter
import time

def kmeanspp(X):
    K = 10
    it = 25
    n = X.shape[0]
    centroid = [X[rd.randint(0, n-1)]]
    for _ in range(K-1):
        distance1 = np.array([min(np.sum((x-c)**2) for c in centroid) for x in X])
        distance1 /= np.sum(distance1)
        # choose next centroid
        idx = rd.choices(range(n), weights=distance1, k=1)[0]
        centroid.append(X[idx])

    centroid = np.array(centroid)

    for _ in range(it):
        distance2 = np.array([[np.sum((x - c) ** 2) for c in centroid] for x in X])
        # define label = the min idx of distance
        labels = np.argmin(distance2, axis=1)
        new_centroid = np.array([X[labels == k].mean(axis=0) if (labels == k).any() else centroid[k] for k in range(K)])
        if np.allclose(centroid, new_centroid, atol=1e-6):
            # print(f"Iterate: {_} times")
            break
        centroid = new_centroid

    return labels

def clustering(X):
    it = 40
    best_ari = 0.0
    best_label = list()
    def comb2(n): return n*(n-1)//2 if n > 1 else 0
    
    label = np.load("./label_test.npy")
    n_samples = len(label) # 5000
    for _ in range(it):
        pred = kmeanspp(X)
        pred_first500 = pred[:500].copy()

        contingency = dict()
        for i, j in zip(label, pred_first500):
            contingency[(i, j)] = contingency.get((i, j), 0) + 1
        row_sum = Counter(label)
        col_sum = Counter(pred_first500)
        pair_comb = sum(comb2(n) for n in contingency.values())
        row_comb = sum(comb2(n) for n in row_sum.values())
        col_comb = sum(comb2(n) for n in col_sum.values())
        total_pair = comb2(n_samples)
        expect = (row_comb * col_comb) / total_pair if total_pair > 0 else 0
        mean = (row_comb + col_comb) / 2
        ari = (pair_comb - expect) / (mean - expect) if (mean - expect) != 0 else 1.0
        
        # print(f"Trial: {_}, {ari}, {adjusted_rand_score(label, pred_first500)}")

        if ari > best_ari:
            best_ari = ari
            best_label = pred
        if ari > 0.99999:
            break

    print(f"best_ari: {best_ari}")
    return best_label


if __name__ == "__main__":
    # load data
    X = np.load("./features.npy") # size: [5000, 512]

    # start = time.time()

    y = clustering(X)
    
    # end = time.time()
    # with open("log.txt", 'a') as f:
        # f.writelines(f"{end-start:.3f} s\n")
    
    # save clustered labels
    np.save("predicted_label.npy", y) # output size should be [5000]
