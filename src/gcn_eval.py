"""
GCN-based concept vector evaluation.
Replaces KNN centroid-averaging with graph neural network classification
on the k-NN graph of raw concept vectors.

No retraining of concept extractors needed — uses existing saved vectors.
Only works for methods returning multiple vectors per concept (CEM, TCAV).

Usage:
    from src.gcn_eval import gcn_classification_accuracy
    acc = gcn_classification_accuracy(load_cem_vectors_simple, dataset, attributes, [43,44,45])
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import kneighbors_graph
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix


def build_knn_graph(vectors, k=15):
    """
    Build a k-NN graph from a matrix of vectors.
    Returns normalized adjacency matrix as a sparse torch tensor.
    """
    n = vectors.shape[0]
    knn = kneighbors_graph(vectors, k, mode='connectivity',
                           include_self=False, metric='cosine')
    adj = knn + knn.T
    adj.data = np.ones_like(adj.data)
    adj = adj.minimum(1.0)

    # Symmetric normalization: D^{-1/2} A D^{-1/2}
    rowsum = np.array(adj.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = csr_matrix(np.diag(d_inv_sqrt))
    adj_norm = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
    return adj_norm


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, adj_norm, h):
        support = h @ self.weight
        out = adj_norm @ support
        return out


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        self.layer1 = GCNLayer(in_dim, hidden_dim)
        self.layer2 = GCNLayer(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, adj_norm, h):
        h = self.layer1(adj_norm, h)
        h = torch.relu(h)
        h = self.dropout(h)
        h = self.layer2(adj_norm, h)
        return torch.log_softmax(h, dim=1)


def train_gcn(model, adj_norm, features, labels, train_mask, val_mask,
              lr=0.01, weight_decay=5e-4, epochs=500, patience=50):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_acc = 0.0
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(adj_norm, features)
        loss = nn.functional.nll_loss(output[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            output = model(adj_norm, features)
            val_acc = (output[val_mask].argmax(1) == labels[val_mask]).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val_acc


def gcn_classification_accuracy(embedding_method, dataset, attributes,
                                random_seeds, k=15, hidden_dim=64,
                                test_size=0.2, pca_dim=128):
    """
    Build a k-NN graph from raw concept vectors, train a GCN to classify
    each vector into its correct concept, and report test accuracy.

    Higher accuracy = concepts are more separable = better concept vectors.

    Note: Requires at least ~100 total vectors across concepts for meaningful
    training. Works well for CEM and other multi-vector methods. Returns NaN
    for methods with too few samples (Label, Concept2Vec, sometimes TCAV).

    Returns:
        (mean_test_acc, std_test_acc) across seeds.
    """
    min_samples = 100

    accuracies = []
    for seed in random_seeds:
        all_vectors = []
        all_labels = []
        for i, attr in enumerate(attributes):
            vectors = embedding_method(attr, dataset, "", seed)
            if len(vectors) == 0:
                continue
            if isinstance(vectors, np.ndarray) and vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)
            all_vectors.append(vectors)
            all_labels.extend([i] * len(vectors))

        if len(all_vectors) < 2:
            continue

        if len(all_labels) < min_samples:
            return float('nan'), float('nan')

        X = np.vstack(all_vectors)
        y = np.array(all_labels)

        # Only evaluate on concepts with >= 2 samples
        unique, counts = np.unique(y, return_counts=True)
        valid_classes = unique[counts >= 2]
        if len(valid_classes) < 2:
            continue
        valid_mask = np.isin(y, valid_classes)
        X = X[valid_mask]
        y = y[valid_mask]
        class_map = {c: i for i, c in enumerate(valid_classes)}
        y = np.array([class_map[l] for l in y])

        # Normalize features
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)

        # PCA reduction for high-dim vectors
        feat_dim = min(pca_dim, X.shape[0] - 1, X.shape[1])
        if X.shape[1] > feat_dim:
            pca = PCA(n_components=feat_dim)
            X = pca.fit_transform(X)

        # Build graph
        adj_sparse = build_knn_graph(X, k=min(k, len(X) - 1))
        adj_dense = torch.FloatTensor(adj_sparse.toarray())

        # Non-stratified split (some classes may have only 2 samples)
        train_val_idx, test_idx = train_test_split(
            np.arange(len(y)), test_size=test_size, random_state=seed
        )
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=0.25, random_state=seed
        )

        features = torch.FloatTensor(X)
        labels = torch.LongTensor(y)

        train_mask = torch.zeros(len(y), dtype=torch.bool)
        val_mask = torch.zeros(len(y), dtype=torch.bool)
        test_mask = torch.zeros(len(y), dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        num_classes = len(valid_classes)
        model = GCN(X.shape[1], hidden_dim, num_classes)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        adj_dense = adj_dense.to(device)
        features = features.to(device)
        labels = labels.to(device)
        train_mask = train_mask.to(device)
        val_mask = val_mask.to(device)
        test_mask = test_mask.to(device)

        _ = train_gcn(model, adj_dense, features, labels, train_mask, val_mask)

        model.eval()
        with torch.no_grad():
            output = model(adj_dense, features)
            test_acc = (output[test_mask].argmax(1) == labels[test_mask]).float().mean().item()

        accuracies.append(test_acc)

    if len(accuracies) == 0:
        return float('nan'), float('nan')
    return float(np.mean(accuracies)), float(np.std(accuracies))
