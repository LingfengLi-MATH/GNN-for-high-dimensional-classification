import numpy as np
from sklearn.model_selection import PredefinedSplit
import torch
import dgl
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
import networkx as nx
import scipy.sparse as sp

def split_to_numpy(x, y, split):
    keys = ['train', 'test', 'valid']
    objs = [x, y]
    return [obj[split[key]].detach().cpu().numpy() for obj in objs for key in keys]

def get_predefined_split(x_train, x_val, y_train, y_val, return_array=True):
    test_fold = np.concatenate([-np.ones_like(y_train), np.zeros_like(y_val)])
    ps = PredefinedSplit(test_fold)
    if return_array:
        x = np.concatenate([x_train, x_val], axis=0)
        y = np.concatenate([y_train, y_val], axis=0)
        return ps, [x, y]
    return ps

def get_vector_representations(encoder_model, dataloader, device='cpu'):
    encoder_model.eval()
    x = []
    x1 = []
    y = []
    z = []
    z1 = []
    batch = []
    edges = []
    edges1 = []
    for data in dataloader:
        data = data.to(device)
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=device)
        h, g, h1, h2, g1, g2, e1, e2 = encoder_model(data.x, data.edge_index, data.batch)
        x.append(g.detach())
        x1.append(g1.detach())
        y.append(data.y.to('cpu'))
        z.append(h.detach())
        z1.append(h1.detach())
        batch.append(data.batch)
        edges.append(data.edge_index)
        edges1.append(e1)
    x = torch.cat(x, dim=0).to('cpu')
    x1 = torch.cat(x1, dim=0).to('cpu')
    z = torch.cat(z, dim=0).to('cpu')
    z1 = torch.cat(z1, dim=0).to('cpu')
    batch = torch.cat(batch, dim=0).to('cpu')
    edges = torch.cat(edges, dim=1).to('cpu')
    edges1 = torch.cat(edges1, dim=1).to('cpu')
    return {
        'x': x, 'y':y[0], 'z':z, 'batch':batch, 'edges':edges, 'x1':x1, 'z1':z1, 'edges1':edges1,
        }


@torch.no_grad()
def get_image_vector_representations(encoder_model, dataloader, device='cpu', noise_std=0.2):
    encoder_model.eval()
    x = []
    x1 = []
    y = []
    for images, labels in dataloader:
        images = images.to(device)
        noisy_images = torch.clamp(images + noise_std * torch.randn_like(images), 0.0, 1.0)
        feat = encoder_model(images)
        feat_noisy = encoder_model(noisy_images)
        x.append(feat.detach().cpu())
        x1.append(feat_noisy.detach().cpu())
        y.append(labels.cpu())

    x = torch.cat(x, dim=0)
    x1 = torch.cat(x1, dim=0)
    y = torch.cat(y, dim=0)
    return {
        'x': x,
        'x1': x1,
        'y': y,
    }





def construct_dgl_graph(vectors, labels, left=None, method='knn', k=10, threshold=0.8, metric='cosine'):
    """
    Construct a DGL graph representation from high-dimensional vectors.
    
    Parameters:
    - vectors: numpy array of shape (n_samples, n_features)
    - method: 'knn' for k-nearest neighbors, 'threshold' for distance threshold graph
    - k: number of nearest neighbors for 'knn' method
    - threshold: distance threshold for 'threshold' method
    - metric: distance metric ('euclidean', 'cosine', etc.)
    
    Returns:
    - DGL graph object with node features
    """
    n_vectors = vectors.shape[0]
    
    # Convert vectors to torch tensors for node features
    node_features = vectors.clone().detach()
    
    if method == 'knn':
        # Construct k-nearest neighbors graph
        adj_matrix = kneighbors_graph(vectors, n_neighbors=k, metric=metric, mode='distance')
        adj_matrix = adj_matrix.maximum(adj_matrix.T)
        # Convert to COO format (source, destination) pairs
        coo_matrix = adj_matrix.tocoo()
        src_nodes = torch.LongTensor(coo_matrix.row)
        dst_nodes = torch.LongTensor(coo_matrix.col)
        
    elif method == 'threshold':
        # Calculate pairwise distances
        adj_matrix = pairwise_distances(vectors, metric=metric)
        
        # Find all pairs below threshold
        src_nodes = []
        dst_nodes = []
        for i in range(n_vectors):
            for j in range(n_vectors):
                if i != j and adj_matrix[i,j] <= threshold:
                    adj_matrix[i,j] = 0
                else:
                    src_nodes.append(i)
                    dst_nodes.append(j)
        adj_matrix = sp.csr_matrix(adj_matrix)
        coo_matrix = adj_matrix.tocoo()
        src_nodes = torch.LongTensor(coo_matrix.row)
        dst_nodes = torch.LongTensor(coo_matrix.col)
        
    else:
        raise ValueError("Method must be either 'knn' or 'threshold'")
    
    
    # Create DGL graph
    g = dgl.graph((src_nodes, dst_nodes), num_nodes=n_vectors)
    
    # Add node features
    g.ndata['feat'] = node_features
    g.ndata['label'] = labels
    
    # Compute weights for threshold method
    weights = torch.FloatTensor([adj_matrix[i,j] for i,j in zip(src_nodes, dst_nodes)])
    # g.edata['weight'] = weights
    g.adj_mat = adj_matrix
    g.Lap = sparse_mx_to_torch_sparse_tensor(GraphLaplacian(adj_matrix))
    
    # visualize_dgl_graph(g, labels)
    if left is None:
        return g
    else:
        gs = dgl.node_subgraph(g, left, relabel_nodes=True)
        r, c = gs.adj().indices()[0], gs.adj().indices()[1]
        v = torch.ones_like(r)
        adj_sub = sp.coo_matrix((v, (r, c)), shape=(len(left), len(left)))
        gs.Lap = sparse_mx_to_torch_sparse_tensor(GraphLaplacian(adj_sub))
        return g,gs

def visualize_dgl_graph(g, y):
    nx_g = g.to_networkx(node_attrs=['feat'])
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(nx_g)
    nx.draw_networkx(nx_g, pos, with_labels=False, node_color=y, node_size=100)
    plt.show()

    
def GraphLaplacian(adj, symmetric=False):
   #print('comput graph Laplacian .... ')
   # adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1)).flatten()
   if symmetric:
       d_inv_sqrt = np.power(row_sum, -0.5).flatten()
       d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
       d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
       res = (sp.eye(len(d_inv_sqrt)) - d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)).tocoo()
   else:
       d_inv = 1 / row_sum
       d_inv[np.isinf(d_inv)] = 0.
       d_mat_inv = sp.diags(d_inv)
       res = (sp.eye(len(d_inv)) - d_mat_inv.dot(adj)).tocoo()
   return res

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def DatasetSplit(num_samples, train_ratio=0.4, val_ratio=0.3, k=10):
    assert train_ratio + val_ratio < 1
    train_size = int(num_samples * train_ratio)
    val_size = int(num_samples * val_ratio)
    splits = []
    for i in range(k):
        indices = torch.randperm(num_samples)
        split = {
            'train': indices[:train_size],
            'valid': indices[train_size: val_size + train_size],
            'test': indices[val_size + train_size:]
        }
        splits.append(split)
    return splits


