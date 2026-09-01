import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F
import numpy as np
from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from GCL.eval import get_split, SVMEvaluator
from GCL.models import DualBranchContrast
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import DataLoader
from torch.utils.data import Subset
from torch_geometric.datasets import TUDataset
from GNN_models import GConv, Encoder, ProjectionHead

def SSPLoss(g1, g2, pos_mask2=None, neg_mask2=None):
    num_nodes = g1.size(0)
    device = g1.device
    pos_mask1 = torch.eye(num_nodes, dtype=torch.float32, device=device)
    neg_mask1 = 1. - pos_mask1
    loss = L.InfoNCE(tau=0.2)
    l1 = loss(anchor=g1, sample=g2, pos_mask=pos_mask1, neg_mask=neg_mask1)
    l2 = loss(anchor=g2, sample=g1, pos_mask=pos_mask1, neg_mask=neg_mask1)
    total_loss = 0.5 * (l1 + l2)
    
    if pos_mask2 is not None:
        l3 = loss(anchor=g1, sample=g2, pos_mask=pos_mask2, neg_mask=neg_mask2)
        l4 = loss(anchor=g2, sample=g1, pos_mask=pos_mask2, neg_mask=neg_mask2)
        total_loss = 0.25 * (l3 + l4) + 0.5 * total_loss
    return total_loss

def train(encoder_model, head, contrast_model, dataloader, optimizer, pos_mask=None, neg_mask=None, device='cpu'):
    encoder_model.train()
    epoch_loss = 0
    idx_start = 0
    for setp, data in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()

        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        
        _, _, _, _, g1, g2, _, _ = encoder_model(data.x, data.edge_index, data.batch)
        g1, g2 = [head(g) for g in [g1, g2]]
        loss = contrast_model(g1=g1, g2=g2, batch=data.batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss

# def splitdata(dataloader, device='cpu'):

def test(encoder_model, dataloader, split=None, device='cpu'):
    encoder_model.eval()
    x = []
    y = []
    for data in dataloader:
        data = data.to(device)
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        _, g, _, _, _, _, _, _ = encoder_model(data.x, data.edge_index, data.batch)
        x.append(g)
        y.append(data.y)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)

    if split is None:
        split = get_split(num_samples=x.size()[0], train_ratio=0.1, test_ratio=0.8)
    result = SVMEvaluator(linear=True)(x, y, split)
    return result

def generate_mask(split, y):
    n = y.size()[0]
    train_idx = split['train']
    pos_M = torch.zeros(n,n)
    neg_M = torch.ones(n,n) - torch.eye(n)
    for i in range(n):
        for j in range(n):
            if i in train_idx and j in train_idx: 
                if y[i]==y[j]:
                    pos_M[i,j] = 1  
                    neg_M[i,j] = 0
    return pos_M, neg_M

def GCL_training(graph_dataset, left_in, file_path, hdim=32, device='cpu'):
    # device = torch.device('cuda:0')
    graph_subset = Subset(graph_dataset, left_in)
    
    epochs = 2000
    hdim = 32
    
    
    # split = get_split(num_samples=graph_dataset.len(), train_ratio=0.05, test_ratio=0.05)
    # pos_mask, neg_mask = generate_mask(split, dataset.y)
    
    dataloader = DataLoader(graph_subset, batch_size=128, shuffle=False)
    input_dim = max(graph_dataset.num_node_features, 1)
    
    aug1 = A.Identity()
    aug2 = A.RandomChoice([A.RWSampling(num_seeds=1000, walk_length=10),
                           A.NodeDropping(pn=0.1),
                           A.FeatureMasking(pf=0.1),
                           A.EdgeRemoving(pe=0.1)], 1)
    gconv = GConv(input_dim=input_dim, hidden_dim=hdim, num_layers=2).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2)).to(device)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='G2G').to(device)
    head = ProjectionHead(hdim*2, hdim*2).to(device)
    optimizer = Adam(list(encoder_model.parameters())+list(head.parameters()), lr=0.001)
    
    with tqdm(total=epochs, desc='(T)') as pbar:
        for epoch in range(1, epochs+1):
            loss = train(encoder_model, head, contrast_model, dataloader, optimizer, device=device)
            pbar.set_postfix({'loss': loss})
            pbar.update()
    torch.save(encoder_model.state_dict(), file_path)
    test_result = test(encoder_model, dataloader, device=device)
    print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')


if __name__ == '__main__':
    dataset = 'MUTAG'
    hdim = 16
    path = osp.join(osp.expanduser('~'), 'datasets')
    file_path = 'models/'+dataset+'_encoder_hdim{}.pth'.format(hdim)
    if osp.isfile(file_path):
        print('graph encoder exists.')
    else:
        print('start training encoder by graph contrastive learning.')
    # dataset_name = 'IMDB-MULTI'
        graph_dataset = TUDataset(path, name=dataset, use_node_attr=True)
        indices = [i for i in range(graph_dataset.len())]
        np.random.seed(0)
        np.random.shuffle(indices)
        left_in = indices[0:int(len(indices)*0.8)]
        hold_out = indices[int(len(indices)*0.8):]
        GCL_training(graph_dataset, left_in, file_path)