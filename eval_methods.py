import dgl.nn 
import dgl.data 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from GCL.eval import get_split
import GCL.augmentors as A
import GCL.losses as L
from GCL.models import DualBranchContrast
import numpy as np
import argparse
from torch.optim.lr_scheduler import StepLR, CyclicLR
from sklearn.svm import LinearSVC, SVC
from tqdm import tqdm
from GNN_models import GConv, Encoder, ProjectionHead, GraphPotts, GraphSAGE
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.metrics import f1_score
from utils import split_to_numpy, get_predefined_split, get_vector_representations, construct_dgl_graph, visualize_dgl_graph, DatasetSplit
from torch.utils.data import Subset
import copy
######################################################################
######################################################################

def eval_baseline(dataset, split, left, hold, input_dim, aug1=None, epochs=200, device='cpu'):
    dataset_left = Subset(dataset, left)
    dataset_hold = Subset(dataset, hold)
    train_set = Subset(dataset_left, split['train'])
    valid_set = Subset(dataset_left, split['valid'])
    test_set = Subset(dataset_left, split['test'])
    train_loader = DataLoader(train_set, batch_size=12, shuffle=False)
    valid_loader = DataLoader(valid_set, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)
    holdout_loader = DataLoader(dataset_hold, batch_size=128, shuffle=False)
    num_classes = torch.unique(dataset.y).size()[0]
    gconv = GConv(input_dim=input_dim, hidden_dim=128, num_layers=2).to(device)
    head = ProjectionHead(128*2, num_classes).to(device)
    optimizer = torch.optim.Adam(list(gconv.parameters())+list(head.parameters()), lr=1e-2)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.9)
    print('number of parameters: {}'.format(sum(p.numel() for p in gconv.parameters() if p.requires_grad)+sum(p.numel() for p in head.parameters() if p.requires_grad)))
    best_valid_acc = 0.
    best_test_acc = 0.
    with tqdm(total=epochs, desc='(T)') as pbar:
        for e in range(epochs):
            gconv.train()
            for setp, data in enumerate(train_loader):
                data = data.to(device)
                optimizer.zero_grad()
        
                if data.x is None:
                    num_nodes = data.batch.size(0)
                    data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
                
                _, g = gconv(data.x, data.edge_index, data.batch)
                g = head(g)
                loss = F.cross_entropy(g, data.y)
                loss.backward()
                optimizer.step()
                scheduler.step()
            
            gconv.eval()
            valid_acc = 0.
            for setp, data in enumerate(valid_loader):
                data = data.to(device)
        
                if data.x is None:
                    num_nodes = data.batch.size(0)
                    data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
                
                _, g = gconv(data.x, data.edge_index, data.batch)
                g = head(g)
                pred = g.argmax(1)
                valid_acc += (pred==data.y).sum()
            valid_acc /= len(split['valid'])
            
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                test_acc = 0.
                for setp, data in enumerate(test_loader):
                    data = data.to(device)
            
                    if data.x is None:
                        num_nodes = data.batch.size(0)
                        data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)                            
                    _, g = gconv(data.x, data.edge_index, data.batch)
                    g = head(g)
                    pred = g.argmax(1)
                    test_acc += (pred==data.y).sum()
                best_test_acc = test_acc / len(split['test'])
                best_gconv, best_head = copy.deepcopy(gconv), copy.deepcopy(head)
            
            pbar.set_postfix({'loss': loss.item(),'best_acc': best_test_acc})
            pbar.update()

    gconv.eval()
    test_acc_1 = 0.
    test_acc_2 = 0.
    for setp, data in enumerate(holdout_loader):
        data = data.to(device)
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        
        _, g = best_gconv(data.x, data.edge_index, data.batch)
        g = best_head(g)
        pred = g.argmax(1)
        test_acc_1 += (pred==data.y).sum()
        
        data.x, data.edge_index, _ = aug1(data.x, data.edge_index)
        _, g = best_gconv(data.x, data.edge_index, data.batch)
        g = best_head(g)
        pred = g.argmax(1)
        test_acc_2 += (pred==data.y).sum()
    test_acc_1 = test_acc_1 / len(hold)
    test_acc_2 = test_acc_2 / len(hold)
    # print('prediction accuracy on the testing set:{}'.format(best_test_acc))
    # print('prediction accuracy on the hold-out set:{}'.format(test_acc_1))
    # print('prediction accuracy on the noisy hold-out set:{}'.format(test_acc_2))
    return best_test_acc.to('cpu'), test_acc_1.to('cpu'), test_acc_2.to('cpu')


def eval_SupCon(dataset, split, left, hold, input_dim, aug1=None, epochs=200, device='cpu'):
    dataset_left = Subset(dataset, left)
    dataset_hold = Subset(dataset, hold)
    train_set = Subset(dataset_left, split['train'])
    valid_set = Subset(dataset_left, split['valid'])
    test_set = Subset(dataset_left, split['test'])
    train_loader = DataLoader(train_set, batch_size=128, shuffle=False)
    valid_loader = DataLoader(valid_set, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)
    holdout_loader = DataLoader(dataset_hold, batch_size=128, shuffle=False)
    num_classes = torch.unique(dataset.y).size()[0]
    gconv = GConv(input_dim=input_dim, hidden_dim=128, num_layers=2).to(device)
    head_CE = ProjectionHead(128*2, num_classes).to(device)
    head_CL = ProjectionHead(128*2, 128*2).to(device)
    optimizer = torch.optim.Adam(list(gconv.parameters())+list(head_CE.parameters())+list(head_CL.parameters()), lr=1e-2)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.9)
    print('number of parameters: {}'.format(sum(p.numel() for p in gconv.parameters() if p.requires_grad)+sum(p.numel() for p in head_CE.parameters() if p.requires_grad)+sum(p.numel() for p in head_CL.parameters() if p.requires_grad)))
    best_valid_acc = 0.
    best_test_acc = 0.
    
    aug = A.RandomChoice([A.RWSampling(num_seeds=1000, walk_length=10),
                           A.NodeDropping(pn=0.1),
                           A.FeatureMasking(pf=0.1),
                           A.EdgeRemoving(pe=0.1)], 1)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='G2G').to(device)
    
    with tqdm(total=epochs, desc='(T)') as pbar:
        for e in range(epochs):
            gconv.train()
            for setp, data in enumerate(train_loader):
                data = data.to(device)
                optimizer.zero_grad()
        
                if data.x is None:
                    num_nodes = data.batch.size(0)
                    data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=device)
                    
                x_aug, edge_index_aug, edge_weight_aug = aug(data.x, data.edge_index)
                
                _, g = gconv(data.x, data.edge_index, data.batch)
                _, g_aug = gconv(x_aug, edge_index_aug, data.batch)
                extra_pos_mask = torch.eq(data.y, data.y.unsqueeze(dim=1)).to(device)
                extra_pos_mask.fill_diagonal_(False)
                extra_neg_mask = torch.ne(data.y, data.y.unsqueeze(dim=1)).to(device)
                extra_neg_mask.fill_diagonal_(False)
                
                g1 = head_CL(g)
                g2 = head_CL(g_aug)
                loss_CL = contrast_model(g1=g1, g2=g2, batch=data.batch, extra_pos_mask=extra_pos_mask, extra_neg_mask=extra_neg_mask)
                
                g_CE = head_CE(g)
                loss_CE = F.cross_entropy(g_CE, data.y)
                
                loss = 0.5 * loss_CL + 0.5 * loss_CE
                loss.backward()
                optimizer.step()
                scheduler.step()
            
            gconv.eval()
            valid_acc = 0.
            for setp, data in enumerate(valid_loader):
                data = data.to(device)
        
                if data.x is None:
                    num_nodes = data.batch.size(0)
                    data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
                
                _, g = gconv(data.x, data.edge_index, data.batch)
                g = head_CE(g)
                pred = g.argmax(1)
                valid_acc += (pred==data.y).sum()
            valid_acc /= len(split['valid'])
            
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                test_acc = 0.
                for setp, data in enumerate(test_loader):
                    data = data.to(device)
            
                    if data.x is None:
                        num_nodes = data.batch.size(0)
                        data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)                            
                    _, g = gconv(data.x, data.edge_index, data.batch)
                    g = head_CE(g)
                    pred = g.argmax(1)
                    test_acc += (pred==data.y).sum()
                best_test_acc = test_acc / len(split['test'])
                best_gconv, best_head = copy.deepcopy(gconv), copy.deepcopy(head_CE)
            
            pbar.set_postfix({'loss': loss.item(),'best_acc': best_test_acc})
            pbar.update()

    gconv.eval()
    test_acc_1 = 0.
    test_acc_2 = 0.
    for setp, data in enumerate(holdout_loader):
        data = data.to(device)
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        
        _, g = best_gconv(data.x, data.edge_index, data.batch)
        g = best_head(g)
        pred = g.argmax(1)
        test_acc_1 += (pred==data.y).sum()
        
        data.x, data.edge_index, _ = aug1(data.x, data.edge_index)
        _, g = best_gconv(data.x, data.edge_index, data.batch)
        g = best_head(g)
        pred = g.argmax(1)
        test_acc_2 += (pred==data.y).sum()
    test_acc_1 = test_acc_1 / len(hold)
    test_acc_2 = test_acc_2 / len(hold)
    # print('prediction accuracy on the testing set:{}'.format(best_test_acc))
    # print('prediction accuracy on the hold-out set:{}'.format(test_acc_1))
    # print('prediction accuracy on the noisy hold-out set:{}'.format(test_acc_2))
    return best_test_acc.to('cpu'), test_acc_1.to('cpu'), test_acc_2.to('cpu')

def eval_svm(rep, left_in, hold_out, split, device='cpu'):
    evaluator = SVC(max_iter=10000)
    x, y = rep['x'][left_in], rep['y'][left_in]
    x_train, x_test, x_val, y_train, y_test, y_val = split_to_numpy(x, y, split)
    ps, [x_train, y_train] = get_predefined_split(x_train, x_val, y_train, y_val)
    params = {'C': [0.001,0.01,0.1,1.,10.,100.]}
    classifier = GridSearchCV(evaluator, params, cv=ps, scoring='accuracy', verbose=0)
    classifier.fit(x_train, y_train) 
    prediction = classifier.predict(x_test)
    # test_macro = f1_score(y_test, prediction, average='macro')
    # test_micro = f1_score(y_test, prediction, average='micro')
    acc = (y_test==prediction).mean()
    # print('prediction accuracy on the testing set:{}'.format(acc))
    
    prediction = classifier.predict(rep['x'][hold_out].cpu())
    acc1 = (rep['y'][hold_out].cpu().numpy()==prediction).mean()
    # print('prediction accuracy on the hold-out set:{}'.format(acc1))
    
    
    prediction = classifier.predict(rep['x1'][hold_out].cpu())
    acc2 = (rep['y'][hold_out].cpu().numpy()==prediction).mean()
    # print('prediction accuracy on the noisy hold-out set:{}'.format(acc2))
    
    return acc, acc1, acc2

def eval_GNN(gs, gf, gfn, split, epochs=200, device='cpu'):
    N = gs.num_nodes()
    indices = np.array([i for i in range(N)])
    features = gs.ndata["feat"]
    edges = gs.edges()
    edges = torch.cat((edges[0].unsqueeze(0),edges[1].unsqueeze(0)))
    labels = gs.ndata["label"]
    num_class = torch.unique(labels).size()[0]
    train_mask = split['train']
    val_mask = split['valid']
    test_mask = split['test']
    
    feat_dim = features.shape[1]
    ######################################################################
    ######################################################################
    # Create the model with given dimensions
    # model = GConv(input_dim=feat_dim, hidden_dim=128, num_layers=2).to(device)
    model = GraphSAGE(input_dim=feat_dim, hidden_dim=128, num_layers=2).to(device)
    head = ProjectionHead(input_dim=256, num_classes=num_class).to(device)
    
    optimizer = torch.optim.Adam(list(model.parameters())+list(head.parameters()), lr=1e-1)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.9)
    # print('number of parameters: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)+sum(p.numel() for p in head.parameters() if p.requires_grad)))
    ######################################################################
    ######################################################################

    best_val_acc = 0
    best_test_acc = 0
    min_loss = 1000.
    with tqdm(total=epochs, desc='(T)') as pbar:
        for e in range(epochs):
            model.train()
            logits,_ = model(features, edges, torch.zeros(features.size()[0]).to(torch.int64).to(device))
            logits = head(logits)
            # Compute prediction
            pred = logits.argmax(1)
            loss = F.cross_entropy(logits[train_mask], labels[train_mask])
    
            # Compute accuracy on training/validation/test
            train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
            model.eval()
            logits,_ = model(features, edges, torch.zeros(features.size()[0]).to(torch.int64).to(device))
            logits = head(logits)
            pred = logits.argmax(1)
            val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
            test_acc = (pred[test_mask] == labels[test_mask]).float().mean()
            
            if best_val_acc < val_acc:
                best_model = copy.deepcopy(model)
                best_head = copy.deepcopy(head)
                best_val_acc = val_acc
                min_loss = loss
                best_test_acc = test_acc
    
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            pbar.set_postfix({'loss': loss.item(),'test_acc': test_acc,'best_acc': best_test_acc})
            pbar.update()

    # print('prediction accuracy on the testing set:{}'.format(best_test_acc))
    
    features, edges, labels = gf.ndata["feat"], gf.edges(), gf.ndata["label"]
    edges = torch.cat((edges[0].unsqueeze(0),edges[1].unsqueeze(0)))
    best_model.eval()
    logits,_ = best_model(features, edges, torch.zeros(features.size()[0]).to(torch.int64).to(device))
    logits = best_head(logits)
    pred = logits.argmax(1)
    test_acc1 = (pred[gf.hold] == labels[gf.hold]).float().mean()
    # print('prediction accuracy on the hold-out set:{}'.format(test_acc1))
    
    
    features, edges, labels = gfn.ndata["feat"], gfn.edges(), gfn.ndata["label"]
    edges = torch.cat((edges[0].unsqueeze(0),edges[1].unsqueeze(0)))
    best_model.eval()
    logits,_ = best_model(features, edges, torch.zeros(features.size()[0]).to(torch.int64).to(device))
    logits = best_head(logits)
    pred = logits.argmax(1)
    test_acc2 = (pred[gfn.hold] == labels[gfn.hold]).float().mean()
    # print('prediction accuracy on the noisy hold-out set:{}'.format(test_acc2))
    
    return best_test_acc.to('cpu'), test_acc1.to('cpu'), test_acc2.to('cpu')
    


def eval_GPN(gs, gf, gfn, split, epochs=200, device='cpu'):
    N = gs.num_nodes()
    indices = np.array([i for i in range(N)])
    features = gs.ndata["feat"]
    edges = gs.edges()
    edges = torch.cat((edges[0].unsqueeze(0),edges[1].unsqueeze(0)))
    labels = gs.ndata["label"]
    num_class = torch.unique(labels).size()[0]
    train_mask = split['train']
    val_mask = split['valid']
    test_mask = split['test']
    labeled_data = torch.cat((torch.tensor(train_mask).unsqueeze(0), labels[train_mask].unsqueeze(0)))
    feat_dim = features.shape[1]
    # etas = [0.1 if args.aug==1 else 0.1]
    etas = [1.]
    best_valid_acc_array = []
    best_test_acc_array = []
    ######################################################################
    ######################################################################
    # Create the model with given dimensions
    for eta in etas:
        model = GraphPotts(input_dim=feat_dim, num_classes=num_class, depth=6, eta=eta).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
        # print('number of parameters: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
        # scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, step_size_up=50, cycle_momentum=False)
        scheduler = StepLR(optimizer, step_size=50, gamma=0.9)
    
        best_val_acc = 0
        best_test_acc = 0
        min_loss = 1000
        with tqdm(total=epochs, desc='(T)') as pbar:
            for e in range(epochs):
                model.train()
                logits = model(features, edges, torch.zeros(features.size()[0]).to(torch.int64).to(device), gs.rep, gs.Lap.to(device))
                # Compute prediction
                pred = logits.argmax(1)
                loss = F.cross_entropy(logits[train_mask], labels[train_mask])
                
                model.eval()
                logits = model(features, edges, torch.zeros(features.size()[0]).to(torch.int64).to(device), gs.rep, gs.Lap.to(device), labeled_data)
                pred = logits.argmax(1)
                val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
                test_acc = (pred[test_mask] == labels[test_mask]).float().mean()
                

                if best_val_acc < val_acc or (best_val_acc==val_acc and loss<min_loss):
                    best_val_acc = val_acc
                    min_loss = loss
                    best_test_acc = test_acc
                    best_model = copy.deepcopy(model)
        
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                pbar.set_postfix({'loss': loss.item(),'test_acc': test_acc,'best_acc': best_test_acc})
                pbar.update()
    # print('prediction accuracy on the testing set:{}'.format(best_test_acc))
    
    features, edges, labels = gf.ndata["feat"], gf.edges(), gf.ndata["label"]
    edges = torch.cat((edges[0].unsqueeze(0),edges[1].unsqueeze(0)))
    best_model.eval()
    # labeled_data = torch.cat((torch.tensor(gf.hold).unsqueeze(0), labels[gf.hold].unsqueeze(0)))
    logits = best_model(features, edges, torch.zeros(features.size()[0]).to(torch.int64).to(device), gf.rep, gf.Lap.to(device))
    pred = logits.argmax(1)
    test_acc1 = (pred[gf.hold] == labels[gf.hold]).float().mean()
    # print('prediction accuracy on the hold-out set:{}'.format(test_acc1))
    
    
    features, edges, labels = gfn.ndata["feat"], gfn.edges(), gfn.ndata["label"]
    edges = torch.cat((edges[0].unsqueeze(0),edges[1].unsqueeze(0)))
    # labeled_data = torch.cat((torch.tensor(gfn.hold).unsqueeze(0), labels[gfn.hold].unsqueeze(0)))
    best_model.eval()
    logits = best_model(features, edges, torch.zeros(features.size()[0]).to(torch.int64).to(device), gfn.rep, gfn.Lap.to(device))
    pred = logits.argmax(1)
    test_acc2 = (pred[gfn.hold] == labels[gfn.hold]).float().mean()
    # print('prediction accuracy on the noisy hold-out set:{}'.format(test_acc2))
    return best_test_acc.to('cpu'), test_acc1.to('cpu'), test_acc2.to('cpu')

from sklearn.base import BaseEstimator, ClassifierMixin


class GraphMBO(BaseEstimator, ClassifierMixin):
    def __init__(self, mu=100., dt=0.1, num_itr=1000, num_class = 2):
        self.num_class = num_class
        self.num_itr = num_itr
        self.mu = mu
        self.dt = dt
        self.eps = 1e-3
    
    def fit(self, g, train_ind):
        g_len = g.rep['y'].size()[0]
        U_h = torch.zeros(g_len, self.num_class)
        U_h[train_ind] = F.one_hot(g.rep['y'][train_ind], self.num_class).float() # [g.rep['y'][train_ind]] = 1.
        train_mask = torch.zeros(g.rep['y'].size()[0], 1)
        train_mask[train_ind, 0] = 1
        U = torch.ones(g_len, self.num_class) / self.num_class
        
        for i in range(self.num_itr):
            U[train_ind] = F.one_hot(g.rep['y'][train_ind], self.num_class).float()
            U = U - self.dt * ( torch.spmm(g.Lap, U) + self.mu * train_mask * (U - U_h))
            U = F.one_hot(torch.argmax(U, 1), num_classes=self.num_class).float() 
           
        U = torch.argmax(U, 1)

        return self
    
    def predict(self, g, train_ind):
        g_len = g.rep['y'].size()[0]
        U_h = torch.zeros(g_len, self.num_class)
        U_h[train_ind] = F.one_hot(g.rep['y'][train_ind], self.num_class).float() # [g.rep['y'][train_ind]] = 1.
        train_mask = torch.zeros(g.rep['y'].size()[0], 1)
        train_mask[train_ind, 0] = 1
        U = torch.ones(g_len, self.num_class) / self.num_class
        
        for i in range(self.num_itr):
            U[train_ind] = F.one_hot(g.rep['y'][train_ind], self.num_class).float()
            U = U - self.dt * ( torch.spmm(g.Lap, U) + self.mu * train_mask * (U - U_h))
            U = F.one_hot(torch.argmax(U, 1), num_classes=self.num_class).float() 
           
        U = torch.argmax(U, 1)
        return U

def eval_MBO(g, left_in, hold_out, split, device='cpu'):
    evaluator = GraphMBO(num_class=torch.max(g.ndata['label'])+1)
    x, y = g.rep['x'], g.rep['y']
    x_train, x_test, x_val, y_train, y_test, y_val = split_to_numpy(x, y, split)
    ps, [x_train, y_train] = get_predefined_split(x_train, x_val, y_train, y_val)
    params = {'mu': [1.,10.,100.,1000.], 'dt':[1.,0.1,0.01,0.001]}
    classifier = GridSearchCV(evaluator, params, cv=ps, scoring='accuracy', verbose=0)
    classifier.fit(g, split['train']) 
    # prediction = classifier.predict(x_test)
    # test_macro = f1_score(y_test, prediction, average='macro')
    # test_micro = f1_score(y_test, prediction, average='micro')
    # y_test = g.rep['y'][split['test']]
    prediction = evaluator.predict(g, split['train'])[split['test']]
    acc = (y_test.numpy()==prediction.numpy()).mean()
    # print('prediction accuracy on the testing set:{}'.format(acc))
    
    # prediction = classifier.predict(rep['x'][hold_out].cpu())
    # acc1 = (rep['y'][hold_out].cpu().numpy()==prediction).mean()
    # print('prediction accuracy on the hold-out set:{}'.format(acc1))
    
    
    # prediction = classifier.predict(rep['x1'][hold_out].cpu())
    # acc2 = (rep['y'][hold_out].cpu().numpy()==prediction).mean()
    # print('prediction accuracy on the noisy hold-out set:{}'.format(acc2))
    
    return acc , acc, acc