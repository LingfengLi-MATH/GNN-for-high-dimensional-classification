import os
import os.path as osp
os.environ["DGLBACKEND"] = "pytorch"
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from GCL.eval import get_split
import GCL.augmentors as A
import numpy as np
import argparse
from torch.optim.lr_scheduler import StepLR, CyclicLR
from sklearn.svm import LinearSVC, SVC
from tqdm import tqdm
from GNN_models import GConv, Encoder, ProjectionHead, GraphPotts
from eval_methods import eval_GNN, eval_baseline, eval_svm, eval_GPN, eval_SupCon, eval_MBO
from ssp_gcl import GCL_training
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.metrics import f1_score
from utils import *
from torch.utils.data import Subset
import copy

parser = argparse.ArgumentParser(description="node classification task")
parser.add_argument("--dataset", type=str, default='PROTEINS', help="dataset to use (PROTEINS,COLLAB,ENZYMES,MUTAG)")

parser.add_argument("--model", type=str, default='GNN', help="choose from GNN, svm, baseline, SupCon, GPN")
parser.add_argument("--gpu", type=int, default=0, help="gpu")
parser.add_argument("--hdim", type=int, default=32, help="dimension of representation")
# parser.add_argument("--lr", type=float, default=1e-1, help="learning rate")
parser.add_argument("--epochs", type=int, default=200, help="number of training epochs")
parser.add_argument("--aug", type=int, default=0, help="use augmented test data")


args = parser.parse_args()
######################################################################
######################################################################
device = torch.device('cuda:{}'.format(args.gpu)) if torch.cuda.is_available() else torch.device('cpu') 
path = osp.join(osp.expanduser('~'), 'datasets')
dataset_name = args.dataset
# dataset = TUDataset(path, name=dataset_name)
graph_dataset = TUDataset(path, name=dataset_name, use_node_attr=True)
indices = [i for i in range(graph_dataset.len())]
np.random.seed(0)
np.random.shuffle(indices)
left = indices[0:int(len(indices)*0.8)]
hold = indices[int(len(indices)*0.8):]
graph_subset = Subset(graph_dataset, left)


file_path = 'models/'+dataset_name+'_encoder_hdim{}.pth'.format(args.hdim)
if osp.isfile(file_path):
    print('graph encoder exists.')
else:
    print('start training encoder by graph contrastive learning.')
    GCL_training(graph_dataset, left, file_path, device=device)


data_path = 'data/'+dataset_name+('_gpu' if torch.cuda.is_available() else '_cpu')+'.pth'



input_dim = max(graph_dataset.num_features, 1)

aug1 = A.RandomChoice([ A.RWSampling(num_seeds=200, walk_length=5),
                        A.NodeDropping(pn=0.3),
                        A.FeatureMasking(pf=0.3),
                        A.EdgeRemoving(pe=0.3)], 4)
aug2 = A.RandomChoice([A.RWSampling(num_seeds=1000, walk_length=10),
                        A.NodeDropping(pn=0.2),
                        A.FeatureMasking(pf=0.2),
                        A.EdgeRemoving(pe=0.2)], 4)

gconv = GConv(input_dim=input_dim, hidden_dim=args.hdim, num_layers=2).to(device)
encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2)).to(device)
encoder_model.load_state_dict(torch.load(file_path, weights_only=True, map_location=device))
encoder_model.eval()

if osp.isfile(data_path):
    loaded_data = torch.load(data_path, map_location=device)
    # splits = DatasetSplit(num_samples=dataset.len(), train_ratio=0.5, val_ratio=0.1, k=10)
    graph_sub = loaded_data['graph_sub']
    graph_full = loaded_data['graph_full']
    graph_full_noisy = loaded_data['graph_full_noisy']
    # rep = loaded_data['rep']
else:  
    dataloader = DataLoader(graph_dataset, batch_size=graph_dataset.len(), shuffle=False)
    rep = get_vector_representations(encoder_model, dataloader, device)
    datasubloader = DataLoader(graph_subset, batch_size=len(left), shuffle=False)
    rep_sub = get_vector_representations(encoder_model, datasubloader, device)
    graph_full, graph_sub = construct_dgl_graph(rep['x'].detach().cpu(), rep['y'], left)
    graph_full.hold = hold
    graph_full.left = left
    graph_full.rep = rep
    graph_sub.rep = rep_sub
    graph_full_noisy = copy.deepcopy(graph_full)
    graph_full_noisy.ndata['feat'][hold] = rep['x1'][hold]

graph_full, graph_sub, graph_full_noisy = graph_full.to(device), graph_sub.to(device), graph_full_noisy.to(device)
test, test_hold, test_hold_noise = [], [], []
for seed in range(1, 6):
    torch.manual_seed(seed)
    split = DatasetSplit(num_samples=len(left), train_ratio=0.25, val_ratio=0.25, k=1)[0]
    
    if args.model=='GNN':
        acc1, acc2, acc3 = eval_GNN(graph_sub, graph_full, graph_full_noisy, split, args.epochs, device) 
    elif args.model=='svm':
        acc1, acc2, acc3 = eval_svm(graph_full.rep, left, hold, split, device)
    elif args.model=='MBO':
        acc1, acc2, acc3 = eval_MBO(graph_sub, left, hold, split, device)
    elif args.model=='baseline':
        acc1, acc2, acc3 = eval_baseline(graph_dataset, split, left, hold, input_dim, aug1, args.epochs, device)
    elif args.model=='SupCon':
        acc1, acc2, acc3 = eval_SupCon(graph_dataset, split, left, hold, input_dim, aug1, args.epochs, device)
    elif args.model=='GPN':
        acc1, acc2, acc3 = eval_GPN(graph_sub, graph_full, graph_full_noisy, split, args.epochs, device)
    test.append(acc1), test_hold.append(acc2), test_hold_noise.append(acc3)

with open("results.txt", "a") as myfile:
    myfile.write('dataset:{}, model:{}\n'.format(args.dataset, args.model))
    myfile.write('prediction accuracy on the testing set:{}({})\n'.format(np.mean(test), np.std(test)))
    myfile.write('prediction accuracy on the hold-out set:{}({})\n'.format(np.mean(test_hold), np.std(test_hold)))
    myfile.write('prediction accuracy on the noisy hold-out set:{}({})\n'.format(np.mean(test_hold_noise), np.std(test_hold_noise)))
print('dataset:{}, model:{}'.format(args.dataset, args.model))    
print('prediction accuracy on the testing set:{}({})'.format(np.mean(test), np.std(test)))
print('prediction accuracy on the hold-out set:{}({})'.format(np.mean(test_hold), np.std(test_hold)))     
print('prediction accuracy on the noisy hold-out set:{}({})'.format(np.mean(test_hold_noise), np.std(test_hold_noise)))     

if not osp.isfile(data_path):
    saved_data = {
        'graph_sub': graph_sub, 
        'graph_full': graph_full,
        'graph_full_noisy': graph_full_noisy, 
        # 'rep': rep 
        }
    torch.save(saved_data, data_path)

