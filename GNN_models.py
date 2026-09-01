import torch
import torch.nn.functional as F

from torch import nn
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool, SAGEConv

def make_gin_conv(input_dim, out_dim):
    return GINConv(nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))

class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.layers.append(make_gin_conv(input_dim, hidden_dim))
            else:
                self.layers.append(make_gin_conv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # project_dim = hidden_dim * num_layers
        # self.project = torch.nn.Sequential(
        #     nn.Linear(project_dim, project_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(project_dim, project_dim))

    def forward(self, x, edge_index, batch):
        z = x
        zs = []
        for conv, bn in zip(self.layers, self.batch_norms):
            z = conv(z, edge_index)
            z = F.relu(z)
            z = bn(z)
            zs.append(z)
        gs = [global_add_pool(z, batch) for z in zs]
        z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
        return z, g
    
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ProjectionHead, self).__init__()
        self.layers = nn.ModuleList()

        self.layers = torch.nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, num_classes))
        
    def forward(self, x):
        y = x
        for layer in self.layers:
            y = layer(y)
        return y
    
class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

    def forward(self, x, edge_index, batch):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index)
        z, g = self.encoder(x, edge_index, batch)
        z1, g1 = self.encoder(x1, edge_index1, batch)
        z2, g2 = self.encoder(x2, edge_index2, batch)
        return z, g, z1, z2, g1, g2, edge_index1, edge_index2

class GraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.layers.append(SAGEConv(input_dim, hidden_dim))
            else:
                self.layers.append(SAGEConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # project_dim = hidden_dim * num_layers
        # self.project = torch.nn.Sequential(
        #     nn.Linear(project_dim, project_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(project_dim, project_dim))

    def forward(self, x, edge_index, batch):
        z = x
        zs = []
        for conv, bn in zip(self.layers, self.batch_norms):
            z = conv(z, edge_index)
            z = F.relu(z)
            z = bn(z)
            zs.append(z)
        gs = [global_add_pool(z, batch) for z in zs]
        z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
        return z, g    

class GraphPotts(nn.Module):
    def __init__(self, input_dim, num_classes, depth, eta=0.1):
        super(GraphPotts, self).__init__()
        self.h_dim_rf = 128
        
        self.gcs = GraphSAGE(input_dim, self.h_dim_rf, num_layers=2)
        self.proj_head = ProjectionHead(self.h_dim_rf*2, num_classes)
        self.bn_proj = nn.BatchNorm1d(num_classes)
        
        self.depth = depth 
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.nc = num_classes

        for i in range(self.depth):
            self.convs.append(make_gin_conv(num_classes, num_classes))
            self.bns.append(nn.BatchNorm1d(num_classes))
        self.dt = 1. / depth
        self.eta = eta  # 
        
        self.convs_rf = nn.ModuleList()
        self.bns_rf = nn.ModuleList()
                
        self.convs_rf.append(SAGEConv(input_dim, input_dim))
        self.bns_rf.append(nn.BatchNorm1d(input_dim))
        self.convs_rf.append(SAGEConv(input_dim, num_classes))
        self.bns_rf.append(nn.BatchNorm1d(num_classes))
        
        

    def forward(self, features, edges, batch, Rep, Lap, labeled_data=None):
        g, _ = self.gcs(features,edges,batch)
        z = self.proj_head(g)
        z = self.bn_proj(z)
        # z = embed
        z = torch.softmax(z, dim=1)
        RF = Rep['z'].detach()
        for i in range(2):
            RF = self.convs_rf[i](RF, Rep['edges'])
            RF = F.relu(RF)
            RF = self.bns_rf[i](RF)
            
        RF = global_mean_pool(RF, Rep['batch'])

        for i in range(self.depth):
            z = z - self.dt * self.convs[i](z, edges) - self.dt * RF - self.eta * self.dt * torch.spmm(Lap, z)
            # z = z - self.dt * RF - self.eta * self.dt * torch.spmm(Lap, z)
            # z = self.bns[i](z)
            if i<self.depth-1: z = torch.softmax(z, dim=1)
            # if not self.training and labeled_data is not None: 
            #     z[labeled_data[0,:]] = F.one_hot(labeled_data[1,:], self.nc).float()

        return z


class ImageEncoder(nn.Module):
    def __init__(self, input_channels=1, hidden_dim=32):
        super(ImageEncoder, self).__init__()
        out_dim = hidden_dim * 2
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_dim * 2, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_dim = out_dim

    def forward(self, x):
        h = self.features(x)
        h = self.pool(h)
        h = torch.flatten(h, 1)
        return h
    