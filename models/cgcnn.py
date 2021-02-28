import torch

import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, ReLU
from torch_scatter import scatter

from torch_geometric.nn import GENConv, DeepGCNLayer, CGConv, global_max_pool, global_mean_pool, global_add_pool
from torch_geometric.data import RandomNodeSampler
class CGCNN(torch.nn.Module):
  def __init__(self, args):
    super(CGCNN, self).__init__()
    self.args = args
    self.num_features = args.num_features  
    self.num_classes = args.num_classes
    self.conv1 = CGConv(3,dim=4)
    self.conv2 = CGConv(3,dim=4)
    self.conv3 = CGConv(3,dim=4)
    self.lin = Linear(6*3,64)
    self.classifier = Linear(64 + 6*3, args.num_classes)
     
  def forward(self, data):
    x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
    x = self.conv1(x,edge_index,edge_attr)
    x1 = torch.cat((global_max_pool(x,batch),global_mean_pool(x,batch)),1)
    x = self.conv2(x,edge_index,edge_attr)
    x2 = torch.cat((global_max_pool(x,batch),global_mean_pool(x,batch)),1)
    x = self.conv2(x,edge_index,edge_attr)
    x3 = torch.cat((global_max_pool(x,batch),global_mean_pool(x,batch)),1)
    x4 = torch.cat((x1, x2, x3),1)
    x4 = self.lin(x4)
    x = global_max_pool(x4,batch)
    x = torch.cat((x,x4),1)
    return self.classifier(x)
