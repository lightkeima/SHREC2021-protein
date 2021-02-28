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
    self.lin = Linear(3,128)
    self.classifier = Linear(128, args.num_classes)
     
  def forward(self, data):
    x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
    x = self.conv1(x,edge_index,edge_attr)
    x1 = self.lin(x)
    x = self.conv2(x,edge_index,edge_attr)
    x2 = self.lin(x)
    x = self.conv3(x,edge_index,edge_attr)
    x3 = self.lin(x)
    x = global_max_pool(x1,batch) + global_max_pool(x2,batch) + global_max_pool(x3,batch)
    x = self.classifier(x)
    return x
