import torch
from torch.nn import Linear
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import GENConv
import torch.nn.functional as F

class CGCNN(torch.nn.Module):
  def __init__(self, args):
    super(CGCNN, self).__init__()
    self.args = args
    self.num_features = args.num_features  
    self.num_classes = args.num_classes

    self.dim_edge_attr = 4
    self.conv1 = GENConv(4, 128)
    self.pool1 = TopKPooling(128, ratio=0.8)
    self.conv2 = GENConv(128,128)
    self.pool2 = TopKPooling(128,ratio=0.8)
    self.lin1 = Linear(256, 128)
    self.lin2 = Linear(128, self.num_classes)
     
  def forward(self, data):
    x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
    #print(x)
    x = torch.cat([x, torch.zeros(x.size(0),1,device="cuda:0")], dim=1)
    x = self.conv1(x, edge_index, edge_attr)
    x = x.relu()
    x, edge_index, edge_attr, batch, _, _ = self.pool1(x, edge_index, edge_attr, batch)
    x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
    x = self.conv2(x, edge_index, None)
    x = x.relu()
    x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
    x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
    
    x = x1 + x2
    
    x = self.lin1(x)
    x = x.relu()  
    x = self.lin2(x)
    #x = F.log_softmax(self.lin2(x), dim=-1)

    return x
