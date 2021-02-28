import torch

import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, ReLU
from torch_scatter import scatter
from torch_cluster import knn_graph

from torch_geometric.nn import GENConv, DeepGCNLayer, CGConv, global_max_pool, global_mean_pool, global_add_pool
from torch_geometric.data import RandomNodeSampler
class CGCNN(torch.nn.Module):
  def __init__(self, args):
    super(CGCNN, self).__init__()
    self.args = args
    self.num_features = args.num_features  
    self.num_classes = args.num_classes
    torch.manual_seed(args.seed)
    self.conv1 = CGConv(3,dim=4)
    self.conv2 = CGConv(3,dim=4)
    self.conv3 = CGConv(3,dim=4)
    self.lin = Linear(3,128)
    self.classifier = Linear(128, args.num_classes)
     
  def forward(self, data):
    x, pos, batch = data.x, data.pos, data.batch
    edge_index = knn_graph(pos, k=16, batch=batch, loop = True)
 
    edge_attr = torch.zeros([len(edge_index[0]),4])
    
    Ox = torch.tensor([[1.0],[0.0],[0.0]])
    Oy = torch.tensor([[0.0],[1.0],[0.0]])
    Oz = torch.tensor([[0.0],[0.0],[1.0]])
    vectors = (data.pos[edge_index[0]]-data.pos[edge_index[1]])

    x_dot = torch.mm(vectors, Ox)
    x_angles = torch.acos(x_dot)

    y_dot = torch.mm(vectors, Oy)
    y_angles = torch.acos(y_dot)

    z_dot = torch.mm(vectors, Oz)
    z_angles = torch.acos(z_dot)

    s = torch.square(data.pos[edge_index[0,:]]-data.pos[edge_index[1,:]])

    distances = torch.sqrt(  
        torch.mm(s,torch.tensor([[1.0],[1.0],[1.0]]))
    ) 
    edge_attr = torch.cat((x_angles, y_angles, z_angles, distances),1)   
    x = self.conv1(x,edge_index,edge_attr)
    x1 = self.lin(x)
    x = self.conv2(x,edge_index,edge_attr)
    x2 = self.lin(x)
    x = self.conv3(x,edge_index,edge_attr)
    x3 = self.lin(x)
    x = global_max_pool(x1,batch) + global_max_pool(x2,batch) + global_max_pool(x3,batch)
    x = self.classifier(x)
    return x
