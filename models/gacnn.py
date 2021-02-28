import torch
import torch.nn.functional as F
from torch_cluster import knn_graph
from torch_geometric.nn import GATConv, global_max_pool, global_mean_pool
from torch import nn


class GACNN(torch.nn.Module):
  def __init__(self, args):
    super(GACNN, self).__init__()
    torch.manual_seed(args.seed)
  def forward(self, data):
    pos, batch = data.pos, data.batch
    
    
