import torch

import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, ReLU
from torch_scatter import scatter

from torch_geometric.nn import GENConv, DeepGCNLayer
from torch_geometric.data import RandomNodeSampler
class CGCNN(torch.nn.Module):
  def __init__(self, args):
    super(CGCNN, self).__init__()
    self.args = args
    self.num_features = args.num_features  
    self.num_classes = args.num_classes
    hidden_channels = 128
    self.node_encoder = Linear(4, hidden_channels)
    #self.edge_encoder = Linear(3, hidden_channels)
    num_layers = 12
    self.layers = torch.nn.ModuleList()
    for i in range(1, num_layers + 1):
      conv = GENConv(hidden_channels, hidden_channels, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
      norm = LayerNorm(hidden_channels, elementwise_affine=True)
      act = ReLU(inplace=True)

      layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
                                 ckpt_grad=i % 3)
      self.layers.append(layer)

    self.lin = Linear(hidden_channels, args.num_classes)
     
  def forward(self, data):
    x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
    #edge_attr = self.edge_encoder(edge_attr)
    x = self.node_encoder(x)
    x = self.layers[0].conv(x, edge_index, None)

    for layer in self.layers[1:]:
      x = layer(x, edge_index, None)

      x = self.layers[0].act(self.layers[0].norm(x))
      x = F.dropout(x, p=0.1, training=self.training)

      return self.lin(x)
