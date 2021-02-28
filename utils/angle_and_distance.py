import torch
from torch_geometric.utils import to_undirected

class AngleDistanceAttribute(object):
  r""" This transform created by modifying the FaceToEdge() transform.
  It does FaceToEdge transform. Further, it will create edge attribute
  for each connection. Edge attributes contain 4 elements.
  The first 3 elements are 3 angles of the segment created 
  by node i to node j and 3 basis vectors. The last element is the distance
  between node i and node j.

  Args:
  remove_faces (bool, optional): If set to :obj:`False`, the face tensor
                                       will not be removed.
                                                           """
  def __init__(self):
    self.remove_faces = remove_faces
  def __call__(self, data):
    edge_attr = torch.zeros([len(edge_index[0]),4])
    
    Ox = torch.tensor([[1.0],[0.0],[0.0]])
    Oy = torch.tensor([[0.0],[1.0],[0.0]])
    Oz = torch.tensor([[0.0],[0.0],[1.0]])
    vectors = (data.x[edge_index[0]]-data.x[edge_index[1]])

    x_dot = torch.mm(vectors, Ox)
    x_angles = torch.acos(x_dot)

    y_dot = torch.mm(vectors, Oy)
    y_angles = torch.acos(y_dot)

    z_dot = torch.mm(vectors, Oz)
    z_angles = torch.acos(z_dot)

    s = torch.square(data.x[edge_index[0,:]]-data.x[edge_index[1,:]])

    distances = torch.sqrt(  
        torch.mm(s,torch.tensor([[1.0],[1.0],[1.0]]))
    )
    data.edge_attr = edge_attr
    torch.set_printoptions(edgeitems=1)
    
    edge_attr = torch.cat((x_angles, y_angles, z_angles, distances),1)
    data.edge_attr = edge_attr
    return data

  def __repr__(self):
    return '{})()'.format(self.__class__.__name__)

