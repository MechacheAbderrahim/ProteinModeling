import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import numpy as np

SEED = 12345
class GCN_surface(torch.nn.Module):
    def __init__(self, k, num_features, embedding_size):
        # Init parent
        super(GCN_surface_p, self).__init__()
        
        self.num_features = num_features
        self.embedding_size = embedding_size
        self.k = k
        
        # GCN layers        
        self.initial_conv = GCNConv(num_features, embedding_size) 
       
        # Message Passing Steps
        self.conv1 = GCNConv(embedding_size, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.conv3 = GCNConv(embedding_size, embedding_size)

        # Classifier
        self.fc1 = Linear(embedding_size * 2, 512)
        self.fc2 = Linear(512, 256)
        self.out = Linear(256, k)

    def forward(self, x, edge_index, batch_index):

        # Conv layers
        x = self.initial_conv(x, edge_index)
        x = F.tanh(x)
        x = self.conv1(x, edge_index) # 
        x = F.tanh(x)
        x = self.conv2(x, edge_index) #
        x = F.tanh(x)
        x = self.conv3(x, edge_index) #
        x = F.tanh(x)

        # Global Pooling (stack different aggregations)
        x = torch.cat([gmp(x, batch_index),
                            gap(x, batch_index)], dim=1)


        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.out(x)

        return out

class GCN_structure(torch.nn.Module):
    def __init__(self, k, num_features, embedding_size):
        # Init parent
        super(GCN_structure_p, self).__init__()
        torch.manual_seed(SEED)
        self.num_features = num_features
        self.embedding_size = embedding_size
        self.k = k
        
        # GCN layers        
        self.initial_conv = GCNConv(num_features, embedding_size) 

        # Message Passing Steps
        self.conv1 = GCNConv(embedding_size, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.conv3 = GCNConv(embedding_size, embedding_size)

        # Classifier
        self.fc1 = Linear(embedding_size * 2, 512)
        self.fc2 = Linear(512, 256)
        self.out = Linear(256, k)

    def forward(self, x, edge_index, batch_index):

        # Conv layers
        x = self.initial_conv(x, edge_index)
        x = F.tanh(x)
        x = self.conv1(x, edge_index) # 
        x = F.tanh(x)
        x = self.conv2(x, edge_index) #
        x = F.tanh(x)
        x = self.conv3(x, edge_index) #
        x = F.tanh(x)

        # Global Pooling (stack different aggregations)
        x = torch.cat([gmp(x, batch_index),
                            gap(x, batch_index)], dim=1)


        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.out(x)

        return out
