import numpy as np

from typing import List, Tuple, Dict

import torch
from torch.nn import LazyLinear, RNN
from torch_geometric.nn import RGCNConv, GCNConv, GATConv, Linear
import torch.nn.functional as F
import torch.optim as optim

from pyearth import Earth
from sklearn.neural_network import MLPRegressor

from utils import trunc_normal

from IPython.display import clear_output
import matplotlib.pyplot as plt

class ParameterModel:
    def parse_data(filename):
        x = []
        y = []
        # TODO
        return x, y
    def test_data(self, signal, params, algo='LSS'):
        if algo == 'LSS':
            yhat = self.predict(signal)
            return np.sum((yhat-params)**2)
        else:
            raise NotImplementedError()
    
class Mars(ParameterModel):
    def __init__(self):
        self.model = Earth()
    def __str__(self):
        return 'MARS'
    def train_data(self, signal, params):
        self.model.fit(signal, params)
    def predict(self, signal):
        return self.model.predict(signal)
    
            
class DNN(ParameterModel):
    def __init__(self, num_params, hidden_layer_sizes, alpha=1e-5):
        self.model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, alpha=alpha)
        self.hidden_layers = hidden_layer_sizes
    def __str__(self):
        return 'DNN\nHidden Layers=' + str(self.hidden_layers)
    def train_data(self, signal, params):
        self.model.fit(signal, params)
    def predict(self, signal):
        return self.model.predict(signal)
    
class RNNParam(ParameterModel):
    class myRNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(myRNN, self).__init__()
            self.rnn = RNN(input_size, hidden_size)
            self.h2o = nn.Linear(hidden_size, output_size)
        def forward(self, signal):
            rnn_out, hidden = self.rnn(signal)
            output = self.h2o(hidden[0])
            return output
            
    def __init__(self, num_params, hidden_layer_sizes, input_size):
        self.model = myRNN(input_size, hidden_layer_sizes, num_params)
    def __str__(self):
        return 'RNN'
    def train_data(self, signal, params):
        
        
        
class ActorCriticRGCN:
    class Actor(torch.nn.Module):
        def __init__(self, CktGraph):
            super().__init__()
            self.num_node_features = CktGraph.num_node_features
            self.action_dim = CktGraph.action_dim
            self.device = CktGraph.device
            self.edge_index = CktGraph.edge_index
            self.edge_type = CktGraph.edge_type
            self.num_nodes = CktGraph.num_nodes
            self.num_relations = CktGraph.num_relations
    
            self.in_channels = self.num_node_features
            self.out_channels = self.action_dim
            self.conv1 = RGCNConv(self.in_channels, 32, self.num_relations,
                                  num_bases=32)
            self.conv2 = RGCNConv(32, 32, self.num_relations,
                                  num_bases=32)
            self.conv3 = RGCNConv(32, 16, self.num_relations,
                                  num_bases=32)
            self.conv4 = RGCNConv(16, 16, self.num_relations,
                                  num_bases=32)
            self.lin1 = LazyLinear(self.out_channels)
    
        def forward(self, state):
            if len(state.shape) == 2:  # if it is not batched graph data (only one data)
                state = state.reshape(1, state.shape[0], state.shape[1])
    
            batch_size = state.shape[0]
            edge_index = self.edge_index
            edge_type = self.edge_type
            device = self.device
    
            actions = torch.tensor(()).to(device)
            for i in range(batch_size):
                x = state[i]
                x = F.relu(self.conv1(x, edge_index, edge_type))
                x = F.relu(self.conv2(x, edge_index, edge_type))
                x = F.relu(self.conv3(x, edge_index, edge_type))
                x = F.relu(self.conv4(x, edge_index, edge_type))
                x = self.lin1(torch.flatten(x))
                x = torch.tanh(x).reshape(1, -1)
                actions = torch.cat((actions, x), axis=0)
    
            return actions
    
    
    class Critic(torch.nn.Module):
        def __init__(self, CktGraph):
            super().__init__()
            self.num_node_features = CktGraph.num_node_features
            self.action_dim = CktGraph.action_dim
            self.device = CktGraph.device
            self.edge_index = CktGraph.edge_index
            self.edge_type = CktGraph.edge_type
            self.num_nodes = CktGraph.num_nodes
            self.num_relations = CktGraph.num_relations
    
            self.in_channels = self.num_node_features + self.action_dim
            self.out_channels = 1
            self.conv1 = RGCNConv(self.in_channels, 32, self.num_relations,
                                  num_bases=32)
            self.conv2 = RGCNConv(32, 32, self.num_relations,
                                  num_bases=32)
            self.conv3 = RGCNConv(32, 16, self.num_relations,
                                  num_bases=32)
            self.conv4 = RGCNConv(16, 16, self.num_relations,
                                  num_bases=32)
            self.lin1 = LazyLinear(self.out_channels)
    
        def forward(self, state, action):
            batch_size = state.shape[0]
            edge_index = self.edge_index
            edge_type = self.edge_type
            device = self.device
    
            action = action.repeat_interleave(self.num_nodes, 0).reshape(
                batch_size, self.num_nodes, -1)
            data = torch.cat((state, action), axis=2)
    
            values = torch.tensor(()).to(device)
            for i in range(batch_size):
                x = data[i]
                x = F.relu(self.conv1(x, edge_index, edge_type))
                x = F.relu(self.conv2(x, edge_index, edge_type))
                x = F.relu(self.conv3(x, edge_index, edge_type))
                x = F.relu(self.conv4(x, edge_index, edge_type))
                x = self.lin1(torch.flatten(x)).reshape(1, -1)
                values = torch.cat((values, x), axis=0)
    
            return values
        
class ActorCriticGCN:
    class Actor(torch.nn.Module):
        def __init__(self, CktGraph):
            super().__init__()
            self.num_node_features = CktGraph.num_node_features
            self.action_dim = CktGraph.action_dim
            self.device = CktGraph.device
            self.edge_index = CktGraph.edge_index
            self.num_nodes = CktGraph.num_nodes
    
            self.in_channels = self.num_node_features
            self.out_channels = self.action_dim
            self.conv1 = GCNConv(self.in_channels, 32)
            self.conv2 = GCNConv(32, 32)
            self.conv3 = GCNConv(32, 16)
            self.conv4 = GCNConv(16, 16)
            self.lin1 = LazyLinear(self.out_channels)
    
        def forward(self, state):
            if len(state.shape) == 2:  # if it is not batched graph data (only one data)
                state = state.reshape(1, state.shape[0], state.shape[1])
    
            batch_size = state.shape[0]
            edge_index = self.edge_index
            device = self.device
    
            actions = torch.tensor(()).to(device)
            for i in range(batch_size):
                x = state[i]
                x = F.relu(self.conv1(x, edge_index))
                x = F.relu(self.conv2(x, edge_index))
                x = F.relu(self.conv3(x, edge_index))
                x = F.relu(self.conv4(x, edge_index))
                x = self.lin1(torch.flatten(x))
                x = torch.tanh(x).reshape(1, -1)
                actions = torch.cat((actions, x), axis=0)
    
            return actions
    
    class Critic(torch.nn.Module):
        def __init__(self, CktGraph):
            super().__init__()
            self.num_node_features = CktGraph.num_node_features
            self.action_dim = CktGraph.action_dim
            self.device = CktGraph.device
            self.edge_index = CktGraph.edge_index
            self.num_nodes = CktGraph.num_nodes
    
            self.in_channels = self.num_node_features + self.action_dim
            self.out_channels = 1
            self.conv1 = GCNConv(self.in_channels, 32)
            self.conv2 = GCNConv(32, 32)
            self.conv3 = GCNConv(32, 16)
            self.conv4 = GCNConv(16, 16)
            self.lin1 = LazyLinear(self.out_channels)
    
        def forward(self, state, action):
            batch_size = state.shape[0]
            edge_index = self.edge_index
            device = self.device
    
            action = action.repeat_interleave(self.num_nodes, 0).reshape(
                batch_size, self.num_nodes, -1)
            data = torch.cat((state, action), axis=2)
    
            values = torch.tensor(()).to(device)
            for i in range(batch_size):
                x = data[i]
                x = F.relu(self.conv1(x, edge_index))
                x = F.relu(self.conv2(x, edge_index))
                x = F.relu(self.conv3(x, edge_index))
                x = F.relu(self.conv4(x, edge_index))
                x = self.lin1(torch.flatten(x)).reshape(1, -1)
                values = torch.cat((values, x), axis=0)
    
            return values
        
class ActorCriticGAT:
    class Actor(torch.nn.Module):
        def __init__(self, CktGraph):
            super().__init__()
            self.num_node_features = CktGraph.num_node_features
            self.action_dim = CktGraph.action_dim
            self.device = CktGraph.device
            self.edge_index = CktGraph.edge_index
            self.num_nodes = CktGraph.num_nodes
    
            self.in_channels = self.num_node_features
            self.out_channels = self.action_dim
            self.conv1 = GATConv(self.in_channels, 32)
            self.conv2 = GATConv(32, 32)
            self.conv3 = GATConv(32, 16)
            self.conv4 = GATConv(16, 16)
            self.lin1 = LazyLinear(self.out_channels)
    
        def forward(self, state):
            if len(state.shape) == 2:  # if it is not batched graph data (only one data)
                state = state.reshape(1, state.shape[0], state.shape[1])
    
            batch_size = state.shape[0]
            edge_index = self.edge_index
            device = self.device
    
            actions = torch.tensor(()).to(device)
            for i in range(batch_size):
                x = state[i]
                x = F.relu(self.conv1(x, edge_index))
                x = F.relu(self.conv2(x, edge_index))
                x = F.relu(self.conv3(x, edge_index))
                x = F.relu(self.conv4(x, edge_index))
                x = self.lin1(torch.flatten(x))
                x = torch.tanh(x).reshape(1, -1)
                actions = torch.cat((actions, x), axis=0)
    
            return actions
    
    class Critic(torch.nn.Module):
        def __init__(self, CktGraph):
            super().__init__()
            self.num_node_features = CktGraph.num_node_features
            self.action_dim = CktGraph.action_dim
            self.device = CktGraph.device
            self.edge_index = CktGraph.edge_index
            self.num_nodes = CktGraph.num_nodes
    
            self.in_channels = self.num_node_features + self.action_dim
            self.out_channels = 1
            self.conv1 = GATConv(self.in_channels, 32)
            self.conv2 = GATConv(32, 32)
            self.conv3 = GATConv(32, 16)
            self.conv4 = GATConv(16, 16)
            self.lin1 = LazyLinear(self.out_channels)
    
        def forward(self, state, action):
            batch_size = state.shape[0]
            edge_index = self.edge_index
            device = self.device
    
            action = action.repeat_interleave(self.num_nodes, 0).reshape(
                batch_size, self.num_nodes, -1)
            data = torch.cat((state, action), axis=2)
    
            values = torch.tensor(()).to(device)
            for i in range(batch_size):
                x = data[i]
                x = F.relu(self.conv1(x, edge_index))
                x = F.relu(self.conv2(x, edge_index))
                x = F.relu(self.conv3(x, edge_index))
                x = F.relu(self.conv4(x, edge_index))
                x = self.lin1(torch.flatten(x)).reshape(1, -1)
                values = torch.cat((values, x), axis=0)
    
            return values
        
class ActorCriticMLP:
    class Actor(torch.nn.Module):
        def __init__(self, CktGraph):
            super().__init__()
            self.num_node_features = CktGraph.num_node_features
            self.action_dim = CktGraph.action_dim
            self.device = CktGraph.device
            self.edge_index = CktGraph.edge_index
            self.num_nodes = CktGraph.num_nodes
    
            self.in_channels = self.num_node_features
            self.out_channels = self.action_dim
            self.mlp1 = Linear(self.in_channels, 32)
            self.mlp2 = Linear(32, 32)
            self.mlp3 = Linear(32, 16)
            self.mlp4 = Linear(16, 16)
            self.lin1 = LazyLinear(self.out_channels)
    
        def forward(self, state):
            if len(state.shape) == 2:  # if it is not batched graph data (only one data)
                state = state.reshape(1, state.shape[0], state.shape[1])
    
            batch_size = state.shape[0]
            device = self.device
    
            actions = torch.tensor(()).to(device)
            for i in range(batch_size):
                x = state[i]
                x = F.relu(self.mlp1(x))
                x = F.relu(self.mlp2(x))
                x = F.relu(self.mlp3(x))
                x = F.relu(self.mlp4(x))
                x = self.lin1(torch.flatten(x))
                x = torch.tanh(x).reshape(1, -1)
                actions = torch.cat((actions, x), axis=0)
    
            return actions
    
    class Critic(torch.nn.Module):
        def __init__(self, CktGraph):
            super().__init__()
            self.num_node_features = CktGraph.num_node_features
            self.action_dim = CktGraph.action_dim
            self.device = CktGraph.device
            self.edge_index = CktGraph.edge_index
            self.num_nodes = CktGraph.num_nodes
    
            self.in_channels = self.num_node_features + self.action_dim
            self.out_channels = 1
            self.mlp1 = Linear(self.in_channels, 32)
            self.mlp2 = Linear(32, 32)
            self.mlp3 = Linear(32, 16)
            self.mlp4 = Linear(16, 16)
            self.lin1 = LazyLinear(self.out_channels)
    
        def forward(self, state, action):
            batch_size = state.shape[0]
            device = self.device
    
            action = action.repeat_interleave(self.num_nodes, 0).reshape(
                batch_size, self.num_nodes, -1)
            data = torch.cat((state, action), axis=2)
    
            values = torch.tensor(()).to(device)
            for i in range(batch_size):
                x = data[i]
                x = F.relu(self.mlp1(x))
                x = F.relu(self.mlp2(x))
                x = F.relu(self.mlp3(x))
                x = F.relu(self.mlp4(x))
                x = self.lin1(torch.flatten(x)).reshape(1, -1)
                values = torch.cat((values, x), axis=0)
    
            return values
