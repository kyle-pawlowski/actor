import numpy as np

from typing import List, Tuple, Dict

import torch
from torch.nn import RNN, NLLLoss, MSELoss, Module#, LazyLinear
#from torch_geometric.nn import RGCNConv, GCNConv, GATConv, Linear
from torch_geometric.nn import Linear
#import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

import pyearth
from sklearn.neural_network import MLPRegressor

from utils import trunc_normal
import os

#from IPython.display import clear_output
import matplotlib.pyplot as plt
import pandas as pd

class ParameterModel:
    def __init__(self, epochs=10):
        self.epochs = epochs
    
    def parse_data_gen(folder_x, file_y):
        # Parse label file
        cwd = os.path.dirname(os.path.abspath(__file__))
        try:
            y_data = pd.read_csv(os.path.join(cwd, file_y))
        except: 
            print('Label file doesn\'t exist: ' + os.path.join(cwd, file_y))
            
        # Loop through data files
        folder_x_path = os.path.join(cwd, folder_x)
        x_files = [os.path.join(folder_x_path, f) for f in os.listdir(folder_x_path) if os.path.isfile(os.path.join(folder_x_path,f))]
        for x_file in x_files:
            filename = os.path.basename(x_file)
            try:
                test_num = int((filename.split('_')[1]).split('.')[0]) - 1
            except IndexError:
                print('Bad data filename format :' + filename)
                continue
            
            try:
                x_data = pd.read_csv(x_file)
            except:
                print('Data File Doesn\'t Exist: ' + x_file)
                continue
            x_volt = np.array(x_data['OUT'])
            
            try:
                y_row = list(y_data['Device']).index(test_num)
            except ValueError:
                print('No label for test #' + str(test_num))
                continue
            y_labels = np.array(y_data.iloc[y_row])
            
            yield (x_volt, y_labels)
        
    def parse_data(folder_x, file_y):
        x_data = None
        y_data = None
        for x, y in __class__.parse_data_gen(folder_x, file_y):
            if x_data is None:
                x_data = [x]
                y_data = [y]
            else:
                x_data = np.append(x_data, [x], axis=0) 
                y_data = np.append(y_data, [y], axis=0)
        return np.array(x_data), np.array(y_data)         
        
    def test_data(self, signal, params, algo='MPE'):
        yhat = self.predict(signal)
        if algo == 'MSE':
            return np.sum((yhat-params)**2)
        elif algo == 'MPE': # mean percent error
            return np.mean(np.abs((yhat-params)/yhat)) 
        else:
            raise NotImplementedError()
    
class Mars(ParameterModel):
    def __init__(self, max_terms=100, max_degree=1):
        self.model = pyearth.Earth(max_terms=max_terms, max_degree=max_degree)
        self.max_degree = max_degree
        self.max_terms = max_terms
    def __str__(self):
        return f'MARS\nMax Degree={self.max_degree}\nMax Terms={self.max_terms}'
    def train_data(self, signal, params):
        self.model.fit(signal, params)
    def predict(self, signal):
        return self.model.predict(signal)
    
            
class DNN(ParameterModel):
    def __init__(self, num_params, hidden_layer_sizes, alpha=1e-5, max_iter=200):
        self.model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, alpha=alpha, max_iter=max_iter)
        self.hidden_layers = hidden_layer_sizes
        self.learning_rate = alpha
        self.epochs = max_iter
    def __str__(self):
        return f'DNN\nHidden Layers= {self.hidden_layers}\nLearning Rate= {self.learning_rate}\nEpochs= {self.epochs}'
    def train_data(self, signal, params):
        self.model.fit(signal, params)
    def predict(self, signal):
        return self.model.predict(signal)
    
class myRNN(Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(myRNN, self).__init__()
        self.rnn = RNN(input_size, hidden_size)
        self.h2o = Linear(hidden_size, output_size)
        
    def forward(self, signal):
        signal = np.reshape(signal, (1, signal.shape[0],signal.shape[1])) # pytorch expects an extra dimension for batches
        signal = torch.Tensor(signal)
        rnn_out, hidden = self.rnn(signal)
        output = self.h2o(hidden[0])
        return output

    def train_data(self, signals, params, learning_rate=0.01, epochs=10):
        for epoch in range(epochs):
            criterion = MSELoss()
            optimizer = optim.SGD(self.parameters(), lr=learning_rate)
            
            preds = self.forward(signals)
            #preds = torch.Tensor.reshape(preds, (preds.shape[1], preds.shape[0], preds.shape[2]))
            params = torch.Tensor(params)
            #params = torch.Tensor(np.reshape(params, (params.shape[0], 1, params.shape[1]))) # pytorch assumes there are batches. 
            total_loss = 0
            for pred, param in zip(preds, params):
                loss = criterion(pred, param)
                total_loss += loss
            total_loss.backward()
            clip_grad_norm_(self.parameters(), 3)
            optimizer.step()
            optimizer.zero_grad()
        
class RNNParam(ParameterModel):
    def __init__(self, num_params, hidden_layer_sizes, input_size, learning_rate=0.01, epochs=10):
        super().__init__(epochs)
        self.learning_rate = learning_rate
        self.model = myRNN(input_size, hidden_layer_sizes, num_params)
        self.hidden_layers = hidden_layer_sizes

    def __str__(self):
        return (f'RNN\nHidden Layers= {self.hidden_layers}\nLearning Rate= {self.learning_rate}\nEpochs= {self.epochs}')
    
    def train_data(self, signal, params):
        self.model.train_data(signal, params, self.learning_rate, epochs=self.epochs)
    
    def predict(self, signal):
        return self.model.forward(signal).detach().numpy()
            
        
'''class ActorCriticRGCN:
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
    
            return values'''
