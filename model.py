import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.stats import ortho_group

class LinearClassifier(nn.Module):
    def __init__(self, hidden_dim, num_classes, bias=True):
        super(LinearClassifier, self).__init__()   
        self.fc = nn.Linear(hidden_dim, num_classes, bias=bias) 

    def forward(self, x):
        out = self.fc(x)
        return out

# neural network
class TunnelNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_nonlinear_layers=0,
                 bias=False, init_method="default", eps=0.0):
        super(TunnelNetwork, self).__init__()        
        # input layer of size data dim * hidden dim
        self.num_layers = num_layers
        num_linear_layers = num_layers - num_nonlinear_layers
        self.num_nonlinear_layers = num_nonlinear_layers
        self.eps = eps
        # hidden layers
        layers = [nn.Sequential(nn.Linear(input_dim, hidden_dim, bias=bias), nn.BatchNorm1d(hidden_dim), nn.ReLU())] #, nn.BatchNorm1d(hidden_dim)
        # layers = [nn.Sequential(nn.Linear(input_dim, hidden_dim, bias=bias), nn.ReLU())]
        for i in range(1, num_nonlinear_layers):
            layers.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim, bias=bias), nn.BatchNorm1d(hidden_dim), nn.ReLU())) #, nn.BatchNorm1d(hidden_dim)
            # layers.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim, bias=bias), nn.ReLU()))
        for i in range(num_linear_layers):
            layers.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim, bias=bias)))

        self.layers = nn.ModuleList(layers)
        # final classifier
        self.fc = nn.Linear(hidden_dim, output_dim, bias=bias)      
        self.init_method = init_method
        self.init_weight(input_dim, hidden_dim, output_dim)
    
    def init_weight(self, input_dim, hidden_dim, output_dim):
        print(f"initialize weights using {self.init_method}!")
        if self.init_method == 'default':
            H, W = self.fc.weight.data.shape 
        elif self.init_method == 'identity':
            for i in range(self.num_layers):
                self.layers[i][0].weight.data = torch.eye(hidden_dim) * self.eps
            H, W = self.fc.weight.data.shape
            self.fc.weight.data = torch.eye(hidden_dim)[:H,:W] * self.eps
        elif self.init_method == "gaussian":
            for i in range(self.num_layers):
                nn.init.kaiming_normal_(self.layers[i][0].weight)
            nn.init.kaiming_normal_(self.fc.weight)
        elif self.init_method == "orthogonal":
            for i in range(0, self.num_layers):
                if i == 0:
                    weight = torch.randn(hidden_dim, input_dim)
                    weight = torch.linalg.svd(weight, full_matrices=False)[0]
                    weight = torch.cat([weight, torch.zeros(hidden_dim, input_dim-hidden_dim)],dim=1)
                    self.layers[i][0].weight.data = weight * self.eps # weight[:,:data_dim] * eps
                else:
                    weight = torch.randn(hidden_dim, hidden_dim)
                    weight = torch.linalg.svd(weight)[0]
                    self.layers[i][0].weight.data = weight * self.eps
            fc_weight = torch.from_numpy(ortho_group.rvs(output_dim)).float()
            fc_weight = torch.cat([fc_weight, torch.zeros(output_dim, hidden_dim-output_dim)], 1)
            # fc_weight = torch.randn(output_dim, hidden_dim)
            self.fc.weight.data = fc_weight * self.eps
        else:
            raise ValueError("Init Method un-defined!")            
    
    def forward(self, x):
        # store each layer's output
        out_list = []
        for layer in self.layers:
            x = layer(x)
            out_list.append(x.clone().detach())
        
        out = self.fc(x)
        return out, out_list