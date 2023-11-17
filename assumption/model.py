import torch
import torch.nn as nn
from scipy.stats import ortho_group

# neural network
class Network(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, non_linear=False, bias=False, eps=0.1, init_method="default"):
        super(Network, self).__init__()  
        self.num_layers = num_layers      
        self.eps = eps
        # input layer of size data dim * hidden dim
        layers = [nn.Sequential(nn.Linear(input_dim, hidden_dim, bias=bias))]
        # hidden layers
        for i in range(1,num_layers):
            if non_linear:
                # ReLU activations
                if i == 1:
                    layers = [nn.Sequential(nn.Linear(input_dim, hidden_dim, bias=bias), nn.ReLU())] 
                    # layers = [nn.Sequential(nn.Linear(data_dim, hidden_dim, bias=bias), nn.BatchNorm1d(hidden_dim), nn.ReLU())] 
                layers.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim, bias=bias), nn.ReLU()))
                # layers.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim, bias=bias), nn.BatchNorm1d(hidden_dim), nn.ReLU()))
            else:
                 layers.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim, bias=bias)))   
        self.layers = nn.ModuleList(layers)
        self.non_linear = non_linear
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
                self.layers[i][0].weight.data = torch.eye(hidden_dim)*eps
            H, W = self.fc.weight.data.shape
            self.fc.weight.data = torch.eye(hidden_dim)[:H,:W] * eps
        elif self.init_method == "gaussian":
            for i in range(self.num_layers):
                nn.init.kaiming_normal_(self.layers[i][0].weight)
            nn.init.kaiming_normal_(self.fc.weight)
        elif self.init_method == "orthogonal":
            for i in range(self.num_layers):
                if i == 0:
                    weight = torch.randn(hidden_dim, input_dim)
                    weight = torch.linalg.svd(weight, full_matrices=False)[0]
                    # weight = torch.cat([weight, torch.zeros(hidden_dim-data_dim, data_dim)],dim=0)
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
            # print(x.shape)
            x = layer(x)
            out_list.append(x.clone().detach())
        
        out = self.fc(x)
        return out, out_list