import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.linalg as scilin

def dict_to_mat(d1):
    """
    transfer the mu_c dictionary to a matrix
    """
    all_mat = []
    for l in d1:
        mat = []
        d1_l = d1[l]
        for key in d1_l:
            mat.append(d1_l[key])
        mat = np.vstack(mat)
        
        all_mat.append(mat)
    return all_mat
        
def max_angle_metric(muc_mat):
    """
    max |<u_k, u_l>|，u_k是k-th class的mean
    k \neq l
    """
    metric_list = []
    for i in range(len(muc_mat)):
        cur_muc = muc_mat[i]
        muc_norm = cur_muc / np.linalg.norm(cur_muc, axis=1)[:,None]
        cosine = muc_norm @ muc_norm.T
        cosine_need = np.tril(cosine, k=-1)
        cosine_need = cosine_need[cosine_need!=0]
        metric_list.append(np.max(cosine_need))
    return metric_list

def compute_info(model, dataloader, num_layers, device):
    # Need mu_G, mu_c_dict for each of the layers' output
    num_data = 0
    mu_G = dict()
    mu_c_dict = dict()
    num_class_dict = dict()
    before_class_dict = dict() # Store features
    num_class_dict = dict() # Store # of samples in each class

    for batch_idx, (inputs, targets) in enumerate(dataloader):

        inputs, targets = inputs.to(device), targets.to(device)
        # Class indices based label (i.e., label = [1,2,1,0,3] 
        # where each entry represent where the label from)
        if len(targets.shape) == 2:
            targets = torch.argmax(targets, dim=1) 

        with torch.no_grad():
            if len(inputs.shape) != 2:
                inputs = inputs.view(inputs.shape[0], -1)
            output, feature_list = model(inputs)
        
        assert len(feature_list) == num_layers
        for i in range(num_layers):
            features = feature_list[i]
            
            if batch_idx == 0:
                mu_G[i] = 0
                mu_c_dict[i] = dict()
                before_class_dict[i] = dict()
                
            mu_G[i] += torch.sum(features, dim=0).detach().cpu().numpy()
            
            for b in range(len(targets)):
                y = targets[b].item()
                if y not in mu_c_dict[i]:
                    mu_c_dict[i][y] = features[b, :].detach().cpu().numpy()
                    before_class_dict[i][y] = [features[b, :].detach().cpu().numpy()]
                    if i == 0:
                        num_class_dict[y] = 1
                else:
                    mu_c_dict[i][y] += features[b, :].detach().cpu().numpy()
                    before_class_dict[i][y].append(features[b, :].detach().cpu().numpy())
                    if i == 0:
                        num_class_dict[y] = num_class_dict[y] + 1

        num_data += targets.shape[0]
    
    for i in range(num_layers):
        mu_G[i] = mu_G[i] / num_data
        for cla in range(len(mu_c_dict[0].keys())):
            mu_c_dict[i][cla] /= num_class_dict[cla]

    return mu_G, mu_c_dict, before_class_dict

# Within-class covariance matrix
def compute_Sigma_W(before_class_dict, mu_c_dict, device):
    num_data = 0
    Sigma_W = 0
    
    for target in before_class_dict.keys():
        class_feature_list = torch.from_numpy(np.array(before_class_dict[target])).float().to(device)
        class_mean = torch.from_numpy(mu_c_dict[target]).float().to(device)
        for feature in class_feature_list:
            diff = feature - class_mean
            Sigma_W += torch.outer(diff,diff)
            num_data += 1
    Sigma_W /= num_data
    
    return Sigma_W.cpu().numpy()

# Between-class covariance matrix
def compute_Sigma_B(mu_c_dict, mu_G, device):
    mu_G = torch.from_numpy(mu_G).float().to(device)
    Sigma_B = 0
    K = len(mu_c_dict)
    for i in range(K):
        class_mean = torch.from_numpy(mu_c_dict[i]).float().to(device)
        diff = class_mean - mu_G
        Sigma_B += torch.outer(diff,diff)

    Sigma_B /= K

    return Sigma_B.cpu().numpy()

# Main operation function
def calculate_nc1(model, dataloader, num_layers, device):
    """
    We'll collect information and calculate NC1 for each of the layer's output
    """
    nc1_list = []
    nc1_tilde_list = []
    ssw_fro_list = []
    ssb_fro_list = []
    mu_G, mu_c_dict, before_class_dict = compute_info(model, dataloader, num_layers, device)
    
    ## Add for input ##
    num_data = 0
    mu_G_inp = 0
    mu_c_dict_inp = {}
    before_class_dict_inp = {}
    num_class_dict_inp = {}
    
    # For inputs
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        features = inputs
        #targets = torch.argmax(targets, dim=1) 
        if len(targets.shape) == 2:
            targets = torch.argmax(targets, dim=1) 

        mu_G_inp += torch.sum(features, dim=0).detach().cpu().numpy()

        for b in range(len(targets)):
            y = targets[b].item()
            if y not in mu_c_dict_inp:
                mu_c_dict_inp[y] = features[b, :].detach().cpu().numpy()
                before_class_dict_inp[y] = [features[b, :].detach().cpu().numpy()]
                num_class_dict_inp[y] = 1
            else:
                mu_c_dict_inp[y] += features[b, :].detach().cpu().numpy()
                before_class_dict_inp[y].append(features[b, :].detach().cpu().numpy())
                num_class_dict_inp[y] = num_class_dict_inp[y] + 1
        
        num_data += targets.shape[0]
    
    mu_G_inp = mu_G_inp / num_data
    for cla in range(len(mu_c_dict_inp.keys())):
        mu_c_dict_inp[cla] /= num_class_dict_inp[cla]
        
    # Get NC1 for inputs
    Sigma_W = compute_Sigma_W(before_class_dict_inp, mu_c_dict_inp, device)
    ssw_fro_list.append(np.trace(Sigma_W))
    Sigma_B = compute_Sigma_B(mu_c_dict_inp, mu_G_inp, device)
    ssb_fro_list.append(np.trace(Sigma_B))
    collapse_metric = np.trace(Sigma_W @ scilin.pinv(Sigma_B)) / len(mu_c_dict_inp)
    nc1_list.append(collapse_metric)
    nc1_tilde = np.trace(Sigma_W) / np.trace(Sigma_B)
    nc1_tilde_list.append(nc1_tilde)
    ## Add for input end ##
            
    # For other layers
    for layer_idx in range(num_layers):
        mu_G_layer = mu_G[layer_idx]
        mu_c_dict_layer = mu_c_dict[layer_idx]
        before_class_dict_layer = before_class_dict[layer_idx]
        
        # Get NC1 for this layer
        Sigma_W = compute_Sigma_W(before_class_dict_layer, mu_c_dict_layer, device)
        # ssw_fro_list.append(np.sqrt(np.sum(Sigma_W ** 2)))
        ssw_fro_list.append(np.trace(Sigma_W))
        Sigma_B = compute_Sigma_B(mu_c_dict_layer, mu_G_layer, device)
        ssb_fro_list.append(np.trace(Sigma_B))
        collapse_metric = np.trace(Sigma_W @ scilin.pinv(Sigma_B)) / len(mu_c_dict_layer)
        nc1_list.append(collapse_metric)
        nc1_tilde = np.trace(Sigma_W) / np.trace(Sigma_B)
        nc1_tilde_list.append(nc1_tilde)
        
    return ssw_fro_list, ssb_fro_list, nc1_list, nc1_tilde_list

def compute_weight_fnorm(model):
    all_weight_norm = {}
    for name, para in model.named_parameters():
        all_weight_norm[name] = torch.linalg.norm(para.data).item()
    return all_weight_norm

def get_gradient_norm(model, verbose=False):
    total_grad_norm_squared = 0
    for key, layer in model.named_parameters():
        grad_weight = layer.grad.data
        if verbose:
            print(f"Current Layer is {key} and the gradient norm is {torch.linalg.norm(grad_weight)}")
        total_grad_norm_squared += torch.linalg.norm(grad_weight) ** 2
    return torch.sqrt(total_grad_norm_squared)

def compute_norm_diff(model):
    """
    Calculate norm(W_iW_i^T - W_{i+1}^T W_{i+1}, 'fro')
    """
    all_weights = []
    for name, para in model.named_parameters():
        #if "layers" in name and "weight" in name:
        if "weight" in name:
            print(name, para.shape)
            all_weights.append(para.data)
    
    norm_diffs = []
    diffs = []
    for i in range(len(all_weights)-2):
        W_cur, W_next = all_weights[i], all_weights[i+1]
        diff = W_cur @ W_cur.T - W_next.T @ W_next
        diffs.append(diff)
        norm_diffs.append(torch.linalg.norm(diff))
    
    return norm_diffs, diffs

def compute_weight_snorm(model):
    # Spectral Norm
    all_weight_norm = {}
    for name, para in model.named_parameters():
        all_weight_norm[name] = torch.linalg.norm(para.data, ord=2).item()
    return all_weight_norm

def check_singular(model, data_dim):
    """
    Calculate singular values
    """
    all_weights = []
    for name, para in model.named_parameters():
        #if "layers" in name and "weight" in name:
        if "weight" in name:
            all_weights.append(para.data)
    
    all_sings = []
    for i in range(len(all_weights)-1):
        W_cur = all_weights[i]
        sings = torch.linalg.svd(W_cur)[1][data_dim-1].item()
        all_sings.append(sings)
        
    return np.array(all_sings)