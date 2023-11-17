import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ortho_group
from tqdm import trange
from utility import *
from dataset import *
from model import Network
import random
import os
import argparse
import torch.optim as optim

# Loss function
def loss_fn(args, pred, target):
    # Just MSE
    alpha = 1 #/ (args.eps ** (args.depth-3) * args.lr)
    N,C = target.shape
    loss = torch.sum((alpha*pred-target) ** 2) / N 

    return loss

def train_epoch(args, model, dataloader, optimizer, criterion, cur_epoch):
    
    model.train()
    # One epoch
    for batch_idx, (inputs, targets) in enumerate(dataloader):

        inputs, targets = inputs.to(args.device), targets.to(args.device)
        
        outputs, _ = model(inputs)
        
        loss = criterion(args, outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()
        
def set_seed(manualSeed=666):
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(manualSeed)

def parse_eval_args():
    parser = argparse.ArgumentParser()

    # parameters
    # Model Selection
    parser.add_argument('--hidden', type=int, default=50)
    parser.add_argument('--depth', type=int, default=4)
    # Initialization
    parser.add_argument('--init', type=str, default="orthogonal", choices=['orthogonal', 'default'])
    parser.add_argument('--eps', type=float, default=0.5)

    # Hardware Setting
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=6)

    # Directory Setting
    # Note dataset mean which dataset to transfer learning on
    parser.add_argument('--data_dir', type=str, default='/scratch/qingqu_root/qingqu1/xlxiao/DL/data')
    parser.add_argument('--dataset', type=str, default="cifar10", choices=['mnist', 'cifar10'])
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--save_path', type=str, default="saved_linear/")

    # Learning Options
    parser.add_argument('--epochs', type=int, default=100000, help='Max Epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')

    args = parser.parse_args()

    return args

def main():
    args = parse_eval_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device

    set_seed(args.seed)
    
    # parameters of input data
    num_samples = 30 # number of samples
    data_dim = 50 # data dimension

    # parameters for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_type = "orthogonal" # Choose from [gaussian, uniform, mnist_partial, orthogonal, cifar]

    # Dataset part
    if dataset_type == "orthogonal":
        trainset, trainloader = orthogonal_dataset(num_samples, data_dim, args.num_classes, bs=num_samples)
        
    model = Network(input_dim=data_dim, hidden_dim=args.hidden, 
                    num_layers=args.depth, output_dim=args.num_classes,
                    init_method=args.init,
                    eps=args.eps).to(device)
    
    criterion = loss_fn
    initial_eta = args.lr
    optimizer = optim.SGD(model.parameters(), lr=initial_eta, momentum = 0.0, weight_decay=0)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.epochs, eta_min=initial_eta/1000)
    
    epochs = args.epochs
    train_losses = []
    
    save_name = f"w{args.hidden}_d{args.depth}_init_{args.init}_eps{args.eps}_seed{args.seed}_assump2/" # the folder to save trained model
        
    checkpoint_dir = args.save_path + save_name
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    for epoch in range(1,epochs+1):
        train_loss = train_epoch(args, model, trainloader, optimizer, criterion, epoch)
        train_losses.append(train_loss)
        scheduler.step()

        if epoch % 100 == 0:
            print(f"Finish Epoch {epoch+1}, current loss is {train_loss}")
        
        state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'train_acc': train_loss,
                    'data': trainset.data,
                    'target': trainset.target
                }
        
        if train_loss < 1e-10:
            print("Done", train_loss)
            break

    print("Save last model")
    path = checkpoint_dir + f'model_last.pth'
    torch.save(state, path)

if __name__ == "__main__":
    main()