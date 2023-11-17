import argparse
import torch
import numpy as np
from partial_data import get_data
from model import TunnelNetwork

def train_epoch(args, model, dataloader):
    all_out = {}
    for i in range(args.depth):
        all_out[i] = []
    model.eval()
    # One epoch
    for batch_idx, (inputs, targets) in enumerate(dataloader):

        inputs, targets = inputs.to(args.device), targets.to(args.device)
        
        with torch.no_grad():
            _, layer_outs = model(inputs)
            
        if batch_idx == 0:
            assert len(layer_outs) == args.depth
        
        for i in range(args.depth):
            all_out[i].append(layer_outs[i])
        
    return all_out

def parse_eval_args():
    parser = argparse.ArgumentParser()

    # parameters
    # Model Selection
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--num_nonlinear', type=int, default=5) 
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--depth', type=int, default=8)
    # Initialization
    parser.add_argument('--init', type=str, default="orthogonal", choices=['orthogonal', 'default'])
    parser.add_argument('--eps', type=float, default=0.1)

    # Hardware Setting
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=6)

    # Directory Setting
    # Note dataset mean which dataset to transfer learning on
    parser.add_argument('--data_dir', type=str, default='/scratch/qingqu_root/qingqu1/xlxiao/DL/data')
    parser.add_argument('--dataset', type=str, default="cifar10", choices=['mnist', 'cifar10'])
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--sample_size', type=int, default=5000)
    parser.add_argument('--save_path', type=str, default=None)

    # Learning Options
    parser.add_argument('--epochs', type=int, default=100, help='Max Epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')

    args = parser.parse_args()

    return args

def main():
    args = parse_eval_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    
    # Dataset part
    trainloader, testloader = get_data(args.dataset, args.data_dir, args.sample_size, args.batch_size, args.num_classes, shuffle=False)
    
    if args.dataset == "mnist":
        data_dim = 784
    elif args.dataset == "cifar10":
        data_dim = 3072
    else:
        raise ValueError("dataset not supported!")
        
    model = TunnelNetwork(input_dim=data_dim, hidden_dim=args.hidden, 
                        num_layers=args.depth, output_dim=args.num_classes,
                        init_method=args.init,
                        num_nonlinear_layers=args.num_nonlinear,
                        eps = args.eps).to(device)

    if args.load_path is not None:
        ckpt = torch.load(args.load_path + '/model_last.pth')
        model.load_state_dict(ckpt['state_dict'])
    else:
        print("Check Vanilla network")
        
    all_out = train_epoch(args, model, trainloader)
    for i in range(args.depth):
        all_out[i] = torch.cat(all_out[i], 0).detach().cpu()

    checkpoint_dir = args.load_path

    state = {
                'all_out': all_out,
            }
    print("Save results")

    if args.load_path is not None:
        checkpoint_dir = args.load_path
    else:
        checkpoint_dir = f"saved_linear/Vanilla_dataset_{args.dataset}_w{args.hidden}_d{args.depth}_nd{args.num_nonlinear}_init_{args.init}_eps{args.eps}_sample{args.sample_size}/" 
    path = checkpoint_dir + '/outputs'
    torch.save(state, path)
    
if __name__ == "__main__":
    main()