import argparse
import random
import os
import torch
import torch.optim as optim
import numpy as np
from partial_data import get_data
from model import TunnelNetwork, LinearClassifier
import json

# Loss function
def loss_fn(args, pred, target):
    alpha = args.alpha
    N,C = target.shape
    loss = torch.sum((alpha*pred-target) ** 2) / N 

    return loss

def train_epoch(args, model, dataloader, optimizer, criterion, cur_epoch):
    
    model.train()
    # One epoch
    top1 = AverageMeter()
    top5 = AverageMeter()
    running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(dataloader):

        inputs, targets = inputs.to(args.device), targets.to(args.device)
        
        outputs, _ = model(inputs)
        
        loss = criterion(args, outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        prec1, prec5 = compute_accuracy(outputs.data, targets.data, topk=(1, 5))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # print statistics
        running_loss += loss.item()
        if batch_idx % 100 == 0:    # print every 100 mini-batches
            print(f'[Epoch {cur_epoch}, {batch_idx + 1:5d}] loss: {running_loss:.3f}, acc: {top1.avg}')
            running_loss = 0.0
        
    return top1.avg, top5.avg

def evaluate_epoch(args, model, dataloader, criterion):
    
    model.eval()
    # One epoch
    test_top1 = AverageMeter()
    test_top5 = AverageMeter()
    running_loss = 0.0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):

            inputs, targets = inputs.to(args.device), targets.to(args.device)
            
            outputs, _ = model(inputs)

            loss = criterion(args, outputs, targets)

            prec1, prec5 = compute_accuracy(outputs.data, targets.data, topk=(1, 5))
            test_top1.update(prec1.item(), inputs.size(0))
            test_top5.update(prec5.item(), inputs.size(0))

    return test_top1.avg, test_top5.avg
    
def compute_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    # Added for One-hot vector
    target = torch.argmax(target, 1)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
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
    parser.add_argument('--num_nonlinear', type=int, default=5) 
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--depth', type=int, default=8)
    # Initialization
    parser.add_argument('--init', type=str, default="orthogonal", choices=['orthogonal', 'default'])
    parser.add_argument('--eps', type=float, default=0.3)

    # Hardware Setting
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=6)

    # Directory Setting
    # Note dataset mean which dataset to transfer learning on
    parser.add_argument('--data_dir', type=str, default='/scratch/qingqu_root/qingqu1/xlxiao/DL/data')
    parser.add_argument('--dataset', type=str, default="cifar10", choices=['cifar10'])
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--sample_size', type=int, default=5000)
    parser.add_argument('--save_path', type=str, default="saved_linear/")

    # Learning Options
    parser.add_argument('--epochs', type=int, default=400, help='Max Epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate') #0.001
    parser.add_argument('--alpha', type=float, default=1.0, help='feature multiplier')

    args = parser.parse_args()

    return args

def main():
    args = parse_eval_args()

    save_name = f"dataset_{args.dataset}_w{args.hidden}_d{args.depth}_nd{args.num_nonlinear}_init_{args.init}_eps{args.eps}_sample{args.sample_size}/"
    checkpoint_dir = args.save_path + save_name
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)

    with open(checkpoint_dir + '/args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device

    set_seed(args.seed)

    if args.dataset == "cifar10":
        data_dim = 3072
    else:
        raise ValueError("dataset not supported!")
    
    # Dataset part
    trainloader, testloader = get_data(args.dataset, args.data_dir, args.sample_size, args.batch_size, args.num_classes)
    
    if args.depth == 0:
        model = LinearClassifier(hidden_dim=data_dim, num_classes=args.num_classes).to(device)
    else:
        model = TunnelNetwork(input_dim=data_dim, hidden_dim=args.hidden, 
                            num_layers=args.depth, output_dim=args.num_classes,
                            init_method=args.init,
                            num_nonlinear_layers=args.num_nonlinear,
                            eps = args.eps).to(device)
    
    criterion = loss_fn
    initial_eta = args.lr
    optimizer = optim.SGD(model.parameters(), lr=initial_eta, momentum = 0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.epochs, eta_min=initial_eta/100) #/1000
    
    epochs = args.epochs
    best_test_acc = 0.0
    train_accs = []
    test_accs = []
    is_best = False 
        
    for epoch in range(epochs):
        train_top1, train_top5 = train_epoch(args, model, trainloader, optimizer, criterion, epoch)
        test_top1, test_top5 = evaluate_epoch(args, model, testloader, criterion)
        train_accs.append(train_top1)
        test_accs.append(test_top1)
        scheduler.step()

        if test_top1 > best_test_acc:
            is_best = True
            best_test_acc = test_top1
        else:
            is_best = False
        print(f"Finish Epoch {epoch}, Training Acc: {train_top1}, Test Acc: {test_top1}, Current best Test Acc: {best_test_acc}, current LR: {scheduler.get_lr()}")
        
        state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'train_acc': train_top1,
                    'test_acc': test_top1
                }
        if is_best:
            print("Save current model (best)")
            path = checkpoint_dir + 'model_best.pth'
            torch.save(state, path)

    print("Save last model")
    path = checkpoint_dir + f'model_last.pth'
    torch.save(state, path)
    
    print(f"Training Accs are: {train_accs}")
    print(f"Test Accs are: {test_accs}")

if __name__ == "__main__":
    main()