import os
import argparse

# Parse arguments early to determine the GPU to use
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true', default=True)
parser.add_argument('--gpu', type=int, default=0)  # New argument for GPU selection
parser.add_argument('--trainbatchsize', type=int, default=8)
parser.add_argument('--validationbatchsize', type=int, default=1)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--threads', type=int, default=1)
parser.add_argument('--img_size', type=int, default=512)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--colordim', type=int, default=3)
parser.add_argument('--pretrained', action='store_true', default=False)
parser.add_argument('--pretrain_net', type=str, default='')
parser.add_argument('--start_epoch', type=int, default=1)
parser.add_argument('--root_dataset', type=str, default='./wbf_data')
parser.add_argument('--optim', type=str, default='sgd')
parser.add_argument('--num_class', type=int, default=9)
parser.add_argument('--checkpoint', type=str, default='./checkpoint')
parser.add_argument('--target_mode', type=str, default='seg')
parser.add_argument('--root_result', type=str, default='./result')
args = parser.parse_args()

# Set CUDA environment variables at the very top
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import numpy as np
import random
import torch
from torchsummary import summary
import torch.nn as nn
from torch.optim import SGD, Adam, ASGD, Adamax, Adadelta, Adagrad, RMSprop
from torch.utils.data import DataLoader
from LQYDataLoader import get_training_set, get_test_set
import time

from SolarNet import SolarNet

# (Keep the rest of your code as it is)
# But modify how you move data and model to the device

def train(epoch, args, model, training_data_loader, optimizer, device):
    model.train()
    epoch_loss = 0
    criterion = nn.BCELoss()
    criterion2 = nn.NLLLoss()
    
    for batch in training_data_loader:
        input, target = batch[0].to(device), batch[1].to(device)
        
        target1 = target[:, 0, :, :]
        target2 = target[:, 1, :, :].long()
        target3 = target[:, 2, :, :].long()
        
        optimizer.zero_grad()
        output = model(input)
        output1, output2, output3 = output[0], output[1], output[2]
        
        loss1 = criterion(output1, target1.unsqueeze(1).float())
        loss2 = criterion2(torch.log_softmax(output2, dim=1), target2)
        loss3 = criterion2(torch.log_softmax(output3, dim=1), target3)
        loss = 0.1 * loss1 + loss2 + loss3
        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    return epoch_loss / len(training_data_loader)

def test(args, model, testing_data_loader, optimizer, num_class, device):
    model.eval()
    total_loss = 0
    criterion = nn.BCELoss()
    criterion2 = nn.NLLLoss()
    
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    intersection_meter3 = AverageMeter()
    union_meter3 = AverageMeter()
    
    with torch.no_grad():
        for batch in testing_data_loader:
            input, target = batch[0].to(device), batch[1].to(device)
            
            target1 = target[:, 0, :, :]
            target2 = target[:, 1, :, :].long()
            target3 = target[:, 2, :, :].long()
            
            output = model(input)
            output1, output2, output3 = output[0], output[1], output[2]
            
            loss1 = criterion(output1, target1.unsqueeze(1).float())
            loss2 = criterion2(torch.log_softmax(output2, dim=1), target2)
            loss3 = criterion2(torch.log_softmax(output3, dim=1), target3)
            loss = 0.1 * loss1 + loss2 + loss3
            total_loss += loss.item()
            
            pred2 = torch.argmax(output2, dim=1).cpu().numpy()
            pred3 = torch.argmax(output3, dim=1).cpu().numpy()
            target2_np = target2.cpu().numpy()
            target3_np = target3.cpu().numpy()
            
            intersection, union = intersectionAndUnion(pred2, target2_np, 6)
            intersection3, union3 = intersectionAndUnion(pred3, target3_np, num_class)
            
            intersection_meter.update(intersection)
            union_meter.update(union)
            intersection_meter3.update(intersection3)
            union_meter3.update(union3)
    
    avg_test_loss = total_loss / len(testing_data_loader)
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    iou3 = intersection_meter3.sum / (union_meter3.sum + 1e-10)
    return avg_test_loss, iou, iou3

def main(args):
    print("here1")
    print(torch.cuda.is_available())
    print("here2")
    if args.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    print("here3")
    
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    print('===> Loading datasets')
    train_set = get_training_set(args.root_dataset, args.img_size, target_mode=args.target_mode, colordim=args.colordim)
    test_set = get_test_set(args.root_dataset, args.img_size, target_mode=args.target_mode, colordim=args.colordim)
    
    training_data_loader = DataLoader(train_set, batch_size=args.trainbatchsize, shuffle=True, num_workers=args.threads)
    testing_data_loader = DataLoader(test_set, batch_size=args.validationbatchsize, shuffle=False, num_workers=args.threads)
    
    device = torch.device("cuda:0" if args.cuda else "cpu")
    model = SolarNet(in_channels=args.colordim, n_class=args.num_class).to(device)
    
    if args.pretrained:
        model.load_state_dict(torch.load(args.pretrain_net))
    
    optimizer = key2opt[args.optim](model.parameters(), lr=args.learning_rate)

    summary(model, input_size=(args.colordim, 512, 512))

    print('===> Training model')
    test_iou = -0.1
    
    for epoch in range(args.start_epoch, args.start_epoch + args.epochs + 1):
        start = time.time()
        train_loss = train(epoch, args, model, training_data_loader, optimizer, device)
        train_time = time.time() - start
        
        avg_test_loss, iou, iou3 = test(args, model, testing_data_loader, optimizer, args.num_class, device)
        test_time = time.time() - start - train_time
        
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Test Loss={avg_test_loss:.4f}")
        print(f"Train Time={train_time:.2f}s, Test Time={test_time:.2f}s")
        print(f"IoU (Orientation)={iou.mean():.4f}, IoU (Superstructures)={iou3.mean():.4f}")
        
        f1 = iou.mean() + iou3.mean()
        if f1 > test_iou:
            test_iou = f1
            checkpoint(epoch, args, model)
        
        with open(os.path.join(args.root_result, 'accuracy.txt'), 'a') as f:
            f.write(f"{epoch}\t{train_loss:.4f}\t{avg_test_loss:.4f}\t{iou.mean():.4f}\t{iou3.mean():.4f}\n")

if __name__ == '__main__':
    args.checkpoint += f'-batchsize{args.trainbatchsize}-learning_rate{args.learning_rate}-optimizer{args.optim}'
    os.makedirs(args.checkpoint, exist_ok=True)
    os.makedirs(args.root_result, exist_ok=True)
    print("here0")
    main(args)
