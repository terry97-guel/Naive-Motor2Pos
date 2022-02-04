#%%
import argparse
import torch
import numpy as np
from dataloader import *
from model import Model
from loss import *
import os
import random
from pathlib import Path
import wandb
import time
from utils.pyart import *

# fix random seeds for reproducibility
SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)
            
    
def train_epoch(model, optimizer, input, label,Loss_Fn, args):
    # forward model
    output = model(input)
    
    # get Pos_loss
    Pos_loss = Loss_Fn(output,label)

    # sum total loss
    total_loss = Pos_loss
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return Pos_loss

def test_epoch(model, input, label, Loss_Fn, args):
    # forward model
    output = model(input)
    
    # get Pos_loss
    Pos_loss = Loss_Fn(output,label)

    return Pos_loss

def main(args):
    #set logger
    if args.wandb:
        wandb.init(project = args.pname)
        wandb.run.name = os.path.split(args.save_dir)[-1]
        
    if torch.cuda.is_available():
        #set device
        # os.environ["CUDA_VISIBLE_DEVICES"]=args.device
        device = torch.device('cuda:'+args.device)
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu:0')
        torch.cuda.set_device(device)

    #set model
    model = Model(args.input_dim)
    model = model.to(device)

    #load weight when requested
    if os.path.isfile(args.resume_dir):
        weight = torch.load(args.resume_dir)
        model.load_state_dict(weight['state_dict'])
        print("loading successful!")
    else:
        print("Nothing to load, Starting from scratch")

    #set optimizer
    optimizer = torch.optim.Adam(model.parameters(),lr= args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                        lr_lambda=lambda epoch: 0.95 ** epoch,
                                        last_epoch=-1,
                                        verbose=False)

    #declare loss function
    if args.loss_function == 'Pos_norm2':
        Loss_Fn = Pos_norm2
    else:
        print("Invalid loss_function")
        exit(0)
    

    #assert path to save model
    pathname = args.save_dir
    Path(pathname).mkdir(parents=True, exist_ok=True)

    #set dataloader
    print("Setting up dataloader")
    train_data_loader = FoldToyDataloader(args.data_path, args.Foldstart, args.Foldend, args.n_workers, args.batch_size)
    test_data_loader = FoldToyDataloader(args.data_path, args.Foldend, -1, args.n_workers, args.batch_size)

    # Ask Jihoon
    # train_data_loader.dataset.input = train_data_loader.dataset.input.to(device)
    # train_data_loader.dataset.label = train_data_loader.dataset.label.to(device)

    # test_data_loader.dataset.input  = test_data_loader.dataset.input.to(device)
    # test_data_loader.dataset.label  = test_data_loader.dataset.label.to(device)


    #set inital Joint position
    # _,label = next(iter(train_data_loader))
    # model.init_pJoint(label)

    print("Initalizing Training loop")
    for epoch in range(args.epochs):
        # Timer start
        time_start = time.time()

        # Train
        model.train()
        data_length = len(train_data_loader)
        train_loss = np.array([])
        for iterate, (input,label) in enumerate(train_data_loader):
            input = input.to(device)
            label = label.to(device)
            # if iterate%2 == 0:
            #     for param in model.q_layer.parameters():
            #         param.requires_grad_ = False
            # else:
            #     for param in model.q_layer.parameters():
            #         param.requires_grad_ = True
            Pos_loss = train_epoch(model, optimizer, input, label, Loss_Fn, args)
            train_loss = np.append(train_loss, Pos_loss.detach().cpu().numpy())
            print('Epoch:{}, Pos_loss:{:.2f}, Progress:{:.2f}%'.format(epoch+1,Pos_loss,100*iterate/data_length), end='\r')
        
        train_loss = train_loss.mean()
        print('TrainLoss:{:.2f}'.format(train_loss))
        
        #Evaluate
        model.eval()
        data_length = len(test_data_loader)

        avg_Pos_loss = np.array([])
        for iterate, (input,label) in enumerate(test_data_loader):
            input = input.to(device)
            label = label.to(device)
            Pos_loss = test_epoch(model, input, label, Loss_Fn, args)

            # metric to plot
            avg_Pos_loss = np.append(avg_Pos_loss, Pos_loss.detach().cpu().numpy())
            print('Testing...{:.2f} Epoch:{}, Progress:{:.2f}%'.format(Pos_loss,epoch+1,100*iterate/data_length) , end='\r')
        

        avg_Pos_loss = avg_Pos_loss.mean()
        print('avg_Pos_loss:{:.2f}'.format(avg_Pos_loss))

        # Timer end    
        time_end = time.time()
        avg_time = time_end-time_start
        eta_time = (args.epochs - epoch) * avg_time
        h = int(eta_time //3600)
        m = int((eta_time %3600)//60)
        s = int((eta_time %60))
        print("Epoch: {}, Pos_loss:{:.2f}, eta:{}:{}:{}".format(epoch+1, avg_Pos_loss, h,m,s))
        
        # Log to wandb
        if args.wandb:
            wandb.log({'TrainLoss':train_loss, 'TimePerEpoch':avg_time,
            'avg_Pos_loss':avg_Pos_loss},step = epoch+1)

        #save model 
        if (epoch+1) % args.save_period==0:
            filename =  pathname + '/checkpoint_{}.pth'.format(epoch+1)
            print("saving... {}".format(filename))
            state = {
                'state_dict':model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'input_dim':args.input_dim
            }
            torch.save(state, filename)

        scheduler.step()
        # schedule args.Vec_loss
        # args.Vec_loss = args.Vec_loss * 0.95


if __name__ == '__main__':
    args = argparse.ArgumentParser(description= 'parse for POENet')
    args.add_argument('--batch_size', default= 1024*4, type=int,
                    help='batch_size')
    args.add_argument('--data_path', default= './data/Multi_2dim_log_spiral',type=str,
                    help='path to data')
    args.add_argument('--save_dir', default= './output/temp',type=str,
                    help='path to save model')
    args.add_argument('--resume_dir', default= './output/',type=str,
                    help='path to load model')
    args.add_argument('--device', default= '1',type=str,
                    help='device to use')
    args.add_argument('--n_workers', default= 2, type=int,
                    help='number of data loading workers')
    args.add_argument('--wd', default= 0.001, type=float,
                    help='weight_decay for model layer')
    args.add_argument('--lr', default= 0.01, type=float,
                    help='learning rate for model layer')
    args.add_argument('--loss_function', default= 'Pos_norm2', type=str,
                    help='get list of loss function')
    args.add_argument('--wandb', action = 'store_true', help = 'Use wandb to log')
    args.add_argument('--input_dim', default= 2, type=int,
                    help='dimension of input')
    args.add_argument('--epochs', default= 100, type=int,
                    help='number of epoch to perform')
    args.add_argument('--save_period', default= 1, type=int,
                    help='number of scenes after which model is saved')
    args.add_argument('--pname', default= 'Naive2posFoldtest',type=str,
                    help='Project name')
    args.add_argument('--Foldstart', default= 0, type=int,
                    help='Number of Fold to start')
    args.add_argument('--Foldend', default= 8, type=int,
                    help='Number of Fole to end')
    args = args.parse_args()
    main(args)
#%%
