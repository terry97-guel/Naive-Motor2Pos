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
import sys

# fix random seeds for reproducibility
SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)

def TrainEpoch(model, optimizer, input, label):
    # forward model
    output = model(input)
    # get posLoss
    posLoss = torch.nn.MSELoss()(output,label)
    # sum total loss
    total_loss = posLoss
    # Optimizer Step
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    return posLoss

def TestEpoch(model, input, label):
    with torch.no_grad():
        # forward model
        output = model(input)
        # get posLoss
        posLoss = torch.nn.MSELoss()(output,label)
        return posLoss

def main(args):
    ### Set logger ###
    if args.WANDB:
        wandb.init(project = args.pname)
        wandb.run.name = os.path.split(args.saveDir)[-1]
        
    ### Print args ###
    print("Reading args...")
    command = ' '.join(sys.argv)
    print(command)

    device = torch.device('cuda:'+args.device)
    torch.cuda.set_device(device)

    ### Set model ###
    IOScale = JsonDataset(os.path.join(args.dataPath,'trainInterpolate.json')).GetIOScale(device)
    input_dim,output_dim = JsonDataset(os.path.join(args.dataPath,'trainInterpolate.json')).GetDataDimension()
    model = Model(input_dim,output_dim,args.nLayers,args.activation)

    ### Load weight when requested ###
    if os.path.isfile(args.resumeDir):
        weight = torch.load(args.resumeDir,map_location=device)
        model.load_state_dict(weight['state_dict'])
        IOScale = weight['IOScale']
        print("loading successful!")
    else:
        print("Nothing to load, Starting from scratch")
    model.IOScale = IOScale
    model = model.to(device)
    
    ### Set optimizer ###
    optimizer = torch.optim.Adam(model.parameters(),lr= args.lr, weight_decay=args.wd, betas=(0.5,0.9))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                        lr_lambda=lambda epoch: args.lrd ** epoch,
                                        last_epoch=-1,
                                        verbose=False)

    ### Assert path to save model ###
    pathname = args.saveDir
    Path(pathname).mkdir(parents=True, exist_ok=True)

    ### Set dataloader ###
    print("Setting up dataloader")
    trainInterpolation  = JsonDataloader(os.path.join(args.dataPath,'trainInterpolate.json'), args.n_workers, args.batchSize,dataRatio=0.5)
    valInterpolation    = JsonDataloader(os.path.join(args.dataPath,'valInterpolate.json' ), args.n_workers, 8*args.batchSize,dataRatio=1)
    testInterpolation   = JsonDataloader(os.path.join(args.dataPath,'testInterpolate.json' ), args.n_workers, 8*args.batchSize,dataRatio=1)
    testExtrapolation   = JsonDataloader(os.path.join(args.dataPath,'testExtrapolate.json' ), args.n_workers, 8*args.batchSize)

    print("Initalizing Training loop")
    for epoch in range(args.epochs):
        # Timer start #
        if (epoch)%10 == 0:
            time_start = time.time()

        # Train #
        model.train()
        data_length = len(trainInterpolation)
        trainPosLoss = np.array([])
        for iterate, (input,label) in enumerate(trainInterpolation):
            input = input.to(device)
            label = label.to(device)

            posLoss = TrainEpoch(model, optimizer, input, label)

            # metric to plot
            trainPosLoss = np.append(trainPosLoss, posLoss.detach().cpu().numpy())
            print('Epoch:{}, posLoss:{:.2f}, Progress:{:.2f}%'.format(epoch+1,posLoss,100*iterate/data_length), end='\r')

        trainPosLoss, trainMaxPosLoss = trainPosLoss.mean(), trainPosLoss.max()
        print('TrainLoss:{:.2f}'.format(trainPosLoss))
        # Log to wandb #
        if args.WANDB:
            wandb.log({'trainPosLoss':trainPosLoss, 'trainMaxPosLoss':trainMaxPosLoss,
            },step = epoch+1)

        # Evaluate on ValidationSet #
        if (epoch+1)%args.TestPeriod == 0:
            model.eval()
            data_length = len(valInterpolation)
            valPosLoss = np.array([])
            for iterate, (input,label) in enumerate(valInterpolation):
                input = input.to(device)
                label = label.to(device)
                posLoss = TestEpoch(model, input, label)

                # metric to plot
                valPosLoss = np.append(valPosLoss, posLoss.detach().cpu().numpy())
                print('Validatiing...{:.2f} Epoch:{}, Progress:{:.2f}%'.format(posLoss,epoch+1,100*iterate/data_length) , end='\r')
            
            valPosLoss, valMaxPosLoss = valPosLoss.mean(), valPosLoss.max()
            print('ValLoss:{:.2f}'.format(valPosLoss))
            # Log to wandb #
            if args.WANDB:
                wandb.log({'InterpolateValPosLoss':valPosLoss,'InterpolateValMaxPosLoss':valMaxPosLoss,
                },step = epoch+1)
                
        # Evaluate on Test set #
        if (epoch+1)%args.TestPeriod == 0:
            model.eval()
            data_length = len(testInterpolation)
            testPosLoss = np.array([])
            for iterate, (input,label) in enumerate(testInterpolation):
                input = input.to(device)
                label = label.to(device)
                posLoss = TestEpoch(model, input, label)

                # metric to plot
                testPosLoss = np.append(testPosLoss, posLoss.detach().cpu().numpy())
                print('Testing...{:.2f} Epoch:{}, Progress:{:.2f}%'.format(posLoss,epoch+1,100*iterate/data_length) , end='\r')
            
            testPosLoss, testMaxPosLoss = testPosLoss.mean(), testPosLoss.max()
            print('TestLoss:{:.2f}'.format(testPosLoss))
            # Log to wandb #
            if args.WANDB:
                wandb.log({'InterpolateTestPosLoss':testPosLoss,'InterpolateTestMaxPosLoss':testMaxPosLoss,
                },step = epoch+1)
                
        # Evaluate on Extrapolation #
        if (epoch+1)%args.TestPeriod == 0:
            model.eval()
            data_length = len(testExtrapolation)
            testPosLoss = np.array([])
            for iterate, (input,label) in enumerate(testExtrapolation):
                input = input.to(device)
                label = label.to(device)
                posLoss = TestEpoch(model, input, label)

                # metric to plot
                testPosLoss = np.append(testPosLoss, posLoss.detach().cpu().numpy())
                print('Testing...{:.2f} Epoch:{}, Progress:{:.2f}%'.format(posLoss,epoch+1,100*iterate/data_length) , end='\r')

            testPosLoss, testMaxPosLoss = testPosLoss.mean(), testPosLoss.max()
            print('TestLoss:{:.2f}'.format(testPosLoss))
            # Log to wandb #
            if args.WANDB:
                wandb.log({'ExtraploateTestPosLoss':testPosLoss,'ExtraploateTestMaxPosLoss':testMaxPosLoss,
                },step = epoch+1)
        
        # Timer end #
        if (epoch+1)%args.TestPeriod == 0:    
            time_end = time.time()
            avg_time = time_end-time_start
            eta_time = (args.epochs - epoch)/args.TestPeriod * avg_time
            h = int(eta_time //3600)
            m = int((eta_time %3600)//60)
            s = int((eta_time %60))
            print("Epoch: {}, eta:{}:{}:{}".format(epoch+1, h,m,s))
        
            # Log to wandb #
            if args.WANDB:
                wandb.log({'TimePerEpoch':avg_time},step = epoch+1)

        #save model #
        if (epoch+1) % args.savePeriod==0:
            filename =  pathname + '/checkpoint_{}.pth'.format(epoch+1)
            print("saving... {}".format(filename))
            state = {
                'state_dict':model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'input_dim':input_dim,
                'IOScale':model.IOScale,
                "nLayers":args.nLayers
            }
            # torch.save(state, filename)

        scheduler.step()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description= 'parse for POENet')
    args.add_argument('--batchSize', default= 32, type=int,
                    help='batch_size')
    args.add_argument('--dataPath', default= './data/Sorosim4dof',type=str,
                    help='path to data')
    args.add_argument('--saveDir', default= './output/temp',type=str,
                    help='path to save model')
    args.add_argument('--resumeDir', default= './output/',type=str,
                    help='path to load model')
    args.add_argument('--device', default= '1',type=str,
                    help='device to use')
    args.add_argument('--n_workers', default= 2, type=int,
                    help='number of data loading workers')
    args.add_argument('--nLayers', default= 1, type=int,
                    help='number layers')
    args.add_argument('--activation', default= 'LRELU',type=str,
                    help='')
    args.add_argument('--wd', default= 0.001, type=float,
                    help='weight_decay for model layer')
    args.add_argument('--lrd', default= 0.95, type=float,
                    help='weight_decay for model layer')
    args.add_argument('--lr', default= 0.01, type=float,
                    help='learning rate for model layer')
    args.add_argument('--WANDB', action = 'store_true', help = 'Use wandb to log')
    args.add_argument('--epochs', default= 100, type=int,
                    help='number of epoch to perform')
    args.add_argument('--savePeriod', default= 50, type=int,
                    help='number of scenes after which model is saved')
    args.add_argument('--TestPeriod', default= 1, type=int,
                    help='number of scenes after which model is Tested')
    args.add_argument('--pname', default= 'NaiveNN with Sorosim Data',type=str,
                    help='Project name')
    
    args = args.parse_args()
    main(args)
#%%
