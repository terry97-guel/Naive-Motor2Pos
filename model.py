#%%
from curses.ascii import NL
import torch.nn as nn
import torch

class Model(nn.Module):
    def __init__(self,inputdim,outputdim,nLayers,activation):
        super(Model,self).__init__()
        self.q_layer = q_layer(inputdim,nLayers,activation)
        for i in range(outputdim):
            setattr(self,"joint{}".format(i+1),Joint(inputdim,nLayers,activation))
        self.nJoint = outputdim
    def forward(self,motor_control):
        batchSize = motor_control.size()[0]
        device = motor_control.device
        
        qExpand = self.q_layer(motor_control)
        jointPositions = torch.zeros(batchSize,self.nJoint,3).to(device)
        for i in range(self.nJoint):
            joint = getattr(self,"joint{}".format(i+1))
            out = joint(qExpand)
            jointPositions[:,i,:] = out
        return jointPositions

class Joint(nn.Module):
    def __init__(self,inputdim,nLayers,activation):
        super(Joint, self).__init__()
        inputdim = inputdim * (4**nLayers)
        LayerList = []
        
        ## Decode
        for _ in range(nLayers):
            # set FC layer
            layer = nn.Linear(inputdim,inputdim//2)
            torch.nn.init.xavier_uniform_(layer.weight)
            # append FC layer #
            LayerList.append(layer)
            # add bn layer #
            if activation == 'LRELU': LayerList.append(torch.nn.BatchNorm1d(inputdim//2))
            # add ac layer #
            if activation == 'LRELU': LayerList.append(torch.nn.LeakyReLU())
            elif activation == 'SIGMOID': LayerList.append(torch.nn.Sigmoid())
            else: raise Exception("Not Valid activation")
            # reduce dim #
            inputdim = inputdim // 2
        layer = nn.Linear(inputdim,3)
        torch.nn.init.xavier_uniform_(layer.weight)
        LayerList.append(layer)
        
        self.layers = torch.nn.Sequential(*LayerList)
        

    def forward(self, motor_control):
        out = self.layers(motor_control)
        return out

class q_layer(nn.Module):
    def __init__(self,inputdim,n_layers=4,activation='SIGMOID'):
        super(q_layer, self).__init__()
        LayerList = []
        for _ in range(n_layers):
            # set FC layer #
            layer = nn.Linear(inputdim,4*inputdim)
            torch.nn.init.xavier_uniform_(layer.weight)
            # append FC layer #
            LayerList.append(layer)
            ### add bn layer ###
            if activation == 'LRELU':  LayerList.append(torch.nn.BatchNorm1d(inputdim*4))
            # add ac layer #
            if activation == 'LRELU': LayerList.append(torch.nn.LeakyReLU())
            elif activation == 'SIGMOID': LayerList.append(torch.nn.Sigmoid())
            else: raise Exception("Not Valid activation")
            # increse dim #
            inputdim = inputdim * 4
        self.layers = torch.nn.Sequential(*LayerList)

    def forward(self, motor_control):
        qExpand = self.layers(motor_control)
        return qExpand
    
#%%