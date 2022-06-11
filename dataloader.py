#%%
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import random
import os 
import json

class JsonDataset(Dataset):
    def __init__(self,dataPath,dataRatio=1):
        with open(dataPath,'r') as RawData:
            rawData = json.load(RawData)
        rawData = rawData['data']
        label, input = [],[]
        for rawDatum in rawData:
            input.append(rawDatum['actuation'])
            label.append(rawDatum['position'])
        
        label = np.array(label)
        input = np.array(input)
        
        dataNum = int(len(label)*dataRatio)
        label,input = label[:dataNum],input[:dataNum]
        self.label, self.input =torch.tensor(1000*label).to(torch.float), torch.tensor(input).to(torch.float)
        
    def __len__(self):
        return len(self.input)
    
    def __getitem__(self,idx):
        return self.input[idx], self.label[idx]

    def GetIOScale(self,device):
        (inputVar,inputMean),(outputVar,outputMean) = torch.var_mean(self.input,0), torch.var_mean(self.label.reshape(-1,3), dim=0)
        inputVar,inputMean,outputVar,outputMean = inputVar.to(device),inputMean.to(device),outputVar.to(device),outputMean.to(device)
        
        # Adjust small variance #
        if (inputVar<1e-9).any():
            print("Warning: inputVar is less than 1e-9... Fixing it to 1")
            inputVar[inputVar<1e-9]   =  1
        if (outputVar<1e-9).any():
            print("Warning: outputVar is less than 1e-9... Fixing it to 1")
            outputVar[outputVar<1e-9] =  1

        inputSigma, outputSigma = torch.sqrt(inputVar), torch.sqrt(outputVar)
        IOScale = {'IScale':(inputMean,inputSigma), 'OScale':(outputMean,outputSigma)}
        
        return IOScale

    def GetDataDimension(self):
        inputDim = self.input.size()[-1]
        outputDim = self.label.size()[1]
        assert self.label.size()[-1] == 3
        return (inputDim,outputDim)
    
class JsonDataloader(DataLoader):
    def __init__(self,data_path, n_workers,batch, shuffle = True,dataRatio=1):
        self.dataset = JsonDataset(data_path,dataRatio)
        super().__init__(self.dataset, batch_size=batch, shuffle=shuffle, num_workers=n_workers)

# class ToyDataset(Dataset):
#     def __init__(self,data_path,IOScale=None):

#         if os.path.isdir(data_path):
#             Scenario_ls = os.listdir(data_path)
#             self.label = torch.tensor([])
#             self.input = torch.tensor([])

#             for scenario in Scenario_ls:
#                 file_path = data_path + '/' + scenario
#                 rawdata = np.loadtxt(file_path, delimiter=',')
#                 self.label = torch.cat((self.label,torch.Tensor(rawdata[:,3:])),0)
#                 self.input = torch.cat((self.input,torch.Tensor(rawdata[:,:3])),0)
                
#         elif os.path.isfile(data_path):
#             file_path = data_path
#             rawdata = np.loadtxt(file_path)
#             self.label = torch.Tensor(rawdata[:,3:])
#             self.input = torch.Tensor(rawdata[:,:3])
        

#         if IOScale != None:
            
#             (inputSigma,inputMean),(outputSigma,outputMean) = IOScale
#             inputSigma,inputMean,outputSigma,outputMean = inputSigma.detach(),inputMean.detach(),outputSigma.detach(),outputMean.detach()
            
#             #repeat
#             outputMean = outputMean.tile(self.label.size()[-1]//3)
#             outputSigma  = outputSigma.tile(self.label.size()[-1]//3)
            
#             self.input = 10*(self.input - inputMean)/inputSigma
#             self.label = 10*(self.label - outputMean)/outputSigma


#     def __len__(self):
#         return len(self.input)
    
#     def __getitem__(self,idx):
#         return self.input[idx], self.label[idx]
    
#     def getIOScale(self):
#         (inputVar,inputMean),(outputVar,outputMean) = (torch.var_mean(self.input,0), torch.var_mean(self.label[:,-3:],0))
        
#         inputVar[inputVar<1e-9]   =  1
#         outputVar[outputVar<1e-9] =  1

#         inputSigma, outputSigma = torch.sqrt(inputVar), torch.sqrt(outputVar)

        
#         return ((inputSigma,inputMean),(outputSigma,outputMean))

# class ToyDataloader(DataLoader):
#     def __init__(self,data_path, IOScale, n_workers,batch, shuffle = True):
#         self.dataset = ToyDataset(data_path, IOScale)
#         super().__init__(self.dataset, batch_size=batch, shuffle=shuffle, num_workers=n_workers)

# class FoldToyDataset(Dataset):
#     def __init__(self,data_path,Foldstart,Foldend):
#         self.label = torch.tensor([])
#         self.input = torch.tensor([])

#         path_list = os.listdir(data_path)
#         path_list.sort()

#         for path in path_list[Foldstart:Foldend]:
#             path_temp = data_path + '/' + path
#             if os.path.isdir(path_temp):
#                 Scenario_ls = os.listdir(path_temp)
                
#                 for scenario in Scenario_ls:
#                     file_path = path_temp + '/' + scenario
#                     assert(os.path.isfile(file_path))
#                     rawdata = np.loadtxt(file_path)
#                     self.label = torch.cat((self.label,torch.Tensor(rawdata[:,:-2])),0)
#                     self.input = torch.cat((self.input,torch.Tensor(rawdata[:,-2:])),0)

#     def __len__(self):
#         return len(self.input)
    
#     def __getitem__(self,idx):
#         return self.input[idx], self.label[idx]

# class FoldToyDataloader(DataLoader):
#     def __init__(self,data_path, Foldstart, Foldend, n_workers,batch, shuffle = True):
#         self.dataset = FoldToyDataset(data_path,Foldstart,Foldend)
#         super().__init__(self.dataset, batch_size=batch, shuffle=shuffle, num_workers=n_workers)
