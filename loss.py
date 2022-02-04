#%%
import torch



def Pos_norm2(output, label):
    loss = torch.nn.MSELoss()(output,label)
    return loss


