import torch 
import torch.nn as nn

from tqdm import tqdm 

def train(dataLoader, model, optimizer, device):
    """
    This function does the training for one epoch.
    :param dataLoader: this is the pytorch dataloader
    :param model: pytorch model
    :param optimizer: optimizer, e.g. adam, sgd, ...
    :param device: cuda/cpu 
    """
    
    # put the model in train mode
    model.train()
    # go over every batch of data in dataloader
    for data in dataLoader:
        inputs = data['image']
        targets = data['targets']
        # move input/targets to cuda/cpu device 
        inputs = inputs.to(device, dtype=torch.float)
        targets = targets.to(device, dtype=torch.float)
        # zero grad the optimzer 
        optimizer.zero_grad()
        # forward step of the model 
        outputs = model(inputs)
        # calc. losss
        loss = nn.BCEWithLogitsLoss()(outputs, targets.view(-1,1))
        # backward step of the loss
        loss.backward()
        # step optimizer
        optimizer.step()
    
def evaluate(dataLoader, model, device):
    """
    This function does the evaluation for one epoch.
    :param dataLoader: this is the pytorch dataloader
    :param model: pytorch model
    :param device: cuda/cpu 
    """
    # put the model in eval mode 
    model.eval()
    # init lists to store targets and outputs 
    finalTargets = []
    finalOutputs = []
    # we use no_grad context as we dont want variables to have gradients 
    with torch.no_grad():
        for data in dataLoader:
            inputs = data['image']
            targets = data['targets']
            inputs = inputs.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.float)
            # forward step to generate pred 
            output = model(inputs)
            # convert targets and ouytputs to list 
            targets = targets.detach().cpu().numpy().tolist()
            output = output.detach().cpu().numpy().tolist()
            # extend the original list 
            finalTargets.extend(targets)
            finalOutputs.extend(output)
    # return final output and final targets 
    return finalOutputs, finalTargets